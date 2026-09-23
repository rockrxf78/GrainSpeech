"""Strict character CTC alignment and Baker pinyin-syllable association."""

import unicodedata

import numpy as np

from text.vocabulary import normalize_pinyin


class AlignmentError(ValueError):
    """A recording cannot be safely included in the aligned training set."""


def hanzi_pinyin(text, pinyin):
    characters = []
    for character in text:
        code = ord(character)
        if 0x3400 <= code <= 0x9FFF or 0x20000 <= code <= 0x2FA1F:
            characters.append(character)
        elif character.isspace() or unicodedata.category(character).startswith("P"):
            continue
        else:
            raise AlignmentError(f"Unsupported non-Hanzi transcript character: {character!r}")
    try:
        syllables = [normalize_pinyin(token) for token in pinyin.split()]
    except ValueError as error:
        raise AlignmentError(str(error)) from error
    if not characters or not syllables:
        raise AlignmentError("Empty Hanzi/pinyin sequence")
    groups = []
    index = 0
    for syllable in syllables:
        if index >= len(characters):
            raise AlignmentError("More pinyin syllables than Hanzi characters")
        start = index
        index += 1
        # The corpus can annotate a Hanzi + erhua suffix as a single syllable.
        if syllable[:-1].endswith("r") and syllable[:-1] != "er":
            if index >= len(characters) or characters[index] not in "\u513f\u5152":
                raise AlignmentError("Erhua annotation has no explicit trailing Hanzi er")
            index += 1
        groups.append((start, index))
    if index != len(characters):
        raise AlignmentError("Hanzi/pinyin counts differ after explicit erhua grouping")
    return characters, syllables, groups


def character_error_rate(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = current
    return previous[-1] / max(1, len(reference))


def viterbi_spans(log_probs, tokens, blank):
    """Return nonblank character spans and posterior scores; never fabricate a path."""
    if log_probs.ndim != 2 or not np.isfinite(log_probs).all():
        raise AlignmentError("Invalid CTC emissions")
    tokens = np.asarray(tokens, dtype=np.int64)
    if not len(tokens) or np.any(tokens == blank):
        raise AlignmentError("Empty target or blank token in CTC target")
    required = len(tokens) + int(np.sum(tokens[1:] == tokens[:-1]))
    if len(log_probs) < required:
        raise AlignmentError("Too few CTC frames for this transcript")
    if tokens.min() < 0 or tokens.max() >= log_probs.shape[1]:
        raise AlignmentError("CTC target token outside vocabulary")
    labels = np.full(2 * len(tokens) + 1, blank, dtype=np.int64)
    labels[1::2] = tokens
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    score = np.full(len(labels), -np.inf, dtype=np.float32)
    score[:2] = log_probs[0, labels[:2]]
    backtrack = np.zeros((len(log_probs), len(labels)), dtype=np.int8)
    for frame in range(1, len(log_probs)):
        advance = np.r_[-np.inf, score[:-1]]
        leap = np.r_[-np.inf, -np.inf, score[:-2]]
        leap[~skip] = -np.inf
        choices = np.stack((score, advance, leap))
        backtrack[frame] = np.argmax(choices, axis=0)
        score = np.max(choices, axis=0) + log_probs[frame, labels]
    state = len(labels) - 1 if score[-1] >= score[-2] else len(labels) - 2
    if not np.isfinite(score[state]):
        raise AlignmentError("No valid complete CTC path")
    path = np.empty(len(log_probs), dtype=np.int64)
    for frame in range(len(log_probs) - 1, -1, -1):
        path[frame] = state
        state -= int(backtrack[frame, state])
    if path[0] not in (0, 1):
        raise AlignmentError("CTC backtracking failed")
    spans = []
    for index, token in enumerate(tokens):
        frames = np.flatnonzero(path == 2 * index + 1)
        if not len(frames):
            raise AlignmentError("A transcript character has no aligned frames")
        confidence = float(np.exp(log_probs[frames, token]).mean())
        spans.append((int(frames[0]), int(frames[-1]), confidence))
    return spans


def syllable_intervals(spans, syllables, groups, seconds_per_frame, center_offset, duration):
    """Place boundaries midway between adjacent CTC character spans.

    CTC blank frames are not silence labels. Internal blanks are apportioned
    at the midpoint, not emitted as fake pauses. These are automatic syllable
    estimates, not ground-truth initial/final boundaries.
    """
    boundaries = [0.0]
    for (_, previous_end), (next_start, _) in zip(groups, groups[1:]):
        left = spans[previous_end - 1][1]
        right = spans[next_start][0]
        boundaries.append((left + right) * seconds_per_frame / 2 + center_offset)
    boundaries.append(duration)
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise AlignmentError("Non-increasing syllable boundaries")
    return [
        {"phone": phone, "start": start, "end": end}
        for phone, start, end in zip(syllables, boundaries, boundaries[1:])
    ]
