"""Explicit, checkpoint-local phone vocabularies and numbered-pinyin lookup."""

import re
import unicodedata


def normalize_pinyin(syllable):
    """Normalize one numbered syllable; do not infer tones or apply sandhi."""
    if not isinstance(syllable, str):
        raise ValueError("Pinyin syllables must be strings")
    syllable = unicodedata.normalize("NFC", syllable).strip().lower()
    syllable = syllable.replace("u:", "v").replace("ü", "v")
    if syllable.endswith("0"):
        syllable = syllable[:-1] + "5"
    if not re.fullmatch(r"[a-zv]+[1-5]", syllable):
        raise ValueError(f"Expected numbered pinyin with tone 1..5: {syllable!r}")
    return syllable


def pinyin_to_phones(pinyin, lexicon):
    """Expand numbered syllables through the lexicon and literal ``sp`` pauses."""
    syllables = pinyin.split() if isinstance(pinyin, str) else list(pinyin)
    if not syllables:
        raise ValueError("Pinyin must contain at least one numbered syllable")
    phones = []
    for token in syllables:
        if token == "sp":
            phones.append("sp")
            continue
        syllable = normalize_pinyin(token)
        if syllable not in lexicon:
            raise ValueError(f"Unknown pinyin syllable: {syllable!r}")
        entry = lexicon[syllable]
        if not isinstance(entry, (list, tuple)) or not entry:
            raise ValueError(f"Lexicon entry {syllable!r} must be a nonempty phone list")
        if any(not isinstance(phone, str) or not phone or
               any(char.isspace() or char in "{}|" for char in phone) or
               phone == "_" for phone in entry):
            raise ValueError(f"Invalid phones in lexicon entry {syllable!r}")
        phones.extend(entry)
    return phones


def validate_symbols(symbols):
    if not isinstance(symbols, (list, tuple)) or len(symbols) < 2:
        raise ValueError("text.symbols must contain '_' and at least one phone")
    if symbols[0] != "_":
        raise ValueError("text.symbols must reserve index 0 for '_' padding")
    if any(not isinstance(phone, str) or not phone or
           any(char.isspace() or char in "{}|" for char in phone)
           for phone in symbols):
        raise ValueError("text.symbols must contain nonempty, whitespace-free raw phones")
    if len(set(symbols)) != len(symbols):
        raise ValueError("text.symbols contains duplicate phones")
    return list(symbols)


class PhoneVocabulary:
    def __init__(self, symbols):
        self.symbols = validate_symbols(symbols)
        self.phone_to_id = {phone: index for index, phone in enumerate(self.symbols)}

    def __len__(self):
        return len(self.symbols)

    def encode(self, phones):
        if isinstance(phones, str):
            phones = phones.strip()
            if phones.startswith("{") and phones.endswith("}"):
                phones = phones[1:-1]
            phones = phones.split()
        else:
            phones = list(phones)
        if not phones:
            raise ValueError("Phone sequence must not be empty")
        unknown = [phone for phone in phones
                   if phone not in self.phone_to_id or phone == "_"]
        if unknown:
            raise ValueError(f"Unknown or padding phones in utterance: {unknown!r}")
        return [self.phone_to_id[phone] for phone in phones]


def vocabulary_from_config(preprocess_config):
    text = preprocess_config["preprocessing"]["text"]
    symbols = text.get("symbols")
    if symbols is None:
        if text.get("language") == "zh":
            raise ValueError("Chinese preprocessing requires explicit text.symbols")
        return None
    return PhoneVocabulary(symbols)
