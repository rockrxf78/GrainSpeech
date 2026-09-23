"""Shared English/ARPAbet frontend, without acoustic-model dependencies."""

from .symbols import symbols

PAUSE_MARKS = {",", ";", ":", ".", "!", "?"}


def text_to_arpabet(text):
    from g2p_en import G2p

    supported = {symbol[1:] for symbol in symbols if symbol.startswith("@")}
    phones = []
    for token in G2p()(text):
        if token in supported:
            phones.append(token)
        elif token in PAUSE_MARKS and phones and phones[-1] != "sp":
            phones.append("sp")

    while phones and phones[-1] == "sp":
        phones.pop()
    if not phones:
        raise ValueError("The text frontend produced no supported ARPAbet phonemes")
    return phones


def parse_phonemes(value):
    phones = value.strip().removeprefix("{").removesuffix("}").split()
    supported = {symbol[1:] for symbol in symbols if symbol.startswith("@")}
    unsupported = sorted(set(phones) - supported)
    if unsupported:
        raise ValueError(f"Unsupported phonemes: {', '.join(unsupported)}")
    if not phones:
        raise ValueError("No phonemes were provided")
    return phones
