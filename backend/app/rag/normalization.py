from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
_DIGIT_WORDS = {digit: word for word, digit in _NUMBER_WORDS.items()}


def search_terms(value: str, *, keep_single: bool = False) -> set[str]:
    """Return literal and conservative normalized variants for retrieval.

    The same function is used by candidate scoring and evidence filtering so a
    valid match cannot be accepted at one stage and rejected at the next merely
    because a document uses ``months`` while the user wrote ``month`` or uses
    ``three`` while the user wrote ``3``.
    """
    terms: set[str] = set()
    for raw in _TOKEN_RE.findall(value):
        token = raw.casefold()
        pieces = [token, *re.split(r"[._/:#-]+", token)]
        for piece in pieces:
            if not piece:
                continue
            variants = {piece, canonical_token(piece)}
            if piece in _NUMBER_WORDS:
                variants.add(_NUMBER_WORDS[piece])
            if piece in _DIGIT_WORDS:
                variants.add(_DIGIT_WORDS[piece])
            terms.update(
                variant for variant in variants if variant and (keep_single or len(variant) > 1)
            )
    return terms


def canonical_phrase(value: str) -> str:
    """Normalize a phrase while preserving order for heading/phrase matching."""
    return " ".join(
        _NUMBER_WORDS.get(token.casefold(), token.casefold())
        for token in re.findall(r"[A-Za-z0-9]+", value)
        if token
    )


def number_word_variant(value: str) -> str:
    """Expand standalone small integers for semantic query retrieval."""
    tokens = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", value)
    return "".join(_DIGIT_WORDS.get(token, token) for token in tokens)


def canonical_token(token: str) -> str:
    lowered = token.casefold()
    if lowered in _NUMBER_WORDS:
        return _NUMBER_WORDS[lowered]
    return _singular(lowered)


def _singular(token: str) -> str:
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token
