"""Produce the distinct token streams used by Nasubi's fitted vectorizers."""

from __future__ import annotations

import re
import threading
import unicodedata
from typing import NamedTuple

import fugashi
import unidic_lite
from sudachipy import Dictionary, SplitMode

_thread = threading.local()
_SEGMENTS = re.compile(r"[^\r\n。！？!?]*[。！？!?]|[^\r\n。！？!?]+")


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sudachi():
    if not hasattr(_thread, "sudachi"):
        _thread.sudachi = Dictionary(dict="core").create()
    return _thread.sudachi


def unidic():
    if not hasattr(_thread, "unidic"):
        # Select the fitted dictionary even when a second UniDic is installed.
        _thread.unidic = fugashi.Tagger(
            f'-r "{unidic_lite.DICDIR}/mecabrc" -d "{unidic_lite.DICDIR}"'
        )
    return _thread.unidic


class Token(NamedTuple):
    surface: str
    lemma: str
    pos: str
    kind: str
    subkind: str
    form: str
    start: int
    end: int


def words(text: str, offset: int = 0) -> list[Token]:
    result = []
    for word in sudachi().tokenize(text, SplitMode.A):
        start, end = word.begin(), word.end()
        surface = text[start:end]
        if surface and not surface.isspace():
            tags = word.part_of_speech()
            result.append(
                Token(
                    normalize(surface),
                    normalize(word.dictionary_form()),
                    *tags[:3],
                    tags[5],
                    offset + start,
                    offset + end,
                )
            )
    return result


def segmented_words(text: str) -> list[Token]:
    result = []
    for match in _SEGMENTS.finditer(text):
        raw = match.group()
        offset = match.start() + len(raw) - len(raw.lstrip())
        block = raw.strip()
        for start in range(0, len(block), 2048):
            result.extend(words(block[start : start + 2048], offset + start))
    return result


class Morph(NamedTuple):
    normalized: str
    reading: str
    pos: str


def morphology(text: str) -> tuple[list[Morph], list[str], list[str]]:
    engine = sudachi()
    fine = [
        Morph(normalize(w.normalized_form()), w.reading_form(), w.part_of_speech()[0])
        for w in engine.tokenize(text, SplitMode.A)
        if w.surface().strip()
    ]
    compounds = [
        normalize(w.normalized_form())
        for w in engine.tokenize(text, SplitMode.C)
        if w.surface().strip()
    ]
    alternate = [normalize(w.feature.lemma or w.surface) for w in unidic()(text)]
    return fine, compounds, alternate
