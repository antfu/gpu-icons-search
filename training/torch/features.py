"""Query words to rows of the feature table. Mirrors `src/features.ts`; a test compares both."""

import re

import numpy as np

MAX_WORD_LENGTH = 24
MAX_QUERY_WORDS = 12
PIECE_LENGTHS = (2, 3, 4)

_PRIME = np.uint32(0x01000193)
_WORD = re.compile(r"[a-z0-9]+")


def split_query(query: str) -> list[str]:
    """Split a query into lowercase words of letters and digits."""
    return [word[:MAX_WORD_LENGTH] for word in _WORD.findall(query.lower())[:MAX_QUERY_WORDS]]


def _start(kind: int, count: int) -> np.ndarray:
    seed = ((0x811C9DC5 ^ kind) * 0x01000193) & 0xFFFFFFFF
    return np.full(count, seed, dtype=np.uint32)


def _mix(value: np.ndarray) -> np.ndarray:
    value = value ^ (value >> np.uint32(16))
    value = value * np.uint32(0x85EBCA6B)
    value = value ^ (value >> np.uint32(13))
    value = value * np.uint32(0xC2B2AE35)
    return value ^ (value >> np.uint32(16))


def word_features(words: list[str], rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Hash each word to its feature rows.

    Returns the rows of all words in one flat array, and the start offset of each word in it.
    The features of a word are the whole word and every piece of 2 to 4 characters of `<word>`.
    """
    texts = [f"<{word}>" for word in words]
    count = len(texts)
    lengths = np.array([len(text) for text in texts], dtype=np.int64)
    width = int(lengths.max()) if count else 0
    codes = np.zeros((count, width), dtype=np.uint32)
    for row, text in enumerate(texts):
        codes[row, : len(text)] = np.frombuffer(text.encode("ascii"), dtype=np.uint8)

    columns = []
    valid = []
    whole = _start(0, count)
    for index in range(width):
        whole = np.where(index < lengths, (whole ^ codes[:, index]) * _PRIME, whole)
    columns.append(_mix(whole))
    valid.append(np.ones(count, dtype=bool))
    for length in PIECE_LENGTHS:
        for start in range(max(0, width - length + 1)):
            value = _start(length, count)
            for index in range(start, start + length):
                value = (value ^ codes[:, index]) * _PRIME
            columns.append(_mix(value))
            valid.append(start + length <= lengths)

    table = np.stack(columns, axis=1) & np.uint32(rows - 1)
    mask = np.stack(valid, axis=1)
    counts = mask.sum(axis=1)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return table[mask].astype(np.int64), offsets
