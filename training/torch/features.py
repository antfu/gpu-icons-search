"""Query words to rows of the feature table. Mirrors `src/features.ts`; a test compares both."""

import re

import numpy as np

MAX_WORD_LENGTH = 24
MAX_QUERY_WORDS = 12
KIND_WORD = 0
KIND_PREFIX = 1
KIND_PIECE = 3
MIN_DELETION_LENGTH = 3
PIECE_LENGTH = 3

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


def _codes(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the character codes of each text in one padded matrix, and the length of each text."""
    lengths = np.array([len(text) for text in texts], dtype=np.int64)
    codes = np.zeros((len(texts), int(lengths.max()) if texts else 0), dtype=np.uint32)
    for row, text in enumerate(texts):
        codes[row, : len(text)] = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    return codes, lengths


def _hash_words(codes: np.ndarray, lengths: np.ndarray, skip: int = -1) -> np.ndarray:
    """Hash each whole word, without the character at index `skip`."""
    value = _start(KIND_WORD, len(lengths))
    for index in range(codes.shape[1]):
        if index != skip:
            value = np.where(index < lengths, (value ^ codes[:, index]) * _PRIME, value)
    return _mix(value)


def word_features(words: list[str], rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Hash each word to its feature rows.

    Returns the rows of all words in one flat array, and the start offset of each word in it.
    The features of a word are the whole word, the word with each single letter removed,
    every prefix of two or more letters, and every piece of three characters of `<word>`.
    """
    count = len(words)
    codes, lengths = _codes(words)
    width = codes.shape[1]
    columns = [_hash_words(codes, lengths)]
    valid = [np.ones(count, dtype=bool)]
    for skip in range(width):
        columns.append(_hash_words(codes, lengths, skip))
        valid.append((skip < lengths) & (lengths >= MIN_DELETION_LENGTH))

    prefix = _start(KIND_PREFIX, count)
    for index in range(width):
        prefix = (prefix ^ codes[:, index]) * _PRIME
        if index >= 1:
            columns.append(_mix(prefix))
            valid.append(index < lengths)

    padded, padded_lengths = _codes([f"<{word}>" for word in words])
    for start in range(max(0, padded.shape[1] - PIECE_LENGTH + 1)):
        value = _start(KIND_PIECE, count)
        for index in range(start, start + PIECE_LENGTH):
            value = (value ^ padded[:, index]) * _PRIME
        columns.append(_mix(value))
        valid.append(start + PIECE_LENGTH <= padded_lengths)

    table = np.stack(columns, axis=1) & np.uint32(rows - 1)
    mask = np.stack(valid, axis=1)
    offsets = np.concatenate([[0], np.cumsum(mask.sum(axis=1))[:-1]]).astype(np.int64)
    return table[mask].astype(np.int64), offsets
