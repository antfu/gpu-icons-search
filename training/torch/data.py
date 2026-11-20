"""Training examples: sample queries, add typing noise, and hash them to feature rows."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from features import MAX_QUERY_WORDS, word_features

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Words that carry no meaning in a description such as "icon for removing an item"
FILLER_BEFORE = [
    "icon",
    "icon for",
    "icon of",
    "an icon for",
    "icon of a",
    "a",
    "an",
    "the",
    "symbol for",
    "button to",
    "button for",
    "picture of a",
    "image of",
    "show",
    "something for",
    "sign for",
    "logo for",
    "i need a",
]
FILLER_AFTER = ["icon", "icons", "button", "symbol", "sign", "logo", "image", "picture", "item", "thing"]


@dataclass
class Batch:
    rows: torch.Tensor
    offsets: torch.Tensor
    word_query: torch.Tensor
    target_ids: torch.Tensor
    target_weights: torch.Tensor


def add_typo(word: str, rng: random.Random) -> str:
    """Apply one typing mistake: swap two letters, drop one, replace one, or insert one."""
    index = rng.randrange(len(word))
    letter = rng.choice(LETTERS)
    kind = rng.randrange(4)
    if kind == 0 and index + 1 < len(word):
        return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    if kind == 1:
        return word[:index] + word[index + 1 :]
    if kind == 2:
        return word[:index] + letter + word[index + 1 :]
    return word[:index] + letter + word[index:]


def add_noise(word: str, rng: random.Random, prefix: float, typo: float) -> str:
    """Cut a word to a prefix, or add typing mistakes, with the given probabilities."""
    if len(word) < 4:
        return word
    draw = rng.random()
    if draw < prefix:
        return word[: rng.randrange(3, len(word))]
    if draw < prefix + typo:
        word = add_typo(word, rng)
        if len(word) >= 8 and rng.random() < 0.3:
            word = add_typo(word, rng)
    return word


class TrainingData:
    """The queries and targets that the TypeScript script `src/targets.ts` wrote."""

    def __init__(self, file: Path, rows: int, seed: int = 0):
        data = json.loads(file.read_text())
        self.vocabulary: list[str] = data["vocabulary"]
        self.rows = rows
        self.rng = random.Random(seed)
        self.texts = [query["text"].split(" ") for query in data["queries"]]
        width = max(len(query["targets"]) for query in data["queries"])
        self.target_ids = np.zeros((len(self.texts), width), dtype=np.int64)
        self.target_weights = np.zeros((len(self.texts), width), dtype=np.float32)
        for index, query in enumerate(data["queries"]):
            for column, (word, weight) in enumerate(query["targets"]):
                self.target_ids[index, column] = word
                self.target_weights[index, column] = weight
        self.target_weights /= self.target_weights.sum(axis=1, keepdims=True)
        weights = np.array([query["weight"] for query in data["queries"]], dtype=np.float64)
        phrase = np.array([len(text) > 1 for text in self.texts])
        # Phrases are many and each one is rare. Give them a fixed share of the examples.
        weights[phrase] *= 0.3 / weights[phrase].sum()
        weights[~phrase] *= 0.7 / weights[~phrase].sum()
        self.cumulative = np.cumsum(weights)

    def _words(self, index: int, prefix: float, typo: float) -> list[str]:
        rng = self.rng
        text = self.texts[index]
        if len(text) == 1:
            words = [add_noise(text[0], rng, prefix, typo)]
        else:
            words = [add_noise(word, rng, 0.0, typo * 0.5) for word in text]
        if rng.random() < 0.15:
            words = rng.choice(FILLER_BEFORE).split(" ") + words
            if rng.random() < 0.4:
                words.append(rng.choice(FILLER_AFTER))
        elif rng.random() < 0.06:
            words.append(rng.choice(FILLER_AFTER))
        return words[:MAX_QUERY_WORDS]

    def encode(self, queries: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hash the words of each query. Return rows, offsets, and the query of each word."""
        words = [word for query in queries for word in query]
        owner = [index for index, query in enumerate(queries) for _ in query]
        rows, offsets = word_features(words, self.rows)
        return torch.from_numpy(rows), torch.from_numpy(offsets), torch.tensor(owner, dtype=torch.int64)

    def sample(self, size: int, prefix: float = 0.25, typo: float = 0.2) -> Batch:
        picks = np.searchsorted(
            self.cumulative, np.random.default_rng(self.rng.randrange(1 << 30)).random(size)
        )
        picks = np.minimum(picks, len(self.texts) - 1)
        rows, offsets, owner = self.encode([self._words(int(index), prefix, typo) for index in picks])
        return Batch(
            rows,
            offsets,
            owner,
            torch.from_numpy(self.target_ids[picks]),
            torch.from_numpy(self.target_weights[picks]),
        )
