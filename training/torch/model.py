"""The model: a gated bag of hashed word features, a small residual layer, and a word table."""

import torch
import torch.nn.functional as F
from torch import nn

from export import quantize_features, quantize_words


class IconWordModel(nn.Module):
    """Score every word of the word list for each query.

    A word of the query is the mean of the learned vectors of its feature rows.
    A learned gate weighs each word, so that filler words count less.
    The weighted mean of the words goes through one residual layer.
    The score of a target word is the cosine between that result and the row of the word.
    """

    def __init__(self, rows: int, words: int, dim: int, hidden: int):
        super().__init__()
        self.features = nn.EmbeddingBag(rows, dim, mode="mean")
        self.gate = nn.Linear(dim, 1)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.words = nn.Parameter(torch.randn(words, dim) * 0.1)
        self.log_scale = nn.Parameter(torch.tensor(3.0))
        self.quantize = False
        """Round the two large tables to their 6-bit storage format in the forward pass."""
        nn.init.normal_(self.features.weight, std=0.1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def encode(
        self, rows: torch.Tensor, offsets: torch.Tensor, word_query: torch.Tensor, queries: int
    ) -> torch.Tensor:
        """Return one unit vector per query.

        `rows` and `offsets` describe the feature rows of each query word, as `nn.EmbeddingBag` expects.
        `word_query` gives the query that each word belongs to.
        """
        table = quantize_features(self.features.weight) if self.quantize else self.features.weight
        word = F.embedding_bag(rows, table, offsets, mode="mean")
        weight = torch.sigmoid(self.gate(word))
        total = torch.zeros(queries, word.shape[1], device=word.device).index_add_(
            0, word_query, weight * word
        )
        norm = torch.zeros(queries, 1, device=word.device).index_add_(0, word_query, weight)
        pooled = total / (norm + 1e-6)
        hidden = pooled + self.fc2(F.relu(self.fc1(pooled)))
        return F.normalize(hidden, dim=-1)

    def forward(
        self, rows: torch.Tensor, offsets: torch.Tensor, word_query: torch.Tensor, queries: int
    ) -> torch.Tensor:
        """Return the scaled cosine of each query with each word of the word list."""
        words = quantize_words(self.words) if self.quantize else self.words
        units = self.encode(rows, offsets, word_query, queries)
        return self.log_scale.exp() * units @ F.normalize(words, dim=-1).T
