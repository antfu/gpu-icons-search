import json
from pathlib import Path

from features import split_query, word_features

FIXTURE = Path(__file__).parent.parent.parent / "test" / "fixtures" / "features.json"


def test_split_query_matches_typescript():
    assert split_query("  Arrow-Up_right  2x!") == ["arrow", "up", "right", "2x"]
    assert split_query("a" * 40) == ["a" * 24]
    assert len(split_query("a b c d e f g h i j k l m n o")) == 12


def test_word_features_match_typescript():
    fixture = json.loads(FIXTURE.read_text())
    words = list(fixture["words"])
    rows, offsets = word_features(words, fixture["rows"])
    ends = [*offsets[1:], len(rows)]
    for word, start, end in zip(words, offsets, ends, strict=True):
        assert sorted(rows[start:end].tolist()) == sorted(fixture["words"][word]), word
