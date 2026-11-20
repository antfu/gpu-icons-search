"""Write the weight file that `src/weights.ts` reads, and the fixtures for parity tests."""

import json
from pathlib import Path

import numpy as np
import torch

LEVELS = 31
"""Largest magnitude of a signed 6-bit weight. The range is symmetric, so -32 is not used."""


def feature_scale(weight: torch.Tensor) -> torch.Tensor:
    """Step size of the feature table: 4 standard deviations cover the 6-bit range."""
    return (4 * weight.detach().std() / LEVELS).clamp_min(1e-8)


def quantize_features(weight: torch.Tensor) -> torch.Tensor:
    """Round the feature table to 6-bit steps. The gradient passes through unchanged."""
    scale = feature_scale(weight)
    rounded = (weight / scale).round().clamp(-LEVELS, LEVELS) * scale
    return weight + (rounded - weight).detach()


def quantize_words(weight: torch.Tensor) -> torch.Tensor:
    """Round each row of the word table to 6-bit steps of its largest value."""
    scale = weight.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / LEVELS
    rounded = (weight / scale).round() * scale
    return weight + (rounded - weight).detach()


def _pack6(values: np.ndarray) -> bytes:
    """Pack integers from -32 to 31 into 6 bits each: four values in three bytes, lowest bits first."""
    unsigned = (values.astype(np.int64).ravel() + 32).astype(np.uint32)
    padded = np.zeros((len(unsigned) + 3) // 4 * 4, dtype=np.uint32)
    padded[: len(unsigned)] = unsigned
    groups = padded.reshape(-1, 4)
    word = groups[:, 0] | (groups[:, 1] << 6) | (groups[:, 2] << 12) | (groups[:, 3] << 18)
    packed = np.stack([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], axis=1)
    return packed.astype(np.uint8).tobytes()


def _float_tensor(name: str, tensor: torch.Tensor) -> tuple[dict, bytes]:
    values = tensor.detach().cpu().to(torch.float32).numpy()
    return {"name": name, "shape": list(values.shape), "type": "f32"}, values.astype("<f4").tobytes()


def _feature_tensor(weight: torch.Tensor, bits: int) -> tuple[dict, bytes]:
    if bits != 6:
        return _float_tensor("features", weight)
    scale = feature_scale(weight)
    values = (weight.detach() / scale).round().clamp(-LEVELS, LEVELS).cpu().numpy()
    return {"name": "features", "shape": list(values.shape), "type": "q6", "scale": float(scale)}, _pack6(
        values
    )


def _word_tensor(weight: torch.Tensor, bits: int) -> tuple[dict, bytes]:
    if bits != 6:
        header, payload = _float_tensor("words", torch.nn.functional.normalize(weight, dim=-1))
        return {**header, "unit": True}, payload
    scale = weight.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / LEVELS
    values = (weight.detach() / scale).round().cpu().numpy()
    return {"name": "words", "shape": list(values.shape), "type": "q6", "scale": 1.0, "unit": True}, _pack6(
        values
    )


def write_weights(model, vocabulary: list[str], file: Path, bits: int) -> None:
    """Write the weight file. With `bits` 6, the two large tables use 6 bits per weight."""
    parts = [
        _feature_tensor(model.features.weight, bits),
        _word_tensor(model.words, bits),
        _float_tensor("gateWeight", model.gate.weight.flatten()),
        _float_tensor("gateBias", model.gate.bias),
        _float_tensor("fc1Weight", model.fc1.weight),
        _float_tensor("fc1Bias", model.fc1.bias),
        _float_tensor("fc2Weight", model.fc2.weight),
        _float_tensor("fc2Bias", model.fc2.bias),
    ]
    payload = bytearray()
    tensors = []
    for header, data in parts:
        payload += b"\0" * (-len(payload) % 4)
        tensors.append({**header, "offset": len(payload)})
        payload += data
    words = "\n".join(vocabulary).encode()
    header = {
        "version": 1,
        "rows": model.features.num_embeddings,
        "dim": model.features.embedding_dim,
        "hidden": model.fc1.out_features,
        "tensors": tensors,
        "vocabulary": {"offset": len(payload), "length": len(words)},
    }
    payload += words
    encoded = json.dumps(header, separators=(",", ":")).encode()
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("wb") as out:
        out.write(len(encoded).to_bytes(4, "little"))
        out.write(encoded)
        out.write(b"\0" * (-(4 + len(encoded)) % 4))
        out.write(payload)
