"""Fit the model on the targets from `src/targets.ts` and export a candidate."""

import argparse
import json
import math
import time
from pathlib import Path

import torch.nn.functional as F

import torch
from data import TrainingData
from features import split_query
from model import IconWordModel

ROOT = Path(__file__).parent.parent
PROBES = [
    "trash",
    "hom",
    "hoem",
    "settings",
    "sign in",
    "icon for removing an item",
    "dark mode",
    "shoping cart",
]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(
    model: IconWordModel, data: TrainingData, queries: list[str], device: torch.device, limit: int = 8
):
    """Return the best words for each query, with their cosines."""
    rows, offsets, owner = data.encode([split_query(query) or ["a"] for query in queries])
    with torch.no_grad():
        scores = (
            model(rows.to(device), offsets.to(device), owner.to(device), len(queries)) / model.log_scale.exp()
        )
    best = scores.topk(limit, dim=-1)
    return [
        [
            (data.vocabulary[index], round(score, 2))
            for index, score in zip(ids.tolist(), values.tolist(), strict=True)
        ]
        for ids, values in zip(best.indices, best.values, strict=True)
    ]


def export(model: IconWordModel, data: TrainingData, args: argparse.Namespace, out: Path) -> None:
    """Write the weights as little-endian float32, with a JSON file that describes them."""
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "features": model.features.weight,
        "gate_weight": model.gate.weight.flatten(),
        "gate_bias": model.gate.bias,
        "fc1_weight": model.fc1.weight,
        "fc1_bias": model.fc1.bias,
        "fc2_weight": model.fc2.weight,
        "fc2_bias": model.fc2.bias,
        "words": F.normalize(model.words, dim=-1),
    }
    tensors = []
    offset = 0
    with (out / "model.bin").open("wb") as file:
        for name, tensor in state.items():
            values = tensor.detach().cpu().to(torch.float32).contiguous()
            file.write(values.numpy().astype("<f4").tobytes())
            tensors.append({"name": name, "shape": list(values.shape), "offset": offset})
            offset += values.numel() * 4
    meta = {
        "rows": args.rows,
        "dim": args.dim,
        "hidden": args.hidden,
        "tensors": tensors,
        "vocabulary": data.vocabulary,
        "training": {"steps": args.steps, "batch": args.batch, "seed": args.seed},
    }
    (out / "model.json").write_text(json.dumps(meta) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / ".cache/train/targets.json")
    parser.add_argument("--out", type=Path, default=ROOT / ".cache/candidate")
    parser.add_argument("--rows", type=int, default=16384, help="rows of the feature table, a power of two")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    data = TrainingData(args.targets, args.rows, args.seed)
    model = IconWordModel(args.rows, len(data.vocabulary), args.dim, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 500
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        ),
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device {device}, {len(data.texts)} queries, {parameters} parameters")

    started = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        batch = data.sample(args.batch)
        logits = model(
            batch.rows.to(device), batch.offsets.to(device), batch.word_query.to(device), args.batch
        )
        log_probability = F.log_softmax(logits, dim=-1).gather(1, batch.target_ids.to(device))
        loss = -(batch.target_weights.to(device) * log_probability).sum(dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        running += loss.item()
        if step % 500 == 0:
            rate = step / (time.time() - started)
            print(f"step {step:6d}  loss {running / 500:.4f}  {rate:.1f} steps/s", flush=True)
            running = 0.0

    model.eval()
    for query, words in zip(PROBES, predict(model, data, PROBES, device), strict=True):
        print(f"{query!r:32} {' '.join(f'{word}:{score}' for word, score in words)}")
    export(model, data, args, args.out)
    print(f"exported to {args.out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
