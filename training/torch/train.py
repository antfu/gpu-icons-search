"""Fit the model on the targets from `src/targets.ts` and export a candidate."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import TrainingData
from export import write_weights
from features import split_query
from model import IconWordModel

ROOT = Path(__file__).parent.parent
FIXTURE_QUERIES = ["home", "hom", "hoem", "trash", "arrow up right", "icon for removing an item", "x", "a11y"]
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
            (data.vocabulary[index], round(score, 4))
            for index, score in zip(ids.tolist(), values.tolist(), strict=True)
        ]
        for ids, values in zip(best.indices, best.values, strict=True)
    ]


def write_fixtures(model: IconWordModel, data: TrainingData, device: torch.device, file: Path) -> None:
    """Store the unit vectors and best words of a few queries, for the parity tests of other runtimes."""
    queries = [split_query(query) for query in FIXTURE_QUERIES]
    rows, offsets, owner = data.encode(queries)
    with torch.no_grad():
        vectors = model.encode(rows.to(device), offsets.to(device), owner.to(device), len(queries))
    best = predict(model, data, FIXTURE_QUERIES, device, limit=8)
    fixtures = [
        {"words": words, "vector": [round(value, 6) for value in vector.tolist()], "best": top}
        for words, vector, top in zip(queries, vectors.cpu(), best, strict=True)
    ]
    file.write_text(json.dumps(fixtures) + "\n")


def export(model: IconWordModel, data: TrainingData, args: argparse.Namespace, device: torch.device) -> None:
    """Write the candidate: weight file, parity fixtures, checkpoint and a report of the run."""
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    model.quantize = args.bits == 6
    write_weights(model, data.vocabulary, out / "model.bin", args.bits)
    write_fixtures(model, data, device, out / "fixtures.json")
    torch.save(model.state_dict(), out / "model.pt")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {"settings": settings, "bytes": (out / "model.bin").stat().st_size, "torch": torch.__version__}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


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
    parser.add_argument(
        "--bits", type=int, default=6, choices=[6, 32], help="bits per weight of the two tables"
    )
    parser.add_argument(
        "--qat-steps", type=int, default=4000, help="last steps that train with 6-bit rounding"
    )
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
        model.quantize = args.bits == 6 and step > args.steps - args.qat_steps
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
    export(model, data, args, device)
    for query, words in zip(PROBES, predict(model, data, PROBES, device), strict=True):
        print(f"{query!r:32} {' '.join(f'{word}:{score:.2f}' for word, score in words)}")
    print(f"exported to {args.out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
