"""Shared command-line options and result identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from pathfm import PathFMPipeline

PROTOCOLS = Path(__file__).parent / "protocols"
QUANTILES = [i / 10 for i in range(1, 10)]


def parser(description, batch_size=64):
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--model",
        help="Hugging Face model ID or local model directory; required for inference",
    )
    result.add_argument("--revision", help="Hugging Face commit or revision")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--data-root",
        type=Path,
        help="Pinned dataset snapshot from the download command",
    )
    result.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    result.add_argument("--context-length", default="native")
    result.add_argument("--steps", type=int, default=1)
    result.add_argument("--num-samples", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=batch_size)
    result.add_argument("--token-budget", type=int, default=32768)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument(
        "--tasks", nargs="+", help="Optional named tasks for smoke checks"
    )
    result.add_argument(
        "--score-only",
        action="store_true",
        help="Aggregate a completed run without loading the model",
    )
    return result


def context_length(value):
    if value == "native":
        return None
    value = int(value)
    if value < 1:
        raise ValueError("Context length must be positive or 'native'")
    return value


def selected_tasks(tasks, args, key=lambda item: item):
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    indexed = list(enumerate(tasks))
    if args.tasks:
        wanted = set(args.tasks)
        indexed = [(index, task) for index, task in indexed if key(task) in wanted]
        missing = wanted - {key(task) for _, task in indexed}
        if missing:
            raise ValueError(f"Unknown tasks: {sorted(missing)}")
    return indexed[args.shard_index :: args.num_shards]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
        temporary = f.name
    os.replace(temporary, path)


def build_pipeline(args):
    if not args.model:
        raise ValueError("--model is required unless --score-only is used")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pipeline = PathFMPipeline.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        steps=args.steps,
        num_samples=args.num_samples,
        token_budget=args.token_budget,
        seed=args.seed,
        source_sampling="stratified",
    )
    identity = pipeline.release_config.get("weights_sha256") or pipeline.release_config.get(
        "provenance", {}
    ).get("source_checkpoint_sha256")
    if not identity:
        raise ValueError("The model config needs a checkpoint or weights SHA256 for benchmark identity")
    protocol = {
        "source_checkpoint": identity,
        "weights": "ema",
        "steps": args.steps,
        "num_samples": args.num_samples,
        "source_sampling": "stratified",
        "sampling_protocol": "global_strata_v1",
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "token_budget": args.token_budget,
        "seed": args.seed,
    }
    encoded = json.dumps(protocol, sort_keys=True).encode()
    protocol["signature"] = hashlib.sha256(encoded).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "protocol.json"
    try:
        with manifest.open("x") as stream:
            json.dump(protocol, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        existing = json.loads(manifest.read_text())
        if existing != protocol:
            raise ValueError(f"{args.output} contains results for a different protocol")
    return pipeline, protocol


def completed(path, protocol):
    path = Path(path)
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if data.get("protocol_signature") != protocol["signature"]:
        raise ValueError(f"Stored result has a different protocol: {path}")
    return True


def require_coverage(names, expected):
    if len(names) != len(set(names)):
        raise ValueError("Duplicate benchmark tasks")
    missing = set(expected) - set(names)
    extra = set(names) - set(expected)
    if missing or extra:
        raise ValueError(
            f"Incomplete benchmark: missing={sorted(missing)}, extra={sorted(extra)}"
        )
