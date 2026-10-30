"""Continue coherent paths from observed dynamical-system states."""

import argparse
from pathlib import Path

import numpy as np

from pathfm import PathFMPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--context",
        type=Path,
        required=True,
        help="NPY array [channels, context] or [batch, channels, context]",
    )
    parser.add_argument("--horizon", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("rollout.npz"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pipeline = PathFMPipeline.from_pretrained(args.model, device=args.device)
    forecast = pipeline.rollout(
        np.load(args.context, allow_pickle=False),
        prediction_length=args.horizon,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        samples=forecast.samples.numpy(),
        first_invalid_step=forecast.first_invalid_step.numpy(),
    )
    print(args.output)


if __name__ == "__main__":
    main()
