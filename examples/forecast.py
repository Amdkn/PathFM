"""Forecast a NumPy time series with a released PathFM model."""

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
        help="NPY array [batch, channels, context]",
    )
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("forecast.npz"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pipeline = PathFMPipeline.from_pretrained(args.model, device=args.device)
    context = np.load(args.context, allow_pickle=False)
    paths = pipeline.predict_samples(context, prediction_length=args.horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **{f"item_{i}": p.numpy() for i, p in enumerate(paths)}
    )
    print(args.output)


if __name__ == "__main__":
    main()
