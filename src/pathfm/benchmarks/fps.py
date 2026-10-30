"""Score saved FEV paths using the empirical FPS definition."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import common
from .fev import task_name, tasks
from .path_scores import _METRICS, score_task


def score_one(spec):
    path, stats, samples = spec
    if not stats.exists():
        raise FileNotFoundError(f"Missing context thresholds: {stats}")
    result = score_task(
        path,
        copula="native",
        num_paths=samples,
        seed=42,
        context_stats_path=stats,
    )
    for exclusions in result["exclusions"].values():
        if exclusions.get("excluded_nonfinite_forecast", 0):
            raise FloatingPointError(f"{path.name}: nonfinite forecast paths")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fev-output", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="For smoke checks; no full-benchmark skill is reported",
    )
    args = p.parse_args()
    output = args.output or args.fev_output / "fps"
    paths = sorted((args.fev_output / "samples").glob("*.npz"))
    if not paths:
        raise FileNotFoundError("No sample archives; run FEV with --save-samples")
    specs = [
        (path, args.fev_output / "context_stats" / path.name, args.num_samples)
        for path in paths
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(score_one, specs):
            results.append(result)
            print(f"{len(results)}/{len(paths)} {result['task_name']}", flush=True)
    frame = pd.DataFrame(
        [{"task_name": r["task_name"], **r["metrics"]} for r in results]
    ).set_index("task_name")
    if not args.allow_partial:
        common.require_coverage(list(frame.index), [task_name(c) for c in tasks()])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "task_scores.csv")
    common.write_json(
        output / "exclusions.json", {r["task_name"]: r["exclusions"] for r in results}
    )
    if args.allow_partial:
        return
    baseline = pd.read_csv(common.PROTOCOLS / "fps-seasonal-naive.csv").set_index(
        "task_name"
    )
    baseline = baseline.loc[frame.index]
    valid = np.isfinite(frame[list(_METRICS)]).all(axis=1) & np.isfinite(
        baseline[list(_METRICS)]
    ).all(axis=1)
    # The common set is explicit; undefined first-passage scores remain visible.
    common_names = sorted(frame.index[valid])
    if len(common_names) != 94:
        raise ValueError(
            f"Expected the 94-task common FPS support, found {len(common_names)}"
        )
    ratios = (
        frame.loc[common_names, list(_METRICS)].values
        / baseline.loc[common_names, list(_METRICS)].values
    )
    log_errors = np.log(np.clip(ratios, 0.01, 100.0)).mean(axis=1)
    skill = float(1 - np.exp(log_errors.mean()))
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(log_errors), size=(1000, len(log_errors)))
    interval = np.quantile(1 - np.exp(log_errors[indices].mean(axis=1)), [0.025, 0.975])
    summary = {
        "tasks": len(frame),
        "common_tasks": len(common_names),
        "skill_score": skill,
        "lower": float(interval[0]),
        "upper": float(interval[1]),
        "excluded_tasks": sorted(set(frame.index) - set(common_names)),
        "num_samples": args.num_samples,
        "seed": 42,
        "estimator": "empirical",
        "score_protocol": "fps_empirical_v1",
    }
    common.write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
