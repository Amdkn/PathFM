"""Evaluate PathFM on the complete 100-task FEV benchmark."""

from __future__ import annotations

import ast
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import common
from .fev_data import (
    context_stats_from_flat,
    seasonal_error_per_item,
    task_file_stem,
    validate_samples,
    window_arrays,
    window_payload,
    write_dump,
)


def tasks():
    return yaml.safe_load((common.PROTOCOLS / "fev-tasks.yaml").read_text())["tasks"]


def task_name(config):
    return config.get("task_name", config["dataset_config"])


def local_paths(root):
    if root is None:
        return {}
    found = {}
    for path in sorted(root.rglob("*.parquet")):
        key = "_".join(path.parent.relative_to(root).parts)
        if key in found:
            raise ValueError(f"Duplicate local dataset: {key}")
        found[key] = path
    return found


def score(output):
    import fev.analysis

    records = [
        json.loads(p.read_text())
        for p in sorted((output / "task_results").glob("*.json"))
    ]
    common.require_coverage(
        [r["task_name"] for r in records], [task_name(c) for c in tasks()]
    )
    local = pd.DataFrame(
        [
            {
                k: str(v) if isinstance(v, list) else v
                for k, v in r.items()
                if not isinstance(v, dict)
            }
            for r in records
        ]
    )
    references = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted((common.PROTOCOLS / "fev-references").glob("*.csv"))
        ],
        ignore_index=True,
    )
    canonical = {
        str(ast.literal_eval(value)): value
        for value in references["extra_metrics"].dropna().unique()
    }
    local["extra_metrics"] = local["extra_metrics"].map(
        lambda v: canonical[str(ast.literal_eval(v))]
    )
    summaries = pd.concat([references, local], ignore_index=True)
    kwargs = dict(
        missing_strategy="impute",
        baseline_model="Seasonal Naive",
        leakage_imputation_model="Chronos-Bolt",
        n_resamples=1000,
        seed=42,
    )
    board = fev.analysis.leaderboard(summaries, **kwargs)
    pairs = fev.analysis.pairwise_comparison(summaries, **kwargs)
    board.to_csv(output / "leaderboard.csv")
    pairs.to_csv(output / "pairwise.csv")
    label = local["model_name"].iloc[0]
    row = board.loc[label]
    common.write_json(
        output / "summary.json",
        {
            "tasks": 100,
            "skill_score": float(row["skill_score"]),
            "skill_score_lower": float(row["skill_score_lower"]),
            "skill_score_upper": float(row["skill_score_upper"]),
            "num_failures": int(row["num_failures"]),
        },
    )
    print(
        board[["skill_score", "skill_score_lower", "skill_score_upper"]].to_string(),
        flush=True,
    )


def main():
    import fev

    p = common.parser(__doc__)
    p.add_argument(
        "--save-samples",
        action="store_true",
        help="Save native forecast paths for FPS scoring",
    )
    args = p.parse_args()
    if args.score_only:
        score(args.output)
        return
    if args.data_root is None:
        raise ValueError(
            "FEV requires --data-root pointing to the pinned dataset snapshot"
        )
    configs = tasks()
    selected = common.selected_tasks(configs, args, task_name)
    paths = local_paths(args.data_root)
    pipeline, protocol = common.build_pipeline(args)
    pipeline.capture_samples = args.save_samples
    label = pipeline.release_config["name"]
    for position, (index, config) in enumerate(selected, 1):
        name = task_name(config)
        stem = task_file_stem(index, name)
        result_path = args.output / "task_results" / f"{stem}.json"
        if common.completed(result_path, protocol):
            if (
                args.save_samples
                and not (args.output / "samples" / f"{stem}.npz").exists()
            ):
                raise ValueError(
                    f"{name} has scores but no saved samples; use a fresh output directory"
                )
            continue
        task_config = dict(config)
        if args.data_root is not None:
            if config["dataset_config"] not in paths:
                raise FileNotFoundError(f"No local data for {name}")
            task_config["dataset_path"] = str(paths[config["dataset_config"]])
            task_config["dataset_config"] = None
        task = fev.Task(**task_config)
        pipeline.set_stream(index)
        started = time.perf_counter()
        predictions, elapsed = pipeline.predict_fev(
            task,
            batch_size=args.batch_size,
            context_length=common.context_length(args.context_length),
            cross_learning=False,
        )
        if args.save_samples:
            payloads = []
            stats = {"num_windows": np.asarray(task.num_windows)}
            for wi, (prediction, window) in enumerate(
                zip(predictions, task.iter_windows(), strict=True)
            ):
                arrays = window_arrays(
                    window, task.clean_and_validate_predictions(prediction), task
                )
                payload = window_payload(
                    window,
                    arrays,
                    seasonal_error_per_item(
                        fev,
                        arrays["y_past"],
                        arrays["y_past_lengths"],
                        task.seasonality,
                    ),
                )
                items = pipeline.take_samples(arrays["y_true"].shape[0])
                samples = torch.stack(items).permute(0, 2, 3, 1).numpy()
                payload["samples"] = validate_samples(samples, arrays["y_true"], wi)
                payloads.append(payload)
                tau, _, _ = context_stats_from_flat(
                    arrays["y_past"], arrays["y_past_lengths"], task.horizon
                )
                stats[f"w{wi}_tau"] = tau
                stats[f"w{wi}_item_ids"] = payload["item_ids"]
            if pipeline._captured_items:
                raise RuntimeError(f"{name}: unconsumed sample paths")
            write_dump(
                args.output / "samples" / f"{stem}.npz",
                quantile_levels=task.quantile_levels,
                windows=payloads,
                meta={
                    "task_name": name,
                    "entry_name": label,
                    "horizon": task.horizon,
                    "seasonality": task.seasonality,
                    "task_idx": index,
                },
            )
            stats_dir = args.output / "context_stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(stats_dir / f"{stem}.npz", **stats)
        summary = task.evaluation_summary(
            predictions,
            model_name=label,
            inference_time_s=elapsed,
            trained_on_this_dataset=False,
            per_quantile_scores=True,
        )
        summary.update(
            task_name=name,
            dataset_path=config["dataset_path"],
            dataset_config=config["dataset_config"],
            protocol_signature=protocol["signature"],
            wall_seconds=time.perf_counter() - started,
        )
        if not np.isfinite(summary["test_error"]):
            raise FloatingPointError(f"{name}: nonfinite FEV score")
        common.write_json(result_path, summary)
        print(
            f"{position}/{len(selected)} {name}: {summary['test_error']:.8f}",
            flush=True,
        )
    if args.num_shards == 1 and not args.tasks:
        score(args.output)


if __name__ == "__main__":
    main()
