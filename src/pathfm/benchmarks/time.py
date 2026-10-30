"""Evaluate PathFM on all 98 TIME tasks with missing-value passthrough."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import yaml
from gluonts.time_feature import get_seasonality

from . import common
from .time_data import Dataset, get_dataset_settings
from .time_scoring import score_windows


def task_list(config):
    return [
        (name, term)
        for name, settings in config["datasets"].items()
        for term in ("short", "medium", "long")
        if settings.get(term, {}).get("prediction_length") is not None
    ]


def key(task):
    return f"{task[0]}/{task[1]}"


def predict(pipeline, dataset, args):
    inputs = list(dataset.test_data.input)
    forecasts = []
    cap = common.context_length(args.context_length)
    for start in range(0, len(inputs), args.batch_size):
        batch = []
        for item in inputs[start : start + args.batch_size]:
            target = np.asarray(item["target"], dtype=np.float32)
            if cap is not None:
                target = target[..., -cap:]
            tensor = torch.tensor(target)
            batch.append(tensor[None] if tensor.ndim == 1 else tensor)
        quantiles, _ = pipeline.predict_quantiles(
            batch,
            prediction_length=dataset.prediction_length,
            quantile_levels=common.QUANTILES,
            batch_size=len(batch),
            context_length=cap,
        )
        for q in quantiles:
            forecasts.append(q.float().cpu().numpy().transpose(2, 0, 1))
    predictions = np.stack(forecasts)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("TIME received nonfinite forecasts")
    return predictions


def score(output):
    records = [
        json.loads(p.read_text())
        for p in sorted((output / "task_results").glob("*.json"))
    ]
    config = yaml.safe_load((common.PROTOCOLS / "time-datasets.yaml").read_text())
    common.require_coverage(
        [r["task_id"] for r in records], [key(t) for t in task_list(config)]
    )
    frame = pd.DataFrame(records).set_index("task_id")
    baseline = pd.read_csv(common.PROTOCOLS / "time-seasonal-naive.csv").set_index(
        "task_id"
    )
    baseline = baseline.loc[frame.index]
    summary = {"tasks": len(frame)}
    for metric in ("MASE", "CRPS"):
        ratios = frame[metric].to_numpy() / baseline[metric].to_numpy()
        if not np.isfinite(ratios).all() or not (ratios > 0).all():
            raise ValueError(f"Invalid normalized TIME {metric}")
        summary[f"{metric}_normalized"] = float(np.exp(np.log(ratios).mean()))
    frame.to_csv(output / "task_scores.csv")
    common.write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = common.parser(__doc__, batch_size=16).parse_args()
    if args.score_only:
        score(args.output)
        return
    if args.data_root is None:
        raise ValueError(
            "TIME requires --data-root pointing to the TIME dataset snapshot"
        )
    config = yaml.safe_load((common.PROTOCOLS / "time-datasets.yaml").read_text())
    selected = common.selected_tasks(task_list(config), args, key)
    pipeline, protocol = common.build_pipeline(args)
    for position, (index, task) in enumerate(selected, 1):
        name, term = task
        task_id = key(task)
        path = args.output / "task_results" / f"{task_id.replace('/', '__')}.json"
        if common.completed(path, protocol):
            continue
        settings = get_dataset_settings(name, term, config)
        dataset = Dataset(
            name=name,
            term=term,
            to_univariate=False,
            prediction_length=settings["prediction_length"],
            test_length=settings["test_length"],
            val_length=settings.get("val_length") or 0,
            storage_path=args.data_root,
        )
        pipeline.set_stream(index)
        started = time.perf_counter()
        quantiles = predict(pipeline, dataset, args)
        metrics = score_windows(
            quantiles,
            list(dataset.test_data.input),
            list(dataset.test_data.label),
            windows=dataset.windows,
            seasonality=get_seasonality(dataset.freq),
            quantiles=common.QUANTILES,
        )
        prediction_dir = args.output / "predictions" / task_id
        prediction_dir.mkdir(parents=True, exist_ok=True)
        shaped = quantiles.reshape(-1, dataset.windows, *quantiles.shape[1:])
        np.savez_compressed(
            prediction_dir / "predictions.npz",
            predictions_quantiles=shaped,
            quantile_levels=np.asarray(common.QUANTILES, dtype=np.float64),
        )
        np.savez_compressed(prediction_dir / "metrics.npz", **metrics)
        scores = {
            metric: float(np.nanmean(metrics[metric])) for metric in ("MASE", "CRPS")
        }
        counts = {
            metric: int(np.isfinite(metrics[metric]).sum())
            for metric in ("MASE", "CRPS")
        }
        if not all(np.isfinite(v) for v in scores.values()):
            raise FloatingPointError(f"{task_id}: nonfinite TIME score")
        result = {
            "task_id": task_id,
            **scores,
            "finite_metric_counts": counts,
            "num_series": len(dataset.hf_dataset),
            "windows": dataset.windows,
            "prediction_length": dataset.prediction_length,
            "target_dim": dataset.target_dim,
            "protocol_signature": protocol["signature"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        common.write_json(path, result)
        print(f"{position}/{len(selected)} {task_id}: {scores}", flush=True)
    if args.num_shards == 1 and not args.tasks:
        score(args.output)


if __name__ == "__main__":
    main()
