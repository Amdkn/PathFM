"""Evaluate PathFM on all 97 GIFT configurations without window or series caps."""

from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

from . import common
from .gift_data import Dataset

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def tasks():
    return json.loads((common.PROTOCOLS / "gift-tasks.json").read_text())


def key(task):
    return f"{task[0]}:{task[1]}"


def canonical(name, term):
    properties = json.loads(
        (common.PROTOCOLS / "gift-dataset-properties.json").read_text()
    )
    parts = name.split("/")
    base = PRETTY_NAMES.get(parts[0].lower(), parts[0].lower())
    frequency = parts[1] if len(parts) > 1 else properties[base]["frequency"]
    return f"{base}/{frequency}/{term}"


def forecasts(pipeline, entries, horizon, args):
    inputs = [{"target": item["target"]} for item in entries]
    quantiles, _ = pipeline.predict_quantiles(
        inputs,
        prediction_length=horizon,
        batch_size=args.batch_size,
        quantile_levels=common.QUANTILES,
        context_length=common.context_length(args.context_length),
    )
    array = torch.stack(quantiles).permute(0, 3, 2, 1).cpu().numpy()
    if np.ndim(entries[0]["target"]) == 1:
        array = array.squeeze(-1)
    if not np.isfinite(array).all():
        raise FloatingPointError("GIFT received nonfinite forecasts")
    return [
        QuantileForecast(
            forecast_arrays=values,
            forecast_keys=[str(q) for q in common.QUANTILES],
            # Preserve the reference notebook's timestamp convention.
            start_date=entry["start"] + len(entry["target"]),
        )
        for values, entry in zip(array, entries, strict=True)
    ]


def metric_list():
    return [
        MASE(),
        MeanWeightedSumQuantileLoss(quantile_levels=common.QUANTILES),
    ]


def score(output):
    records = [
        json.loads(p.read_text())
        for p in sorted((output / "task_results").glob("*.json"))
    ]
    common.require_coverage([r["task_id"] for r in records], [key(t) for t in tasks()])
    frame = pd.DataFrame(records).set_index("dataset")
    baseline = pd.read_csv(common.PROTOCOLS / "gift-seasonal-naive.csv").set_index(
        "dataset"
    )
    baseline = baseline.loc[frame.index]
    summary = {"configurations": len(frame)}
    for metric in ("MASE", "CRPS"):
        ratios = frame[metric].to_numpy() / baseline[metric].to_numpy()
        if not np.isfinite(ratios).all() or not (ratios > 0).all():
            raise ValueError(f"Invalid normalized GIFT {metric}")
        summary[f"{metric}_normalized"] = float(np.exp(np.log(ratios).mean()))
    frame.to_csv(output / "task_scores.csv")
    common.write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = common.parser(__doc__).parse_args()
    if args.score_only:
        score(args.output)
        return
    if args.data_root is None:
        raise ValueError(
            "GIFT requires --data-root pointing to the GIFT-Eval dataset snapshot"
        )
    os.environ["GIFT_EVAL"] = str(args.data_root)
    selected = common.selected_tasks(tasks(), args, key)
    pipeline, protocol = common.build_pipeline(args)
    for position, (index, task) in enumerate(selected, 1):
        name, term = task
        task_id = key(task)
        path = (
            args.output
            / "task_results"
            / f"{task_id.replace('/', '__').replace(':', '__')}.json"
        )
        if common.completed(path, protocol):
            continue
        dataset = Dataset(name=name, term=term, to_univariate=False)
        pipeline.set_stream(index)
        started = time.perf_counter()
        windows = dataset.test_data.windows
        per_window = []
        for wi in range(windows):
            print(
                f"{position}/{len(selected)} {task_id}: forecasting window {wi + 1}/{windows}",
                flush=True,
            )
            entries = list(itertools.islice(dataset.test_data.input, wi, None, windows))
            per_window.append(
                forecasts(pipeline, entries, dataset.prediction_length, args)
            )
        ordered = [
            forecast for group in zip(*per_window, strict=True) for forecast in group
        ]
        record = (
            evaluate_forecasts(
                ordered,
                test_data=dataset.test_data,
                metrics=metric_list(),
                batch_size=1024,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(dataset.freq),
            )
            .reset_index(drop=True)
            .to_dict(orient="records")[0]
        )
        # GluonTS uses these names before the published table adds eval_metrics/.
        scores = {
            "MASE": float(record["MASE[0.5]"]),
            "CRPS": float(record["mean_weighted_sum_quantile_loss"]),
        }
        if not all(np.isfinite(v) for v in scores.values()):
            raise FloatingPointError(f"{task_id}: nonfinite GIFT score")
        result = {
            "task_id": task_id,
            "dataset": canonical(name, term),
            **scores,
            "all_metrics": record,
            "windows": windows,
            "num_series": len(dataset.hf_dataset),
            "num_entries": len(ordered),
            "prediction_length": dataset.prediction_length,
            "target_dim": dataset.target_dim,
            "dataset_fingerprint": getattr(dataset.hf_dataset, "_fingerprint", None),
            "protocol_signature": protocol["signature"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        common.write_json(path, result)
        print(f"{position}/{len(selected)} {task_id}: {scores}", flush=True)
    if args.num_shards == 1 and not args.tasks:
        score(args.output)


if __name__ == "__main__":
    main()
