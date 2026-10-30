"""FEV arrays, sample archives, and context-derived event thresholds."""

from pathlib import Path
from typing import Any

import numpy as np


def task_file_stem(task_index: int, name: str) -> str:
    """The per-task filename stem, shared by dumps, task JSONs and context stats.

    One definition on purpose. The evaluator used to slugify every non-alphanumeric
    character while the dump writer replaced only "/", and the two agreed on the
    fev-bench task names by luck rather than by rule -- a task name carrying a "."
    or a "-" would have silently split the JSON away from its dump.
    """

    return f"{task_index:03d}_{name.replace('/', '_')}"


def seasonal_error_per_item(fev, y_past, y_past_lengths, seasonality: int):
    """Per-item absolute seasonal error, the scale every dimension divides by.

    Taken from fev's own private helper rather than reimplemented: it is the
    denominator of the benchmark's SQL metric, so a difference here would silently
    rescale every fps-bench score away from the FEV numbers beside it.
    """

    from fev.metrics import _abs_seasonal_error_per_item

    return _abs_seasonal_error_per_item(
        y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
    )


def window_payload(window, arrays: dict[str, np.ndarray], seasonal_error) -> dict:
    """One window's dump entry, from the arrays the evaluator already built."""

    test_table = window._get_past_future_test_data()[2].data.table
    return {
        "y_true": arrays["y_true"],
        "quantiles": arrays["q_pred"],
        "seasonal_error": seasonal_error,
        "item_ids": np.asarray(
            test_table.column(window.id_column).to_pylist(), dtype=str
        ),
        "cutoff": str(window.cutoff),
    }


def validate_samples(samples: np.ndarray, y_true: np.ndarray, window_index: int):
    """Captured paths must cover every item, with no non-finite value.

    A path-emitting entry that answered only some items would be scored on those
    items alone and would look better than it is, so this is an error rather than
    an exclusion.
    """

    samples = np.asarray(samples)
    rows, horizon, dimensions = y_true.shape
    if (
        samples.ndim != 4
        or samples.shape[0] != rows
        or samples.shape[2] != horizon
        or samples.shape[3] != dimensions
    ):
        raise ValueError(
            f"captured paths for window {window_index} have shape {samples.shape}, "
            f"expected [N={rows}, m, H={horizon}, D={dimensions}]"
        )
    if not np.isfinite(samples).all():
        bad = int((~np.isfinite(samples)).sum())
        raise ValueError(
            f"captured paths for window {window_index} hold {bad} non-finite "
            "values; scoring would silently drop those items"
        )
    return samples


def write_dump(
    dump_path: Path,
    *,
    quantile_levels: list[float],
    windows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> Path:
    """Write one task's dump, atomically."""

    payload: dict[str, np.ndarray] = {
        "quantile_levels": np.asarray(quantile_levels, dtype=np.float64),
        "num_windows": np.asarray(len(windows)),
    }
    for key, value in meta.items():
        payload[key] = np.asarray(value)
    for index, window in enumerate(windows):
        payload[f"w{index}_y_true"] = window["y_true"].astype(np.float32)
        payload[f"w{index}_seasonal_error"] = window["seasonal_error"].astype(
            np.float64
        )
        payload[f"w{index}_item_ids"] = window["item_ids"]
        payload[f"w{index}_cutoff"] = np.asarray(window["cutoff"])
        if "samples" in window:
            samples = np.asarray(window["samples"], dtype=np.float32)
            # float16 halves the dump, but overflows to inf beyond 65504 (e.g.
            # australian_tourism counts reach ~8.6e4), which would poison the
            # energy score; such windows stay float32. The choice is per window,
            # so both dtypes coexist within one entry.
            finite = samples[np.isfinite(samples)]
            largest = float(np.abs(finite).max()) if finite.size else 0.0
            dtype = (
                np.float16 if largest <= float(np.finfo(np.float16).max) else np.float32
            )
            payload[f"w{index}_samples"] = samples.astype(dtype)
        else:
            payload[f"w{index}_quantiles"] = window["quantiles"].astype(np.float32)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    staging = dump_path.with_suffix(".npz.tmp.npz")
    np.savez_compressed(staging, **payload)
    staging.replace(dump_path)
    return dump_path


def window_arrays(window, cleaned_predictions, task) -> dict[str, np.ndarray]:
    """Rebuild the arrays ``fev`` scores, so prefixes can be scored too.

    This mirrors ``fev.task.EvaluationWindow.compute_metrics`` exactly; the full
    horizon is cross-checked against ``fev``'s own summary for every task.
    """

    import pyarrow.compute as pc

    past_data, _, test_data = window._get_past_future_test_data()
    columns = task.target_columns
    rows = len(test_data)
    horizon = task.horizon
    dimensions = len(columns)
    levels = task.quantile_levels
    test_table = test_data.data.table
    y_true = np.stack(
        [
            pc.list_flatten(test_table.column(column)).to_numpy(zero_copy_only=False)
            for column in columns
        ],
        axis=1,
        dtype=np.float64,
    ).reshape(rows, horizon, dimensions)
    prediction_tables = {
        column: cleaned_predictions[column].data.table for column in columns
    }
    y_pred = np.stack(
        [
            pc.list_flatten(prediction_tables[column].column("predictions")).to_numpy(
                zero_copy_only=False
            )
            for column in columns
        ],
        axis=1,
        dtype=np.float64,
    ).reshape(rows, horizon, dimensions)
    quantile_columns = []
    for column in columns:
        for level in levels:
            quantile_columns.append(
                pc.list_flatten(prediction_tables[column].column(str(level))).to_numpy(
                    zero_copy_only=False
                )
            )
    q_pred = np.stack(quantile_columns, axis=1, dtype=np.float64).reshape(
        rows,
        horizon,
        dimensions,
        len(levels),
    )
    past_table = past_data.data.table
    y_past = np.stack(
        [
            pc.list_flatten(past_table.column(column)).to_numpy(zero_copy_only=False)
            for column in columns
        ],
        axis=1,
        dtype=np.float64,
    )
    y_past_lengths = pc.list_value_length(past_table.column(columns[0])).to_numpy()
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "q_pred": q_pred,
        "y_past": y_past,
        "y_past_lengths": y_past_lengths,
    }


def context_stats_from_flat(
    y_past_flat: np.ndarray, y_past_lengths: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (tau, context_last, n_context) from the flattened context arrays.

    Parameters
    ----------
    y_past_flat : [sum(lengths), D] float array, items concatenated along axis 0
        (the exact layout of the evaluator's ``y_past``).
    y_past_lengths : [N] int array of per-item context lengths.
    horizon : task horizon H.

    Returns per (item, dim), using only the finite context values:
    - tau [N, D] float32: nanmax of the last min(H, n_valid) context steps
      (NaN if the item has no finite values, or if that trailing window is
      all-NaN due to interior missing values);
    - context_last [N, D] float32: last finite context value (NaN if none);
    - n_context [N, D] int32: number of finite context values.
    """
    y_past_flat = np.asarray(y_past_flat, dtype=np.float64)
    y_past_lengths = np.asarray(y_past_lengths, dtype=np.int64)
    if y_past_flat.ndim != 2:
        raise ValueError(
            f"y_past_flat must be [sum(lengths), D], got shape {y_past_flat.shape}"
        )
    N = len(y_past_lengths)
    D = y_past_flat.shape[1]
    offsets = np.concatenate([[0], np.cumsum(y_past_lengths)])
    if offsets[-1] != y_past_flat.shape[0]:
        raise ValueError(
            f"sum of y_past_lengths ({offsets[-1]}) != y_past_flat rows ({y_past_flat.shape[0]})"
        )

    tau = np.full((N, D), np.nan, dtype=np.float32)
    context_last = np.full((N, D), np.nan, dtype=np.float32)
    n_context = np.zeros((N, D), dtype=np.int32)

    for i in range(N):
        seg = y_past_flat[offsets[i] : offsets[i + 1]]  # [L, D]
        L = seg.shape[0]
        if L == 0:
            continue
        finite = np.isfinite(seg)
        if finite.all():
            # Fast path: fully observed context (the vast majority of items).
            n_context[i] = L
            context_last[i] = seg[-1]
            tau[i] = seg[-min(horizon, L) :].max(axis=0)
            continue
        for d in range(D):
            col = seg[:, d]
            mask = finite[:, d]
            n_valid = int(mask.sum())
            n_context[i, d] = n_valid
            if n_valid == 0:
                continue
            context_last[i, d] = col[mask][-1]
            window = col[-min(horizon, n_valid) :]
            if np.isfinite(window).any():
                tau[i, d] = np.nanmax(window)
    return tau, context_last, n_context
