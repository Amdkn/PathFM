"""Six scores of the delivered empirical forecast distribution."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .empirical_scores import crps_samples, energy_score

_METRICS = ("es", "vs05", "des", "cumsum_crps", "max_crps", "fpt_crps")
_VS_PAIR_CHUNK = 4096
_VS_BUDGET_ELEMENTS = 2**25
_PAIR_CHUNK = 4
_CHUNK_BUDGET_ELEMENTS = 2**26


def build_window_paths(
    z, window_idx, *, task_name, quantile_levels, copula, num_paths, seed, dump_name
):
    if copula != "native":
        raise ValueError("PathFM FPS scoring requires native forecast paths")
    samples = z[f"w{window_idx}_samples"]
    if samples.shape[1] < num_paths:
        raise ValueError(f"{dump_name}: insufficient saved paths")
    return lambda d: np.asarray(samples[:, :num_paths, :, d], dtype=np.float64)


def _safemean(arr: np.ndarray, axis=None) -> float | np.ndarray:
    """Compute mean ignoring NaN, Inf, and -Inf values.

    Verbatim copy of ``fev.metrics.Metric._safemean`` (vendored fev, 0.9.0) so
    the exclusion semantics cannot drift from the SQL validation metric.
    """
    mask = ~np.isfinite(arr)
    if mask.any():
        arr = np.where(mask, np.nan, arr)
    return np.nanmean(arr, axis=axis)


def _item_chunk_size(num_paths: int, horizon: int) -> int:
    """Items per ES chunk so the pairwise buffer stays within the element budget."""
    per_item = max(1, _PAIR_CHUNK * num_paths * horizon)
    return max(1, _CHUNK_BUDGET_ELEMENTS // per_item)


def energy_scores_per_item(
    paths: np.ndarray, y_true: np.ndarray, seasonal_error: np.ndarray
) -> dict[str, np.ndarray | int]:
    """Scaled empirical energy scores per item for one (window, dim).

    Parameters
    ----------
    paths : np.ndarray [N, P, H]
        Sample paths (float64).
    y_true : np.ndarray [N, H]
        Ground truth; non-finite cells drop the corresponding step from the
        path (see module docstring).
    seasonal_error : np.ndarray [N]
        Per-item seasonal error; zero or non-finite values make the item's
        score non-finite (excluded downstream by ``_safemean``).

    Returns
    -------
    dict with ``scores [N]`` and the per-cause exclusion counts.
    """
    n_items, num_paths, horizon = paths.shape
    finite = np.isfinite(y_true)  # [N, H]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        se = seasonal_error[:, None]
        y_scaled = np.where(finite, y_true / se, 0.0)
        paths_scaled = np.where(finite[:, None, :], paths / se[..., None], 0.0)

    scores = np.empty(n_items, dtype=np.float64)
    chunk = _item_chunk_size(num_paths, horizon)
    with np.errstate(invalid="ignore", over="ignore"):
        for i0 in range(0, n_items, chunk):
            sl = slice(i0, i0 + chunk)
            scores[sl] = energy_score(
                paths_scaled[sl], y_scaled[sl], chunk_size=_PAIR_CHUNK,
            )
    # Items with no finite observation would otherwise score 0 on the empty
    # subpath; in fev's SQL every cell of such an item is NaN, so the item is
    # excluded entirely. Mirror that.
    no_finite_step = ~finite.any(axis=1)
    scores[no_finite_step] = np.nan
    # Attribute exclusions so silent forecast corruption is visible downstream:
    # a non-finite final score is model-independent (seasonal error, truth) or
    # the model's own doing (non-finite forecast values on scored steps).
    se_bad = ~np.isfinite(seasonal_error) | (seasonal_error == 0)
    final_bad = ~np.isfinite(scores)
    return {
        "scores": scores,
        "excluded_seasonal_error": int((se_bad & ~no_finite_step).sum()),
        "excluded_no_finite_truth": int(no_finite_step.sum()),
        "excluded_nonfinite_forecast": int(
            (final_bad & ~se_bad & ~no_finite_step).sum()
        ),
    }


def _attribute_exclusions(
    scores: np.ndarray, causes: list[tuple[str, np.ndarray]]
) -> dict[str, int]:
    """Count excluded items per cause; first-listed cause wins (ES scorer semantics).

    Any remaining non-finite score not claimed by a listed cause is the
    model's own doing and lands in ``excluded_nonfinite_forecast``.
    """
    out: dict[str, int] = {}
    claimed = np.zeros(scores.shape, dtype=bool)
    for name, mask in causes:
        out[name] = int((mask & ~claimed).sum())
        claimed |= mask
    out["excluded_nonfinite_forecast"] = int((~np.isfinite(scores) & ~claimed).sum())
    return out


def variogram_scores_per_item(
    paths: np.ndarray,
    y_true: np.ndarray,
    seasonal_error: np.ndarray,
    p: float = 0.5,
    pair_chunk: int = _VS_PAIR_CHUNK,
    budget_elements: int = _VS_BUDGET_ELEMENTS,
) -> np.ndarray:
    """Scaled variogram scores of order p per item for one (window, dim).

    Per item, over the step pairs (t < s) where BOTH truth steps are finite::

        mean over valid pairs of ( |ys_t - ys_s|^p - (1/P) sum_i |Xs_i,t - Xs_i,s|^p )^2

    with ``Xs = paths / se`` and ``ys = y_true / se``. Items with fewer than
    2 finite truth steps (no valid pair) score NaN; non-finite forecast or
    scaling values on a valid pair make the item non-finite (excluded by
    ``_safemean`` downstream).

    Parameters
    ----------
    paths : np.ndarray [N, P, H]
    y_true : np.ndarray [N, H]
    seasonal_error : np.ndarray [N]
    p : float
        Variogram order (default 0.5).
    pair_chunk, budget_elements : int
        Memory bounds: temporaries hold item_chunk * P * pair_chunk float64
        elements, with item_chunk derived from ``budget_elements``.

    Returns
    -------
    np.ndarray [N]
    """
    n_items, num_paths, horizon = paths.shape
    finite = np.isfinite(y_true)  # [N, H]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        se = seasonal_error[:, None]
        ys = y_true / se  # NaN/inf cells are masked per pair below
        paths_scaled = paths / se[..., None]

    scores = np.full(n_items, np.nan, dtype=np.float64)
    t_idx, s_idx = np.triu_indices(horizon, k=1)
    n_pairs = t_idx.size
    if n_pairs == 0:
        return scores
    valid = finite[:, t_idx] & finite[:, s_idx]  # [N, n_pairs]
    counts = valid.sum(axis=1)

    pair_chunk = max(1, min(pair_chunk, n_pairs))
    item_chunk = max(1, budget_elements // (num_paths * pair_chunk))
    acc = np.zeros(n_items, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        for i0 in range(0, n_items, item_chunk):
            isl = slice(i0, min(i0 + item_chunk, n_items))
            x_c = paths_scaled[isl]
            y_c = ys[isl]
            for p0 in range(0, n_pairs, pair_chunk):
                psl = slice(p0, p0 + pair_chunk)
                t_c, s_c = t_idx[psl], s_idx[psl]
                dx = x_c[:, :, t_c]  # fancy indexing copies, so in-place ops are safe
                dx -= x_c[:, :, s_c]
                np.abs(dx, out=dx)
                if p == 0.5:
                    np.sqrt(dx, out=dx)  # fast path for the default order
                elif p != 1.0:
                    np.power(dx, p, out=dx)
                model_term = dx.mean(axis=1)  # [items, pairs]
                obs_term = np.abs(y_c[:, t_c] - y_c[:, s_c]) ** p
                sq = np.square(obs_term - model_term)
                # Invalid pairs contribute exactly 0 (their NaNs are masked);
                # NaN/inf on a VALID pair survives and poisons the item.
                acc[isl] += np.where(valid[isl, psl], sq, 0.0).sum(axis=1)
        scores = acc / counts  # counts == 0 -> 0/0 -> NaN (< 2 finite steps)
    scores[counts == 0] = np.nan
    return scores


def cumsum_max_crps_per_item(
    paths: np.ndarray, y_true: np.ndarray, seasonal_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CRPS of the horizon-sum and horizon-max path functionals, scaled.

    Both functionals run over the finite-truth steps only (the ES scorer's
    step-masking analog); items with no finite truth step score NaN.

    Parameters
    ----------
    paths : np.ndarray [N, P, H]
    y_true : np.ndarray [N, H]
    seasonal_error : np.ndarray [N]

    Returns
    -------
    (cumsum_crps [N], max_crps [N])
    """
    finite = np.isfinite(y_true)  # [N, H]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        se = seasonal_error[:, None]
        ys = np.where(finite, y_true / se, 0.0)
        paths_scaled = paths / se[..., None]
        mask3 = finite[:, None, :]

        s_paths = np.where(mask3, paths_scaled, 0.0).sum(axis=2)  # [N, P]
        s_y = ys.sum(axis=1)  # [N]
        cum = crps_samples(s_paths, s_y)

        m_paths = np.where(mask3, paths_scaled, -np.inf).max(axis=2)  # [N, P]
        m_y = np.where(finite, ys, -np.inf).max(axis=1)  # [N]
        mx = crps_samples(m_paths, m_y)

    no_finite_step = ~finite.any(axis=1)
    cum[no_finite_step] = np.nan
    mx[no_finite_step] = np.nan
    return cum, mx


def des_coordinate_mask(finite: np.ndarray) -> np.ndarray:
    """Valid coordinates of the difference transform z, from the finite-truth mask.

    Coordinate 0 (the level anchor) needs step 0 finite; coordinate t >= 1
    (the increment v_t - v_{t-1}) needs steps t-1 AND t finite.

    Parameters
    ----------
    finite : np.ndarray [..., H] bool

    Returns
    -------
    np.ndarray [..., H] bool
    """
    mask = finite.copy()
    mask[..., 1:] &= finite[..., :-1]
    return mask


def des_per_item(
    paths: np.ndarray, y_true: np.ndarray, seasonal_error: np.ndarray,
) -> np.ndarray:
    """Differenced energy scores per item for one (window, dim), scaled.

    Per item, the observation and every sample path are divided by the
    seasonal error and mapped through the invertible transform
    z(v) = (v_1, v_2 - v_1, ..., v_H - v_{H-1}) (the first coordinate is the
    level anchor); the empirical energy score of the transformed vectors is
    returned. Coordinates invalid under :func:`des_coordinate_mask` are zeroed
    in BOTH the observation and every sample path (the subpath-restriction
    trick the ES scorer uses on levels). Items with no valid coordinate score
    NaN; non-finite forecast or scaling values on a valid coordinate make the
    item non-finite (excluded by ``_safemean`` downstream).

    Parameters
    ----------
    paths : np.ndarray [N, P, H]
    y_true : np.ndarray [N, H]
    seasonal_error : np.ndarray [N]

    Returns
    -------
    np.ndarray [N]
    """
    n_items, num_paths, horizon = paths.shape
    finite = np.isfinite(y_true)  # [N, H]
    mask = des_coordinate_mask(finite)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        se = seasonal_error[:, None]
        ys = y_true / se
        paths_scaled = paths / se[..., None]
        # np.diff with prepend=0 is exactly z: (v_1 - 0, v_2 - v_1, ...).
        # Masked-out coordinates carry NaN from masked truth; the np.where
        # zeroes them. NaN/inf on a VALID coordinate survives and poisons
        # the item.
        z_y = np.where(mask, np.diff(ys, axis=-1, prepend=0.0), 0.0)
        z_x = np.where(
            mask[:, None, :], np.diff(paths_scaled, axis=-1, prepend=0.0), 0.0
        )

        scores = np.empty(n_items, dtype=np.float64)
        chunk = _item_chunk_size(num_paths, horizon)
        for i0 in range(0, n_items, chunk):
            sl = slice(i0, i0 + chunk)
            scores[sl] = energy_score(
                z_x[sl], z_y[sl], chunk_size=_PAIR_CHUNK
            )
    scores[~mask.any(axis=1)] = np.nan
    return scores


def first_passage_times(x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Smallest 1-indexed t with ``x[..., t-1] > tau``, else H + 1 (censored atom).

    Parameters
    ----------
    x : np.ndarray [..., H]
        Paths or truth, ORIGINAL units.
    tau : np.ndarray
        Thresholds, broadcastable against ``x[..., 0]``.

    Returns
    -------
    float64 ndarray of first-passage times in {1, ..., H+1} with the
    broadcast batch shape. Non-finite x cells never register a crossing
    (callers must exclude corrupt paths explicitly).
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        exceed = x > np.asarray(tau, dtype=np.float64)[..., None]
    first = exceed.argmax(axis=-1) + 1  # first True, or index 0 when none
    return np.where(exceed.any(axis=-1), first, x.shape[-1] + 1).astype(np.float64)


def fpt_crps_per_item(
    paths: np.ndarray, y_true: np.ndarray, tau: np.ndarray,
) -> dict:
    """CRPS of the first-passage time over tau, per item, ORIGINAL units.

    Requires a fully finite truth horizon AND a finite tau; items failing
    either score NaN, as do items with any non-finite forecast value (a
    crossing time cannot be read off a corrupt path).

    Parameters
    ----------
    paths : np.ndarray [N, P, H]
        UNSCALED sample paths.
    y_true : np.ndarray [N, H]
        UNSCALED truth.
    tau : np.ndarray [N]
        Thresholds from the context stats (original units).

    Returns
    -------
    dict with ``scores [N]`` and the per-cause exclusion counts.
    """
    bad_truth = ~np.isfinite(y_true).all(axis=1)
    bad_tau = ~np.isfinite(tau)
    bad_paths = ~np.isfinite(paths).all(axis=(1, 2))

    t_paths = first_passage_times(paths, tau[:, None])  # [N, P]
    t_y = first_passage_times(y_true, tau)  # [N]
    scores = crps_samples(t_paths, t_y)
    scores[bad_truth | bad_tau | bad_paths] = np.nan

    exclusions = _attribute_exclusions(
        scores,
        [("excluded_nonfinite_truth", bad_truth), ("excluded_missing_tau", bad_tau)],
    )
    return {"scores": scores, **exclusions}


def score_task(
    npz_path: Path,
    *,
    copula: str,
    num_paths: int,
    seed: int,
    context_stats_path: Path | None = None,
    metrics: tuple[str, ...] | list[str] = _METRICS,
) -> dict:
    """Per-task path metrics from one dump file (+ optional context stats).

    Reduction per metric: ``_safemean`` over items -> mean over dims -> mean
    over windows, identical to the ES scorer. Returns a dict with one entry
    per scored metric plus per-metric exclusion counts; ``fpt_crps`` is
    absent when the context-stats file is missing. ``metrics`` selects a
    subset (always scored in ``_METRICS`` order); the paths are built
    identically regardless, so any subset scores the SAME paths.
    """
    z = np.load(npz_path, allow_pickle=False)
    task_name = str(z["task_name"].item())
    num_windows = int(z["num_windows"])
    quantile_levels = np.asarray(z["quantile_levels"], dtype=np.float64)

    metrics = [m for m in _METRICS if m in set(metrics)]
    stats = None
    if "fpt_crps" in metrics:
        if context_stats_path is not None and context_stats_path.exists():
            stats = np.load(context_stats_path, allow_pickle=False)
            stats_windows = int(stats["num_windows"])
            if stats_windows != num_windows:
                raise ValueError(
                    f"{npz_path.name}: context stats have {stats_windows} windows, dump has {num_windows}"
                )
        else:
            warnings.warn(
                f"{npz_path.stem}: no context-stats npz at {context_stats_path}; "
                "SKIPPING fpt_crps for this task (other metrics still scored)",
                stacklevel=1,
            )
            metrics = [m for m in metrics if m != "fpt_crps"]

    window_scores: dict[str, list[float]] = {m: [] for m in metrics}
    exclusions: dict[str, dict[str, int]] = {m: {} for m in metrics}
    for w in range(num_windows):
        y_true = np.asarray(z[f"w{w}_y_true"], dtype=np.float64)  # [N, H, D]
        seasonal_error = np.asarray(
            z[f"w{w}_seasonal_error"], dtype=np.float64
        )  # [N, D]
        n_items, _, n_dims = y_true.shape

        get_paths = build_window_paths(
            z,
            w,
            task_name=task_name,
            quantile_levels=quantile_levels,
            copula=copula,
            num_paths=num_paths,
            seed=seed,
            dump_name=npz_path.name,
        )

        tau = None
        if stats is not None:
            tau = np.asarray(stats[f"w{w}_tau"], dtype=np.float64)  # [N, D]
            if tau.shape != (n_items, n_dims):
                raise ValueError(
                    f"{npz_path.name} window {w}: tau shape {tau.shape} != dump items/dims {(n_items, n_dims)}"
                )
            ids_key = f"w{w}_item_ids"
            if (
                ids_key in z.files
                and ids_key in stats.files
                and not np.array_equal(
                    np.asarray(z[ids_key], dtype=str),
                    np.asarray(stats[ids_key], dtype=str),
                )
            ):
                raise ValueError(
                    f"{npz_path.name} window {w}: item ids differ between dump and context stats"
                )

        per_dim = {m: np.empty(n_dims, dtype=np.float64) for m in metrics}
        for d in range(n_dims):
            paths = get_paths(d)  # built ONCE per dim: all metrics score the SAME paths
            y_d, se_d = y_true[:, :, d], seasonal_error[:, d]
            finite = np.isfinite(y_d)
            se_bad = ~np.isfinite(se_d) | (se_d == 0)

            item_scores: dict[str, np.ndarray] = {}
            dim_exclusions: dict[str, dict[str, int]] = {}
            if "es" in metrics:
                es = energy_scores_per_item(paths, y_d, se_d)
                item_scores["es"] = es["scores"]
                dim_exclusions["es"] = {
                    k: v for k, v in es.items() if k.startswith("excluded_")
                }

            if "vs05" in metrics:
                vs = variogram_scores_per_item(paths, y_d, se_d)
                item_scores["vs05"] = vs
                dim_exclusions["vs05"] = _attribute_exclusions(
                    vs,
                    [
                        ("excluded_insufficient_truth", finite.sum(axis=1) < 2),
                        ("excluded_seasonal_error", se_bad),
                    ],
                )

            if "cumsum_crps" in metrics or "max_crps" in metrics:
                cum, mx = cumsum_max_crps_per_item(paths, y_d, se_d)
                no_finite = ~finite.any(axis=1)
                causes = [
                    ("excluded_no_finite_truth", no_finite),
                    ("excluded_seasonal_error", se_bad),
                ]
                if "cumsum_crps" in metrics:
                    item_scores["cumsum_crps"] = cum
                    dim_exclusions["cumsum_crps"] = _attribute_exclusions(cum, causes)
                if "max_crps" in metrics:
                    item_scores["max_crps"] = mx
                    dim_exclusions["max_crps"] = _attribute_exclusions(mx, causes)

            if tau is not None and "fpt_crps" in metrics:
                fpt = fpt_crps_per_item(paths, y_d, tau[:, d])
                item_scores["fpt_crps"] = fpt.pop("scores")
                dim_exclusions["fpt_crps"] = fpt

            if "des" in metrics:
                des = des_per_item(paths, y_d, se_d)
                no_coord = ~des_coordinate_mask(finite).any(axis=1)
                item_scores["des"] = des
                dim_exclusions["des"] = _attribute_exclusions(
                    des,
                    [
                        ("excluded_no_valid_coordinate", no_coord),
                        ("excluded_seasonal_error", se_bad),
                    ],
                )

            for m in metrics:
                per_dim[m][d] = _safemean(item_scores[m])
                for key, count in dim_exclusions[m].items():
                    exclusions[m][key] = exclusions[m].get(key, 0) + count
        for m in metrics:
            window_scores[m].append(float(np.mean(per_dim[m])))

    return {
        "task_name": task_name,
        "metrics": {m: float(np.mean(window_scores[m])) for m in metrics},
        "exclusions": exclusions,
        "horizon": int(z["horizon"]),
        "seasonality": int(z["seasonality"]),
        "num_windows": num_windows,
        "estimator": "empirical",
        "score_protocol": "fps_empirical_v1",
    }
