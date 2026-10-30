"""CRPS and energy scores of empirical forecast ensembles."""

from __future__ import annotations

import numpy as np


def _sorted_pair_sum(samples: np.ndarray) -> np.ndarray:
    """Return sum_{i,j} |x_i - x_j| over all ordered pairs, via sorting.

    For sorted x, sum_{i<j} (x_(j) - x_(i)) = sum_k x_(k) * (2k + 1 - m) with
    k = 0..m-1, and the full ordered double sum is twice that.  O(m log m).
    """
    m = samples.shape[-1]
    xs = np.sort(samples, axis=-1)
    weights = 2.0 * np.arange(m, dtype=np.float64) + 1.0 - m
    return 2.0 * (xs * weights).sum(axis=-1)


def crps_samples(samples, y) -> np.ndarray:
    """Empirical CRPS of an m-member sample forecast against observations.

    score = mean_i |x_i - y| - sum_{i,j} |x_i - x_j| / (2 m^2).
    The ordered double sum is computed in O(m log m) by sorting.

    Parameters
    ----------
    samples : array_like, shape (..., m)
        Forecast sample paths along the trailing axis.
    y : array_like, shape (...)
        Observations, broadcastable against the batch shape of ``samples``.

    Returns
    -------
    ndarray with the broadcast batch shape.
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = samples.shape[-1]
    if m < 1:
        raise ValueError("At least one forecast sample is required")
    c = 1.0 / (2.0 * m * m)
    term1 = np.abs(samples - y[..., None]).mean(axis=-1)
    return term1 - c * _sorted_pair_sum(samples)


def energy_score(samples, y, chunk_size: int = 4) -> np.ndarray:
    """Empirical energy score of a multivariate m-member sample forecast.

    ES = mean_i ||x_i - y||_2 - sum_{i,j} ||x_i - x_j||_2 / (2 m^2).
    For d = 1 the energy score reduces to CRPS.

    The O(m^2 d) pairwise term is computed in chunks of ``chunk_size``
    samples to bound temporary memory.

    Parameters
    ----------
    samples : array_like, shape (..., m, d)
    y : array_like, shape (..., d)
    chunk_size : int
        Number of samples per pairwise chunk (peak temporary memory is
        proportional to batch * chunk_size * m * d).

    Returns
    -------
    ndarray with the broadcast batch shape.
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    m = samples.shape[-2]
    if m < 1:
        raise ValueError("At least one forecast sample is required")
    c = 1.0 / (2.0 * m * m)
    term1 = np.sqrt(np.square(samples - y[..., None, :]).sum(axis=-1)).mean(axis=-1)
    batch_shape = samples.shape[:-2]
    d = samples.shape[-1]
    pair_sum = np.zeros(batch_shape, dtype=np.float64)
    # One reusable buffer keeps peak memory at a single chunk allocation.
    buf = np.empty(batch_shape + (min(chunk_size, m), m, d), dtype=np.float64)
    for i0 in range(0, m, chunk_size):
        chunk = samples[..., i0 : i0 + chunk_size, :]
        diff = buf[..., : chunk.shape[-2], :, :]
        np.subtract(chunk[..., :, None, :], samples[..., None, :, :], out=diff)
        np.square(diff, out=diff)
        dist = diff.sum(axis=-1)
        np.sqrt(dist, out=dist)
        pair_sum += dist.sum(axis=(-2, -1))
    return term1 - c * pair_sum
