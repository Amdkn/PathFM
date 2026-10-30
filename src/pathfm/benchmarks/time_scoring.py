"""TIME's per-window MASE and quantile-CRPS reductions."""

from __future__ import annotations

import numpy as np


def score_windows(predictions, inputs, labels, *, windows, seasonality, quantiles):
    """Score [items, quantiles, channels, horizon] predictions in protocol order."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 4 or len(predictions) % windows:
        raise ValueError("Predictions must contain complete per-series windows")
    if len(inputs) != len(predictions) or len(labels) != len(predictions):
        raise ValueError(
            "Predictions, inputs, and labels must have the same item count"
        )
    count, levels, channels, horizon = predictions.shape
    if levels != len(quantiles):
        raise ValueError("Unexpected quantile count")
    median = list(quantiles).index(0.5)
    scores = {name: np.full((count, channels), np.nan) for name in ("MASE", "CRPS")}
    for item, (past, future) in enumerate(zip(inputs, labels, strict=True)):
        history = np.asarray(past["target"], dtype=np.float64).reshape(channels, -1)
        target = np.asarray(future["target"], dtype=np.float64).reshape(
            channels, horizon
        )
        for channel in range(channels):
            observed = ~np.isnan(target[channel])
            if not observed.any():
                continue
            truth = target[channel, observed]
            forecast = predictions[item, :, channel][:, observed]
            # TIME removes missing context values before its seasonal differences.
            context = history[channel, ~np.isnan(history[channel])]
            if len(context) > seasonality:
                scale = np.abs(context[seasonality:] - context[:-seasonality]).mean()
                if scale > 0:
                    scores["MASE"][item, channel] = (
                        np.abs(truth - forecast[median]).mean() / scale
                    )
            magnitude = np.abs(truth).sum()
            if magnitude > 0:
                losses = []
                for level, values in zip(quantiles, forecast, strict=True):
                    loss = 2 * np.abs(
                        (truth - values) * ((values >= truth).astype(float) - level)
                    )
                    losses.append(loss.sum() / magnitude)
                scores["CRPS"][item, channel] = np.mean(losses)
    return {
        name: values.reshape(count // windows, windows, channels)
        for name, values in scores.items()
    }
