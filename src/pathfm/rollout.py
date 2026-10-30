"""Independently continued sample paths for dynamical systems."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .sampling import denormalized_samples, flow_sample, validate_sampler


@dataclass
class RolloutForecast:
    """Samples [batch, paths, channels, horizon] and invalid-step indicators."""

    samples: torch.Tensor
    first_invalid_step: torch.Tensor


@torch.inference_mode()
def rollout(
    model,
    inputs,
    prediction_length,
    *,
    num_samples=16,
    steps=2,
    sampler_method=None,
    seed=42,
    source_sampling="iid",
    max_rows=4608,
    amp=True,
    max_abs_value=1e6,
):
    """Continue each sampled path independently from its own history.

    The initial context is standardized once per channel.
    Each native forecast block also uses the checkpoint's instance normalization.
    Invalid paths freeze at their last valid value and retain their failure index.
    """
    if prediction_length < 1 or num_samples < 2 or max_rows < 1:
        raise ValueError(
            "Require a positive horizon/row budget and at least two samples"
        )
    if sampler_method is None:
        sampler_method = "euler" if model.parallel_knots is None else "parallel"
    validate_sampler(model, sampler_method, steps)
    if max_abs_value is not None and max_abs_value <= 0:
        raise ValueError("max_abs_value must be positive or None")
    if isinstance(inputs, torch.Tensor):
        context = inputs.detach().float().cpu().numpy()
    else:
        context = np.asarray(inputs, dtype=np.float32)
    if context.ndim == 1:
        context = context[None, None]
    elif context.ndim == 2:
        context = context[None]
    if context.ndim != 3 or context.shape[-1] == 0:
        raise ValueError("Expected [batch, channels, context] or [channels, context]")
    if not np.isfinite(context).all():
        raise ValueError("Dynamical-system rollout requires a finite initial context")
    device = model.backbone.device
    amp = bool(amp and device.type == "cuda")
    windows, channels, context_length = context.shape
    paths = num_samples
    horizon = model.max_horizon
    blocks = math.ceil(prediction_length / horizon)
    location = context.mean(axis=-1, keepdims=True)
    scale = context.std(axis=-1, keepdims=True)
    scale = np.where(scale > 1e-10, scale, 1e-10)
    standardized = ((context - location) / scale).astype(np.float32)
    chunk_windows = max(1, max_rows // (channels * num_samples))
    out = np.empty((windows, paths, channels, blocks * horizon), dtype=np.float32)
    sentinel = np.iinfo(np.int64).max
    first_invalid = np.full((windows, paths), sentinel, dtype=np.int64)
    generator = torch.Generator(device=device).manual_seed(seed)

    def sample(rows, group_ids, count, sampling):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            static = model.prepare_static(
                rows,
                num_output_patches=model.max_output_patches,
                group_ids=group_ids,
            )
        values = flow_sample(
            model,
            static,
            torch.ones(rows.shape[0], horizon, device=device),
            num_samples=count,
            ode_steps=steps,
            method=sampler_method,
            source_sampling=sampling,
            generator=generator,
            amp=amp,
        )
        return denormalized_samples(model, static, values)

    for start in range(0, windows, chunk_windows):
        stop = min(start + chunk_windows, windows)
        n = stop - start
        loc = location[start:stop, None]
        scl = scale[start:stop, None]
        rolling = np.repeat(standardized[start:stop, None], paths, axis=1).copy()
        frozen_state = np.zeros((n, paths, channels), dtype=np.float32)
        invalid = first_invalid[start:stop].reshape(n * paths)
        for bi in range(blocks):
            block_start = bi * horizon
            if bi == 0:
                rows = torch.from_numpy(
                    np.ascontiguousarray(rolling[:, 0].reshape(n * channels, -1))
                ).to(device)
                groups = torch.arange(n, device=device).repeat_interleave(channels)
                raw = sample(rows, groups, num_samples, source_sampling)
                block = (
                    raw.float()
                    .cpu()
                    .numpy()
                    .reshape(n, channels, paths, horizon)
                    .transpose(0, 2, 1, 3)
                )
            else:
                rows = torch.from_numpy(
                    np.ascontiguousarray(rolling.reshape(n * paths * channels, -1))
                ).to(device)
                groups = torch.arange(n * paths, device=device).repeat_interleave(
                    channels
                )
                raw = sample(rows, groups, 1, "iid")
                block = raw.float().cpu().numpy().reshape(n, paths, channels, horizon)
            task_space = (block * scl + loc).reshape(n * paths, channels, horizon)
            bad = ~np.isfinite(task_space)
            if max_abs_value is not None:
                bad |= np.abs(task_space) > max_abs_value
            bad_any = bad.any(axis=(-2, -1))
            was_frozen = invalid != sentinel
            newly = bad_any & ~was_frozen
            if newly.any():
                first = np.where(bad.any(axis=-2), np.arange(horizon), horizon).min(
                    axis=-1
                )
                invalid[newly] = block_start + first[newly]
            flat = block.reshape(n * paths, channels, horizon)
            frozen = frozen_state.reshape(n * paths, channels)
            for index in np.flatnonzero(newly):
                offset = invalid[index] - block_start
                frozen[index] = (
                    flat[index, :, offset - 1]
                    if offset > 0
                    else rolling.reshape(n * paths, channels, -1)[index, :, -1]
                )
                flat[index, :, max(offset, 0) :] = frozen[index, :, None]
            for index in np.flatnonzero(was_frozen):
                flat[index] = frozen[index, :, None]
            np.nan_to_num(flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            block = flat.reshape(n, paths, channels, horizon)
            out[start:stop, :, :, block_start : block_start + horizon] = (
                block * scl + loc
            )
            rolling = np.concatenate([rolling, block], axis=-1)[..., -context_length:]
        first_invalid[start:stop] = invalid.reshape(n, paths)
    first_invalid = np.minimum(first_invalid, prediction_length)
    return RolloutForecast(
        torch.from_numpy(out[..., :prediction_length].copy()),
        torch.from_numpy(first_invalid),
    )
