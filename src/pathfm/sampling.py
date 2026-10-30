"""Euler integration, parallel decoding, and reproducible source draws."""

from __future__ import annotations

from typing import Any

import torch

from .model import PathFM

SOURCE_SAMPLINGS = ("iid", "stratified")
HISTORY_CACHE_MIN_TOKENS = 8192


def supports_packed_paths(model: PathFM) -> bool:
    return True


def sampler_model_evaluations(method: str, steps: int) -> int:
    if method not in ("parallel", "euler") or steps < 1:
        raise ValueError("Use a positive step count with parallel or Euler sampling")
    return steps


def validate_sampler(model: PathFM, method: str, steps: int) -> None:
    sampler_model_evaluations(method, steps)
    if method == "parallel":
        parallel_block_length(model, steps)
    elif model.parallel_knots is not None:
        raise ValueError("This model requires parallel decoding")


def _euler_integrate(state, predict, blend, steps, amp):
    step_size = 1.0 / steps
    for step in range(steps):
        flow_time = torch.full(
            state.shape[:-1], step * step_size, device=state.device, dtype=torch.float32
        )
        with torch.autocast(
            device_type=state.device.type, dtype=torch.bfloat16, enabled=amp
        ):
            velocity = predict(state, flow_time).float()
        state = blend(state + step_size * velocity)
    return state


def _sampling_source(source, rows, num_samples, horizon, *, sampling, device, generator, scale):
    """Use a slice of an already scaled ensemble, or draw a complete ensemble."""
    if source is None:
        return draw_source(
            rows, num_samples, horizon, sampling=sampling, device=device,
            generator=generator, scale=scale,
        )
    expected = (rows, num_samples, horizon)
    if source.shape != expected:
        raise ValueError(f"source has shape {tuple(source.shape)}, expected {expected}")
    if not source.is_floating_point() or not bool(torch.isfinite(source).all()):
        raise ValueError("source must contain finite floating-point values")
    return source.to(device=device, dtype=torch.float32)


@torch.inference_mode()
def flow_sample(
    model,
    static,
    valid_mask,
    *,
    num_samples,
    ode_steps,
    source_sampling="stratified",
    generator=None,
    amp=True,
    final_step="euler",
    method="parallel",
    source=None,
):
    """Draw normalized paths using the checkpoint's sampling method."""
    if final_step != "euler":
        raise ValueError("Only Euler final updates are supported")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    validate_sampler(model, method, ode_steps)
    rows = static["history_embeds"].shape[0]
    horizon = static["future_length"]
    device = static["history_embeds"].device
    source = _sampling_source(
        source,
        rows,
        num_samples,
        horizon,
        sampling=source_sampling,
        device=device,
        generator=generator,
        scale=model.source_scale,
    ).reshape(rows * num_samples, horizon)
    if method == "euler":
        expanded = model.expand_static(static, num_samples)
        unknown = model.unknown_target_mask(
            expanded, valid_mask.repeat_interleave(num_samples, dim=0)
        )
        def blend(values):
            return model.blend_state(values, expanded, unknown)

        state = _euler_integrate(
            blend(source),
            lambda values, time: model.forward_prepared(values, time, expanded),
            blend, ode_steps, amp,
        )
        return state.reshape(rows, num_samples, horizon)
    return _parallel_sample(
        model,
        static,
        valid_mask,
        source=source,
        num_samples=num_samples,
        num_blocks=ode_steps,
        amp=amp,
    )


@torch.inference_mode()
def flow_sample_packed(
    model,
    static,
    valid_mask,
    *,
    num_samples,
    ode_steps,
    source_sampling="stratified",
    generator=None,
    amp=True,
    final_step="euler",
    method="parallel",
    history_cache=None,
    history_cache_min_tokens=HISTORY_CACHE_MIN_TOKENS,
    source=None,
):
    """Decode paths with a shared history prefix and optional history cache."""
    if final_step != "euler":
        raise ValueError("Only Euler final updates are supported")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    validate_sampler(model, method, ode_steps)
    rows = static["history_embeds"].shape[0]
    horizon = static["future_length"]
    device = static["history_embeds"].device
    packed = model.prepare_packed_draws(static, num_samples)
    if (
        history_cache is None
        and (method != "parallel" or ode_steps > 1)
        and history_cache_applies(static, history_cache_min_tokens)
    ):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            history_cache = model.encode_history(static)
    unknown = model.unknown_target_mask(static, valid_mask)
    source = _sampling_source(
        source,
        rows,
        num_samples,
        horizon,
        sampling=source_sampling,
        device=device,
        generator=generator,
        scale=model.source_scale,
    ).permute(1, 0, 2)
    states = model.blend_state(source, static, unknown)
    if method == "euler":
        cache = history_cache if isinstance(history_cache, dict) else None
        states = _euler_integrate(
            states,
            lambda values, time: model.forward_packed(
                values, time, packed, history_cache=cache
            ),
            lambda values: model.blend_state(values, static, unknown),
            ode_steps, amp,
        )
        return states.permute(1, 0, 2)
    return _parallel_sample_packed(
        model,
        static,
        packed,
        unknown,
        states=states,
        num_blocks=ode_steps,
        amp=amp,
        history_cache=history_cache if isinstance(history_cache, dict) else None,
    )


def stratified_gaussian(
    batch_size: int,
    num_samples: int,
    length: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-coordinate stratified standard normal draws.

    The returned tensor has shape ``(batch_size, num_samples, length)``.
    """
    if num_samples < 2:
        raise ValueError(
            "Stratified sampling requires at least two samples; a single stratum would return the zero vector"
        )
    ranks = (
        torch.arange(num_samples, device=device, dtype=torch.float32) + 0.5
    ) / num_samples
    values = torch.special.ndtri(ranks)
    permutations = torch.rand(
        batch_size, length, num_samples, device=device, generator=generator
    ).argsort(dim=-1)
    return values[permutations].movedim(-1, 1)


def draw_source(
    batch_size: int,
    num_samples: int,
    length: int,
    *,
    sampling: str,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Draw Gaussian sources in normalized arcsinh space.

    Apply the checkpoint's source standard deviation after generating noise.
    """
    if sampling not in SOURCE_SAMPLINGS:
        raise ValueError(f"Unknown source sampling {sampling!r}")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if sampling == "iid":
        draw = torch.randn(
            batch_size,
            num_samples,
            length,
            device=device,
            generator=generator,
            dtype=torch.float32,
        )
    else:
        draw = stratified_gaussian(
            batch_size, num_samples, length, device=device, generator=generator
        )
    return draw if scale == 1.0 else scale * draw


def history_cache_applies(static: dict[str, Any], min_tokens: int) -> bool:
    """Whether the history reaches the configured cache threshold."""
    rows = static["history_embeds"].shape[0]
    return rows * int(static["num_context_patches"]) >= int(min_tokens)


def parallel_block_length(model: PathFM, num_blocks: int) -> int:
    """Block length for ``num_blocks`` evaluations, refusing untrained lengths."""
    grid_size = model.parallel_grid_size
    if grid_size is None:
        raise ValueError("The model has no parallel heads")
    if num_blocks < 1 or grid_size % num_blocks:
        raise ValueError(
            f"{num_blocks} blocks do not tile the {grid_size}-interval grid"
        )
    length = grid_size // num_blocks
    if length not in model.parallel_block_menu:
        raise ValueError(
            f"block length {length} was not trained; the menu is {list(model.parallel_block_menu)}"
        )
    return length


@torch.no_grad()
def _parallel_sample(
    model: PathFM,
    static: dict[str, Any],
    valid_mask: torch.Tensor,
    *,
    source: torch.Tensor,
    num_samples: int,
    num_blocks: int,
    amp: bool,
) -> torch.Tensor:
    """Fused parallel-decoding blocks over an expanded (row, path) batch."""
    length = parallel_block_length(model, num_blocks)
    rows = static["history_embeds"].shape[0]
    future_length = static["future_length"]
    device = source.device
    expanded = model.expand_static(static, num_samples)
    unknown = model.unknown_target_mask(
        expanded, valid_mask.repeat_interleave(num_samples, dim=0)
    )

    def blend(values: torch.Tensor) -> torch.Tensor:
        """Hold the known coordinates fixed."""
        return model.blend_state(values, expanded, unknown)

    state = blend(source)
    for block in range(num_blocks):
        start = block * length
        _, start_time, end_time = model.parallel_block_weights(start, length)
        flow_time = torch.full(
            (state.shape[0],), start_time, device=device, dtype=torch.float32
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            velocity = model.forward_fused(
                state, flow_time, expanded, start=start, length=length
            ).float()
        state = blend(state + (end_time - start_time) * velocity)
    return state.reshape(rows, num_samples, future_length)


@torch.no_grad()
def _parallel_sample_packed(
    model: PathFM,
    static: dict[str, Any],
    packed: dict[str, Any],
    unknown: torch.Tensor,
    *,
    states: torch.Tensor,
    num_blocks: int,
    amp: bool,
    history_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Fused parallel-decoding blocks with every path packed behind one prefix."""
    length = parallel_block_length(model, num_blocks)
    num_samples, rows = states.shape[:2]
    device = states.device

    def blend(values: torch.Tensor) -> torch.Tensor:
        """Hold the known coordinates fixed."""
        return model.blend_state(values, static, unknown)

    for block in range(num_blocks):
        start = block * length
        _, start_time, end_time = model.parallel_block_weights(start, length)
        flow_times = torch.full(
            (num_samples, rows), start_time, device=device, dtype=torch.float32
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            velocity = model.forward_fused_packed(
                states,
                flow_times,
                packed,
                start=start,
                length=length,
                history_cache=history_cache,
            ).float()
        states = blend(states + (end_time - start_time) * velocity)
    return states.permute(1, 0, 2)


def denormalized_samples(
    model: PathFM, static: dict[str, Any], samples: torch.Tensor
) -> torch.Tensor:
    """Invert Chronos-2 instance normalisation for every sample path."""
    loc, scale = static["loc_scale"]
    rows, num_samples, horizon = samples.shape
    flat = samples.reshape(rows * num_samples, horizon)
    expanded = (
        loc.repeat_interleave(num_samples, dim=0),
        scale.repeat_interleave(num_samples, dim=0),
    )
    return model.denormalize(flat, expanded).reshape(rows, num_samples, horizon)
