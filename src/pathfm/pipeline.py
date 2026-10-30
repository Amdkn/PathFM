"""Path forecasts through the Chronos-2 input and quantile interfaces.

Chronos-2 handles validation, covariates, padding, batching, and DataFrames.
The flow sampler replaces its predictive distribution with coherent paths.
Known future covariates remain fixed and quantiles are read from those paths.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from chronos import Chronos2Pipeline

from .model import PathFM


class PathFMPipeline(Chronos2Pipeline):
    """Chronos-2 inference plumbing driven by a native-flow sampler."""

    @classmethod
    def from_pretrained(
        cls,
        model_id,
        *,
        device="cpu",
        revision=None,
        cache_dir=None,
        local_files_only=False,
        steps=None,
        num_samples=None,
        **kwargs,
    ):
        """Load a local export or Hugging Face model repository."""
        from .loading import load_model

        flow, config = load_model(
            model_id,
            device=device,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        defaults = dict(config.get("inference_defaults", {}))
        defaults.update(kwargs)
        default_steps = defaults.pop("steps", 1)
        defaults["ode_steps"] = steps if steps is not None else default_steps
        defaults["num_samples"] = (
            num_samples if num_samples is not None else defaults.get("num_samples", 32)
        )
        defaults.setdefault("amp", torch.device(device).type == "cuda")
        pipeline = cls(flow, **defaults)
        pipeline.release_config = config
        pipeline.model_id = str(model_id)
        return pipeline

    def __init__(
        self,
        flow: PathFM,
        *,
        num_samples: int = 32,
        ode_steps: int = 1,
        sampler_method: Literal["parallel", "euler"] = "parallel",
        source_sampling: str = "stratified",
        final_step: Literal["euler"] = "euler",
        token_budget: int = 131_072,
        amp: bool = True,
        seed: int = 42,
        packed_paths: bool | None = None,
        history_cache: bool = True,
        history_cache_min_tokens: int | None = None,
        max_output_patches: int | None = None,
    ):
        super().__init__(flow.backbone)
        if final_step != "euler":
            raise ValueError("Only Euler final updates are supported")
        from .sampling import validate_sampler

        validate_sampler(flow, sampler_method, ode_steps)
        if num_samples < 2:
            raise ValueError("At least two forecast paths are needed to read quantiles")
        from .sampling import (
            HISTORY_CACHE_MIN_TOKENS,
            supports_packed_paths,
        )

        if packed_paths is None:
            packed_paths = supports_packed_paths(flow)
        elif packed_paths and not supports_packed_paths(flow):
            raise ValueError(
                "packed_paths requires history-only attention, no time "
                "token, and a history-free modulation scope"
            )
        self.packed_paths = bool(packed_paths)
        # encode each window's history once and reuse it across every solver
        # evaluation and path chunk; only meaningful on the packed sampler,
        # and only worth it from history_cache_min_tokens history tokens up
        self.history_cache = bool(history_cache)
        self.history_cache_min_tokens = int(
            HISTORY_CACHE_MIN_TOKENS
            if history_cache_min_tokens is None
            else history_cache_min_tokens
        )
        self.flow = flow
        self.num_samples = int(num_samples)
        self.ode_steps = int(ode_steps)
        self.sampler_method = sampler_method
        self.source_sampling = source_sampling
        self.final_step = final_step
        if max_output_patches is not None:
            if not 1 <= int(max_output_patches) <= int(flow.max_output_patches):
                raise ValueError(
                    f"max_output_patches must lie in [1, "
                    f"{flow.max_output_patches}], got {max_output_patches}"
                )
        self._max_output_patches = (
            int(max_output_patches) if max_output_patches is not None else None
        )
        self.token_budget = int(token_budget)
        # set when a group was too wide to draw every path in one pass
        self._path_chunked = False
        self._remaining_horizon: int | None = None
        # when enabled, every step stashes its raw paths so a caller can
        # study point-forecast reductions the quantile summary discards
        self.capture_paths = False
        self._paths: list[torch.Tensor] = []
        self._paths_unrolled = False
        self.capture_samples = False
        self._captured_items = []
        self.amp = bool(amp)
        self.seed = int(seed)
        self._target_rows: torch.Tensor | None = None
        self._requested_horizon: int | None = None
        self._stream_key = 0
        self._call_index = 0

    def predict_samples(self, inputs, prediction_length=None, **kwargs):
        """Return one [targets, samples, horizon] tensor per input item.

        A direct forecast may cover at most ``model_prediction_length`` steps.
        Use ``rollout`` for independently continued longer sample paths.
        """
        horizon = (
            self.model_prediction_length
            if prediction_length is None
            else prediction_length
        )
        if horizon < 1:
            raise ValueError("prediction_length must be positive")
        if horizon > self.model_prediction_length:
            raise ValueError("Use rollout() for sample paths beyond the native horizon")
        if self._captured_items:
            raise RuntimeError("Unconsumed benchmark samples remain in the pipeline")
        self.capture_samples = True
        try:
            self.predict(inputs, prediction_length=horizon, **kwargs)
            result, self._captured_items = self._captured_items, []
            return result
        except BaseException:
            self._captured_items = []
            self._paths = []
            self._paths_unrolled = False
            raise
        finally:
            self.capture_samples = False
            self.capture_paths = False

    def take_samples(self, count):
        """Consume the next benchmark window's captured input items."""
        if count > len(self._captured_items):
            raise RuntimeError("The requested forecast samples were not captured")
        items = self._captured_items[:count]
        del self._captured_items[:count]
        return items

    def rollout(
        self,
        inputs,
        prediction_length,
        *,
        num_samples=None,
        steps=None,
        seed=None,
        max_rows=4608,
        max_abs_value=1e6,
    ):
        """Return independently continued [batch, paths, channels, horizon] samples."""
        from .rollout import rollout

        return rollout(
            self.flow,
            inputs,
            prediction_length,
            num_samples=self.num_samples if num_samples is None else num_samples,
            steps=self.ode_steps if steps is None else steps,
            sampler_method=self.sampler_method,
            seed=self.seed if seed is None else seed,
            source_sampling=self.source_sampling,
            max_rows=max_rows,
            amp=self.amp,
            max_abs_value=max_abs_value,
        )

    # ------------------------------------------------------------------
    def set_stream(self, key: int) -> None:
        """Start a fresh, reproducible noise stream for one task.

        Sampler noise is keyed by ``(seed, stream key, call index)``, so a task's
        forecasts do not depend on how many tasks preceded it. Callers set one
        stream per task, which makes results independent of task sharding.
        """

        self._stream_key = int(key)
        self._call_index = 0
        self._path_chunked = False

    def sampler_config(self) -> dict[str, Any]:
        from .sampling import sampler_model_evaluations

        return {
            "num_samples": self.num_samples,
            "ode_steps": self.ode_steps,
            "sampler_method": self.sampler_method,
            "max_output_patches": self.max_output_patches,
            "source_sampling": self.source_sampling,
            "sampling_protocol": "global_strata_v1",
            "final_step": self.final_step,
            "model_evaluations_per_path": sampler_model_evaluations(
                self.sampler_method,
                self.ode_steps,
            ),
            "amp": self.amp,
            "seed": self.seed,
            "token_budget": self.token_budget,
            "path_chunked": self._path_chunked,
            "packed_paths": self.packed_paths,
            "history_cache": self.history_cache,
            "history_cache_min_tokens": self.history_cache_min_tokens,
        }

    def _predict_batch(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        unrolled_quantiles_tensor: torch.Tensor,
        prediction_length: int,
        max_output_patches: int,
        target_idx_ranges: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        mask = torch.zeros(context.shape[0], dtype=torch.bool)
        for start, end in target_idx_ranges:
            mask[start:end] = True
        if not bool(mask.any()):
            raise ValueError("A prediction batch has no target rows")
        self._target_rows = mask.to(self.model.device)
        self._requested_horizon = int(prediction_length)
        # Horizons longer than the flow's patch limit are produced in several
        # chunks by the released long-horizon heuristic, so the coordinates
        # still wanted have to be counted down rather than read from the total.
        self._remaining_horizon = int(prediction_length)
        if self.capture_samples:
            self.capture_paths = True
        if self.capture_paths:
            self._paths_unrolled |= prediction_length > (
                max_output_patches * self.model_output_patch_size
            )
        try:
            result = super()._predict_batch(
                context=context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                unrolled_quantiles_tensor=unrolled_quantiles_tensor,
                prediction_length=prediction_length,
                max_output_patches=max_output_patches,
                target_idx_ranges=target_idx_ranges,
            )
            if self.capture_samples:
                paths = self.drain_paths()[..., :prediction_length]
                self._captured_items.extend(
                    paths[start:end].clone() for start, end in target_idx_ranges
                )
            return result
        finally:
            self._target_rows = None
            self._requested_horizon = None
            self._remaining_horizon = None

    def drain_paths(self) -> torch.Tensor:
        """The raw forecast paths of the last prediction, shape [rows, paths, h].

        Only valid when the horizon fitted in one forward. Under long-horizon
        unrolling each chunk re-samples from a fed-back trajectory, so the
        per-chunk paths are not draws from one joint distribution and stacking
        them would invent a coherence the sampler never produced. That case
        raises rather than returning something misleading.
        Ordinary input batches are concatenated along the row dimension.
        """

        captures, unrolled = self._paths, self._paths_unrolled
        self._paths = []
        self._paths_unrolled = False
        if not captures:
            raise RuntimeError("no paths captured; set capture_paths first")
        if unrolled:
            raise RuntimeError(
                "the horizon was produced in unrolled chunks, so the "
                "paths are not a joint sample; restrict this study to horizons "
                "that fit one forward"
            )
        return captures[0] if len(captures) == 1 else torch.cat(captures, dim=0)

    @property
    def model_prediction_length(self) -> int:
        """The effective flow block length used by defaults and the public guard."""

        return self.max_output_patches * self.model_output_patch_size

    @property
    def max_output_patches(self) -> int:
        """Limit each forward pass to the exported model's native block.

        Chronos-2 unrolls longer quantile forecasts using its long-horizon rule.
        ``max_output_patches`` can explicitly select a shorter block.
        """

        if self._max_output_patches is not None:
            return int(self._max_output_patches)
        return int(self.flow.max_output_patches)

    def _sample_chunks(self, rows: int, sequence_length: int) -> list[int]:
        """Forecast-path counts to draw per pass for one row chunk.

        Group attention keeps every task's channels together.
        Split paths when one group exceeds the token budget.
        Stratified paths share one source ensemble across all passes.
        """

        per_pass = self.token_budget // max(rows * sequence_length, 1)
        if per_pass >= self.num_samples:
            return [self.num_samples]
        per_pass = max(1, per_pass)
        counts = [per_pass] * (self.num_samples // per_pass)
        remainder = self.num_samples % per_pass
        if remainder:
            counts.append(remainder)
        return counts

    def _sample_chunks_packed(
        self,
        rows: int,
        history_patches: int,
        appended_per_path: int,
    ) -> list[int]:
        """Path counts per pass when the paths share one packed sequence.

        A packed pass costs ``rows * (history + paths * appended)`` tokens:
        the paths extend the sequence instead of multiplying the batch, so
        the budget bounds the path count linearly after the shared history
        is paid once.
        """

        per_row = self.token_budget // max(rows, 1)
        per_pass = (per_row - history_patches) // max(appended_per_path, 1)
        if per_pass >= self.num_samples:
            return [self.num_samples]
        per_pass = max(1, per_pass)
        counts = [per_pass] * (self.num_samples // per_pass)
        remainder = self.num_samples % per_pass
        if remainder:
            counts.append(remainder)
        return counts

    def _group_chunks(
        self,
        group_ids: torch.Tensor,
        tokens_per_row: int,
    ) -> list[torch.Tensor]:
        """Row index chunks that keep every Chronos-2 group intact."""

        max_rows = max(
            1,
            self.token_budget // max(tokens_per_row, 1),
        )
        order = torch.argsort(group_ids, stable=True)
        sorted_ids = group_ids[order]
        boundaries = torch.unique_consecutive(
            sorted_ids,
            return_counts=True,
        )[1].tolist()
        chunks: list[torch.Tensor] = []
        current: list[torch.Tensor] = []
        rows = 0
        cursor = 0
        for size in boundaries:
            indices = order[cursor : cursor + size]
            cursor += size
            if current and rows + size > max_rows:
                chunks.append(torch.cat(current))
                current = []
                rows = 0
            current.append(indices)
            rows += size
        if current:
            chunks.append(torch.cat(current))
        return chunks

    @torch.inference_mode()
    def _predict_step(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor | None,
        num_output_patches: int,
    ) -> torch.Tensor:
        from .sampling import (
            denormalized_samples,
            draw_source,
            flow_sample,
            flow_sample_packed,
            history_cache_applies,
        )

        if self._target_rows is None or self._requested_horizon is None:
            raise RuntimeError("_predict_step was called outside a batch")
        device = self.model.device
        patch = self.model_output_patch_size
        output_size = num_output_patches * patch
        rows = context.shape[0]
        target_rows = self._target_rows
        if len(target_rows) != rows:
            # The unrolling heuristic replicates every row once per unrolled
            # quantile trajectory and gives each replica its own group, so the
            # captured mask has to be expanded by that exact factor. Anything
            # else is a real mismatch.
            if rows % len(target_rows):
                raise RuntimeError(
                    "The captured target mask does not match the batch: "
                    f"{len(target_rows)} rows captured, {rows} received"
                )
            target_rows = target_rows.repeat_interleave(rows // len(target_rows))
        if future_covariates is None:
            covariates = torch.full(
                (rows, output_size),
                float("nan"),
                device=device,
                dtype=torch.float32,
            )
        else:
            covariates = future_covariates.to(device=device, dtype=torch.float32)
            if covariates.shape[-1] < output_size:
                padding = torch.full(
                    (rows, output_size - covariates.shape[-1]),
                    float("nan"),
                    device=device,
                    dtype=torch.float32,
                )
                covariates = torch.cat([covariates, padding], dim=-1)
            else:
                covariates = covariates[..., :output_size]
        known_mask = torch.isfinite(covariates).to(torch.float32)
        known_values = torch.nan_to_num(covariates, nan=0.0)
        horizon = min(self._remaining_horizon, output_size)
        steps = torch.arange(output_size, device=device)
        inside = (steps < horizon).to(torch.float32).unsqueeze(0)
        valid = target_rows.to(torch.float32).unsqueeze(1) * inside
        # this chunk consumes what it emits, so the next one masks its own tail
        self._remaining_horizon = max(0, self._remaining_horizon - output_size)

        levels = self.model.quantiles.detach().float().cpu().numpy()
        captured: torch.Tensor | None = None
        predictions = torch.zeros(
            rows,
            len(levels),
            output_size,
            dtype=torch.float32,
        )
        sequence_length = context.shape[-1] // patch + 1 + num_output_patches
        if self.packed_paths:
            # Paths extend the packed sequence instead of multiplying the
            # batch, so a row costs the shared history plus one appended
            # block per path.
            tokens_per_row = context.shape[-1] // patch + self.num_samples * (
                1 + num_output_patches
            )
        else:
            tokens_per_row = sequence_length * self.num_samples
        for chunk in self._group_chunks(group_ids, tokens_per_row):
            static = self.flow.prepare_static(
                context[chunk].to(device=device, dtype=torch.float32),
                num_output_patches=num_output_patches,
                group_ids=group_ids[chunk].to(device),
                future_covariates=known_values[chunk],
                future_covariates_mask=known_mask[chunk],
                requested_horizon=horizon,
            )
            generator = torch.Generator(device=device).manual_seed(
                self.seed * 1_000_003 + self._stream_key * 10_007 + self._call_index
            )
            self._call_index += 1
            # Packing pays once the history re-encode dominates: measured, a
            # single row at 128 context patches runs 0.68x packed (one short
            # packed sequence underfills the GPU against a 32-path batch)
            # while everything wider or longer wins, up to 11.9x. Fall back
            # to the expanded sampler in the one losing regime.
            packed_here = self.packed_paths and (
                len(chunk) > 1 or static["num_context_patches"] >= 256
            )
            if packed_here:
                counts = self._sample_chunks_packed(
                    len(chunk),
                    static["num_context_patches"],
                    1 + num_output_patches,
                )
            else:
                counts = self._sample_chunks(len(chunk), sequence_length)
            sampling = self.source_sampling
            source = None
            if len(counts) > 1 and sampling == "stratified":
                # Stratify the complete ensemble once, then split computation.
                source = draw_source(
                    len(chunk), self.num_samples, output_size,
                    sampling=sampling, device=device, generator=generator,
                    scale=self.flow.source_scale,
                )
            if len(counts) > 1:
                self._path_chunked = True
            sampler = flow_sample_packed if packed_here else flow_sample
            options = {}
            if packed_here:
                # one history encode per window, shared by every path chunk,
                # under the sampler's own precision; tiny windows skip it
                options["history_cache"] = False
                if (
                    self.history_cache
                    and history_cache_applies(
                        static,
                        self.history_cache_min_tokens,
                    )
                    and (
                        self.sampler_method != "parallel"
                        or self.ode_steps > 1
                        or len(counts) > 1
                    )
                ):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=self.amp,
                    ):
                        options["history_cache"] = self.flow.encode_history(static)
            pieces = []
            start = 0
            for count in counts:
                source_options = (
                    {"source": source[:, start:start + count]} if source is not None else {}
                )
                pieces.append(sampler(
                    self.flow,
                    static,
                    valid[chunk],
                    num_samples=count,
                    ode_steps=self.ode_steps,
                    source_sampling=sampling,
                    generator=generator,
                    amp=self.amp,
                    final_step=self.final_step,
                    method=self.sampler_method,
                    **options,
                    **source_options,
                ))
                start += count
            # one call when the budget allows, so the usual path is unchanged
            normalized = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
            paths = denormalized_samples(self.flow, static, normalized)
            if self.capture_paths:
                if captured is None:
                    captured = torch.zeros(
                        rows, paths.shape[1], output_size, dtype=torch.float32
                    )
                captured[chunk.cpu()] = paths.float().cpu()
            values = np.quantile(
                paths.float().cpu().numpy().astype(np.float64),
                levels,
                axis=1,
            ).transpose(1, 0, 2)
            predictions[chunk.cpu()] = torch.from_numpy(values).float()
        if not torch.isfinite(predictions[target_rows.cpu()]).all():
            raise FloatingPointError("The flow produced non-finite forecasts")
        if captured is not None:
            self._paths.append(captured)
        return predictions.to(device=context.device, dtype=context.dtype)
