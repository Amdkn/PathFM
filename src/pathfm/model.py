"""PathFM inference architecture with velocity or fused parallel heads."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from chronos.chronos2.layers import Chronos2RotaryEmbedding
from chronos.chronos2.model import Chronos2Encoder, Chronos2Model
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .normalization import FlooredInstanceNorm


class FlowTimeEmbedding(nn.Module):
    """Sundial-style sinusoidal embedding for scalar flow time."""

    def __init__(self, hidden_dim: int, feature_dim: int = 256):
        super().__init__()
        if feature_dim % 2:
            raise ValueError("feature_dim must be even")
        half_dim = feature_dim // 2
        self.register_buffer(
            "frequencies",
            torch.exp(
                -math.log(10000)
                * torch.arange(half_dim, dtype=torch.float32)
                / half_dim
            ),
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        angles = time * 1000 * self.frequencies.to(device=time.device, dtype=time.dtype)
        return self.projection(torch.cat([angles.cos(), angles.sin()], dim=-1))


def validate_parallel_knots(knots: Sequence[float]) -> tuple[float, ...]:
    """Strictly increasing flow-time knots running from 0 to 1."""
    values = tuple((float(value) for value in knots))
    if len(values) < 2:
        raise ValueError("parallel_knots needs at least two knots")
    if abs(values[0]) > 1e-09 or abs(values[-1] - 1.0) > 1e-09:
        raise ValueError("parallel_knots must run from 0 to 1")
    if any((b <= a for a, b in zip(values, values[1:]))):
        raise ValueError("parallel_knots must be strictly increasing")
    return (0.0,) + values[1:-1] + (1.0,)


def parallel_block_lengths(
    menu: Sequence[int] | None, grid_size: int
) -> tuple[int, ...]:
    """Block lengths a parallel decoder serves; every divisor by default."""
    if menu is None:
        return tuple(
            (length for length in range(1, grid_size + 1) if grid_size % length == 0)
        )
    values = tuple(sorted({int(value) for value in menu}))
    for length in values:
        if length < 1 or grid_size % length:
            raise ValueError(
                f"block length {length} must divide the grid size {grid_size}"
            )
    return values


class QKNormProjection(nn.Module):
    """Query/key projection with per-head RMS normalization."""

    def __init__(self, linear: nn.Linear, num_heads: int, head_dim: int):
        super().__init__()
        self.linear = linear
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.temperature = nn.Parameter(torch.empty(self.num_heads))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.linear(hidden)
        shape = projected.shape
        heads = projected.reshape(*shape[:-1], self.num_heads, self.head_dim)
        rms = heads.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        scale = (self.temperature / rms.squeeze(-1)).to(heads.dtype)
        return (heads * scale.unsqueeze(-1)).reshape(shape)


class PathFM(nn.Module):
    @staticmethod
    def _single_row_group_attention(values, attention):
        return attention.o(attention.v(values))

    @staticmethod
    def _single_row_shortcut_applies(attention):
        return (
            float(getattr(attention, "dropout", 0.0)) == 0.0 or not attention.training
        )

    def __init__(
        self,
        backbone: Chronos2Model,
        *,
        max_output_patches: int,
        flow_entry_block: int,
        source_scale: float,
        instance_scale_floor: float = 0.0001,
        parallel_knots: Sequence[float] | None = None,
        parallel_block_menu: Sequence[int] | None = None,
    ):
        super().__init__()
        if source_scale <= 0 or instance_scale_floor <= 0:
            raise ValueError("Source scale and normalization floor must be positive")
        self.backbone = backbone
        self.source_scale = float(source_scale)
        self.instance_scale_floor = float(instance_scale_floor)
        self.backbone.instance_norm = FlooredInstanceNorm(
            backbone.instance_norm, instance_scale_floor
        )
        self.patch_length = int(backbone.chronos_config.input_patch_size)
        self.max_output_patches = int(max_output_patches)
        if (
            not 1
            <= self.max_output_patches
            <= backbone.chronos_config.max_output_patches
        ):
            raise ValueError("Invalid maximum output patch count")
        self.max_horizon = self.max_output_patches * self.patch_length
        self.horizon = self.max_horizon
        self.num_future_patches = self.max_output_patches
        self.hidden_dim = int(backbone.config.d_model)
        self.flow_entry_block = int(flow_entry_block)
        self.num_blocks = len(backbone.encoder.block)
        if not 0 < self.flow_entry_block < self.num_blocks:
            raise ValueError("Flow entry must be inside the encoder")
        self.flow_time_embedding = FlowTimeEmbedding(self.hidden_dim)
        self.gate_projection = nn.Linear(self.hidden_dim, 1)
        self.final_modulation = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.adaln_shared = nn.Linear(self.hidden_dim, 6 * self.hidden_dim)
        self.adaln_block_offsets = nn.Parameter(
            torch.empty(self.num_blocks, 6 * self.hidden_dim)
        )
        trunk = backbone.output_patch_embedding
        if parallel_knots is None:
            self.parallel_knots = None
            self.parallel_block_menu = ()
            self.clean_output_layer = nn.Linear(
                trunk.output_layer.in_features, self.patch_length
            )
            self.clean_residual_layer = nn.Linear(
                trunk.residual_layer.in_features, self.patch_length
            )
        else:
            self.parallel_knots = validate_parallel_knots(parallel_knots)
            grid_size = len(self.parallel_knots) - 1
            self.parallel_block_menu = parallel_block_lengths(
                parallel_block_menu, grid_size
            )
            self.parallel_output_layer = nn.Linear(
                trunk.output_layer.in_features, grid_size * self.patch_length
            )
            self.parallel_residual_layer = nn.Linear(
                trunk.residual_layer.in_features, grid_size * self.patch_length
            )
        trunk.output_layer = nn.Identity()
        trunk.residual_layer = nn.Identity()
        self._apply_qk_norm()

    def _apply_qk_norm(self) -> None:
        """Wrap every backbone attention's query and key projections."""
        wrapped = 0
        for block in self.backbone.encoder.block:
            for layer in block.layer:
                attention = getattr(layer, "self_attention", None)
                if attention is None or not hasattr(attention, "q"):
                    continue
                for name in ("q", "k"):
                    setattr(
                        attention,
                        name,
                        QKNormProjection(
                            getattr(attention, name),
                            num_heads=int(attention.n_heads),
                            head_dim=int(attention.kv_proj_dim),
                        ),
                    )
                    wrapped += 1
        if not wrapped:
            raise RuntimeError("No attention projections found to wrap")

    def normalize_target(
        self, target: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        normalized, _ = self.backbone.instance_norm(target, loc_scale)
        return normalized

    def denormalize(
        self, values: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.backbone.instance_norm.inverse(values, loc_scale)

    def _group_layout(
        self, group_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, Any]:
        """Describe an exact block-diagonal form of the group attention."""
        order = torch.argsort(group_ids, stable=True)
        sorted_ids = group_ids[order]
        _, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        if counts.numel() == 0 or not bool(torch.all(counts == counts[0])):
            return {"uniform": False, "reason": "ragged group sizes"}
        group_size = int(counts[0])
        num_groups = int(counts.numel())
        sequence_length = int(attention_mask.shape[-1])
        permuted = attention_mask[order].reshape(
            num_groups, group_size, sequence_length
        )
        key_valid = (
            permuted.permute(0, 2, 1).reshape(num_groups * sequence_length, group_size)
            > 0
        )
        empty = ~key_valid.any(dim=-1, keepdim=True)
        key_valid = key_valid | empty
        mask = torch.where(
            key_valid,
            torch.zeros((), dtype=attention_mask.dtype, device=key_valid.device),
            torch.full(
                (),
                torch.finfo(attention_mask.dtype).min,
                dtype=attention_mask.dtype,
                device=key_valid.device,
            ),
        )
        return {
            "uniform": True,
            "order": order,
            "inverse": torch.argsort(order),
            "num_groups": num_groups,
            "group_size": group_size,
            "mask": mask[:, None, None, :],
            "fully_masked_positions": int(empty.sum()),
        }

    def _time_attention_mask(
        self,
        attention_mask: torch.Tensor,
        num_context_patches: int,
        floating_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Additive mask for the time-axis attention."""
        sequence_length = int(attention_mask.shape[-1])
        device = attention_mask.device
        is_history = torch.arange(sequence_length, device=device) < num_context_patches
        allowed = (~is_history)[:, None] | is_history[None, :]
        key_valid = allowed[None] & (attention_mask > 0)[:, None, :]
        empty = ~key_valid.any(dim=-1, keepdim=True)
        key_valid = key_valid | empty & is_history[None, None, :]
        mask = torch.where(
            key_valid,
            torch.zeros((), dtype=floating_dtype, device=device),
            torch.full(
                (), torch.finfo(floating_dtype).min, dtype=floating_dtype, device=device
            ),
        )
        return mask.unsqueeze(1)

    def prepare_static(
        self,
        context: torch.Tensor,
        *,
        num_output_patches: int | None = None,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        requested_horizon: int | None = None,
    ) -> dict[str, Any]:
        """Embed the state-independent part of the native sequence."""
        backbone = self.backbone
        if context.ndim != 2:
            raise ValueError("context must have shape (rows, context_length)")
        if num_output_patches is None:
            num_output_patches = self.max_output_patches
        num_output_patches = int(num_output_patches)
        if not 1 <= num_output_patches <= self.max_output_patches:
            raise ValueError(
                f"num_output_patches must lie in [1, {self.max_output_patches}], got {num_output_patches}"
            )
        batch_size = context.shape[0]
        future_length = num_output_patches * self.patch_length
        if requested_horizon is None:
            requested_horizon = future_length
        requested_horizon = int(requested_horizon)
        if not 1 <= requested_horizon <= future_length:
            raise ValueError(
                f"requested_horizon must lie in [1, {future_length}], got {requested_horizon}"
            )
        patched_context, attention_mask, loc_scale = backbone._prepare_patched_context(
            context=context, context_mask=context_mask
        )
        history_embeds = backbone.input_patch_embedding(patched_context)
        register_ids = torch.full(
            (batch_size, 1),
            backbone.config.reg_token_id,
            device=history_embeds.device,
            dtype=torch.long,
        )
        register_embeds = backbone.shared(register_ids)
        patched_future, patched_known_mask = backbone._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        blank_future_embeds = backbone.input_patch_embedding(patched_future)
        known_mask = patched_known_mask.reshape(batch_size, future_length)
        known_component = patched_future[
            ..., self.patch_length : 2 * self.patch_length
        ].reshape(batch_size, future_length)
        appended_tokens = 1 + num_output_patches
        full_mask = torch.cat(
            [
                attention_mask.to(history_embeds.dtype),
                torch.ones(
                    batch_size,
                    appended_tokens,
                    dtype=history_embeds.dtype,
                    device=history_embeds.device,
                ),
            ],
            dim=-1,
        )
        if group_ids is None:
            group_ids = torch.arange(
                batch_size, dtype=torch.long, device=history_embeds.device
            )
        elif group_ids.shape != (batch_size,):
            raise ValueError("group_ids must have shape (rows,)")
        num_context_patches = int(attention_mask.shape[-1])
        sequence_length = num_context_patches + appended_tokens
        floating_dtype = history_embeds.dtype
        layout = self._group_layout(group_ids, full_mask)
        static = {
            "history_embeds": history_embeds,
            "register_embeds": register_embeds,
            "blank_future_embeds": blank_future_embeds,
            "patched_future": patched_future,
            "known_mask": known_mask.to(torch.float32),
            "known_component": known_component.to(torch.float32),
            "attention_mask": full_mask,
            "extended_attention_mask": self._time_attention_mask(
                full_mask, num_context_patches, floating_dtype
            ),
            "group_time_mask": None
            if layout.get("uniform")
            else Chronos2Encoder._construct_and_invert_group_time_mask(
                group_ids, full_mask, floating_dtype
            ),
            "position_ids": torch.arange(
                sequence_length, dtype=torch.long, device=history_embeds.device
            ).unsqueeze(0),
            "register_index": num_context_patches,
            "sequence_length": sequence_length,
            "num_context_patches": num_context_patches,
            "num_output_patches": num_output_patches,
            "future_length": future_length,
            "requested_horizon": requested_horizon,
            "group_ids": group_ids,
            "group_layout": layout,
            "loc_scale": loc_scale,
        }
        if self.flow_entry_block:
            static["entry_hidden"] = self._encode_entry(
                torch.cat(
                    [history_embeds, register_embeds, blank_future_embeds], dim=-2
                ),
                static,
            )
        return static

    def expand_static(self, static: dict[str, Any], count: int) -> dict[str, Any]:
        """Repeat every row of a prepared prefix ``count`` times."""
        if count < 1:
            raise ValueError("count must be positive")
        if count == 1:
            return static
        batch_size = static["history_embeds"].shape[0]

        def repeat(values: torch.Tensor) -> torch.Tensor:
            return values.repeat_interleave(count, dim=0)

        group_ids = static["group_ids"].repeat_interleave(count, dim=0)
        sample_index = torch.arange(
            count, device=group_ids.device, dtype=torch.long
        ).repeat(batch_size)
        group_ids = group_ids * count + sample_index
        attention_mask = repeat(static["attention_mask"])
        floating_dtype = static["history_embeds"].dtype
        loc, scale = static["loc_scale"]
        layout = self._group_layout(group_ids, attention_mask)
        entry = (
            {"entry_hidden": repeat(static["entry_hidden"])}
            if "entry_hidden" in static
            else {}
        )
        return {
            **static,
            **entry,
            "history_embeds": repeat(static["history_embeds"]),
            "register_embeds": repeat(static["register_embeds"]),
            "blank_future_embeds": repeat(static["blank_future_embeds"]),
            "patched_future": repeat(static["patched_future"]),
            "known_mask": repeat(static["known_mask"]),
            "known_component": repeat(static["known_component"]),
            "attention_mask": attention_mask,
            "extended_attention_mask": self._time_attention_mask(
                attention_mask, static["num_context_patches"], floating_dtype
            ),
            "group_time_mask": None
            if layout.get("uniform")
            else Chronos2Encoder._construct_and_invert_group_time_mask(
                group_ids, attention_mask, floating_dtype
            ),
            "group_ids": group_ids,
            "group_layout": layout,
            "loc_scale": (repeat(loc), repeat(scale)),
        }

    def unknown_target_mask(
        self, static: dict[str, Any], valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Coordinates the flow transports: valid and not known in advance."""
        expected = (static["known_mask"].shape[0], static["future_length"])
        if tuple(valid_mask.shape) != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        return valid_mask.to(torch.float32) * (1.0 - static["known_mask"])

    def blend_state(
        self, state: torch.Tensor, static: dict[str, Any], unknown_mask: torch.Tensor
    ) -> torch.Tensor:
        """Hold known covariates fixed and zero every masked coordinate."""
        return unknown_mask * state + static["known_component"]

    def gate(self, time_embedding: torch.Tensor) -> torch.Tensor:
        """Return one scalar gate per example."""
        return self.gate_projection(time_embedding).squeeze(-1)
        batch_size = time_embedding.shape[0]
        value = self.fixed_gate
        return value.to(time_embedding.dtype).expand(batch_size)

    def future_tokens(
        self, state: torch.Tensor, static: dict[str, Any], gate: torch.Tensor
    ) -> torch.Tensor:
        """Mix the blank-future embedding with the flow-state embedding."""
        blank = static["blank_future_embeds"]
        batch_size = blank.shape[0]
        future_length = static["future_length"]
        if state.shape != (batch_size, future_length):
            raise ValueError(
                f"state must have shape {(batch_size, future_length)}, got {tuple(state.shape)}"
            )
        patched_state = state.reshape(
            batch_size, static["num_output_patches"], self.patch_length
        ).to(static["patched_future"].dtype)
        patched = torch.cat(
            [
                static["patched_future"][..., : self.patch_length],
                patched_state,
                static["patched_future"][..., 2 * self.patch_length :],
            ],
            dim=-1,
        )
        state_embeds = self.backbone.input_patch_embedding(patched)
        return blank + gate.to(blank.dtype)[:, None, None] * (state_embeds - blank)

    def _modulation_mask(self, static: dict[str, Any]) -> torch.Tensor | None:
        mask = torch.ones(
            static["sequence_length"],
            device=static["history_embeds"].device,
            dtype=static["history_embeds"].dtype,
        )
        mask[: static["register_index"]] = 0.0
        return mask
        mask[static["register_index"]] = 0.0
        return mask

    @staticmethod
    def _modulate_time_major(
        values: torch.Tensor,
        norm: nn.Module,
        scale: torch.Tensor,
        shift: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = norm(values)
        scale = scale.unsqueeze(1).to(normalized.dtype)
        shift = shift.unsqueeze(1).to(normalized.dtype)
        if mask is not None:
            token_mask = mask.to(normalized.dtype)[None, :, None]
            scale = scale * token_mask
            shift = shift * token_mask
        return normalized * (1 + scale) + shift

    @staticmethod
    def _modulate_batch_major(
        values: torch.Tensor,
        norm: nn.Module,
        scale: torch.Tensor,
        shift: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = norm(values)
        scale = scale.unsqueeze(0).to(normalized.dtype)
        shift = shift.unsqueeze(0).to(normalized.dtype)
        if mask is not None:
            token_mask = mask.to(normalized.dtype)[:, None, None]
            scale = scale * token_mask
            shift = shift * token_mask
        return normalized * (1 + scale) + shift

    @classmethod
    def _grouped_attention(
        cls, values: torch.Tensor, attention: nn.Module, layout: dict[str, Any]
    ) -> torch.Tensor:
        """Batched group attention over the rows of each Chronos-2 group."""
        rows, sequence_length, hidden_dim = values.shape
        num_groups = layout["num_groups"]
        group_size = layout["group_size"]
        if group_size == 1 and cls._single_row_shortcut_applies(attention):
            return cls._single_row_group_attention(values, attention)
        grouped = (
            values[layout["order"]]
            .reshape(num_groups, group_size, sequence_length, hidden_dim)
            .permute(0, 2, 1, 3)
            .reshape(num_groups * sequence_length, group_size, hidden_dim)
        )
        attended = attention(grouped, mask=layout["mask"].to(grouped.dtype))[0]
        return (
            attended.reshape(num_groups, sequence_length, group_size, hidden_dim)
            .permute(0, 2, 1, 3)
            .reshape(rows, sequence_length, hidden_dim)[layout["inverse"]]
        )

    @staticmethod
    def _time_attention(
        attention: nn.Module,
        normed: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        prefix: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chronos-2 time attention with an optional cached key/value prefix."""
        heads, width = (attention.n_heads, attention.kv_proj_dim)
        query = rearrange(attention.q(normed), "b s (h d) -> b h s d", h=heads, d=width)
        key = rearrange(attention.k(normed), "b s (h d) -> b h s d", h=heads, d=width)
        value = rearrange(attention.v(normed), "b s (h d) -> b h s d", h=heads, d=width)
        cos, sin = attention.rope_embed(value, position_ids)
        query, key = Chronos2RotaryEmbedding.apply_rotary_pos_emb(query, key, cos, sin)
        if prefix is None:
            keys, values = (key, value)
        else:
            keys = torch.cat([prefix[0], key], dim=2)
            values = torch.cat([prefix[1], value], dim=2)
        if attention.config._attn_implementation == "sdpa":
            attended, _ = attention._sdpa_attention(query, keys, values, mask)
        else:
            attended, _ = attention._eager_attention(query, keys, values, mask)
        attended = rearrange(attended, "b h s d -> b s (h d)")
        return (attention.o(attended), key, value)

    def _encode_entry(
        self, hidden: torch.Tensor, static: dict[str, Any]
    ) -> torch.Tensor:
        """Run the flow-free entry blocks with full attention."""
        encoder = self.backbone.encoder
        base_mask = Chronos2Encoder._expand_and_invert_time_attention_mask(
            static["attention_mask"], hidden.dtype
        )
        hidden = encoder.dropout(hidden)
        for index in range(self.flow_entry_block):
            block = encoder.block[index]
            time_layer = block.layer[0]
            normed = time_layer.layer_norm(hidden)
            attention = time_layer.self_attention(
                normed, position_ids=static["position_ids"], mask=base_mask
            )
            hidden = hidden + time_layer.dropout(attention[0])
            group_layer = block.layer[1]
            layout = static["group_layout"]
            if layout.get("uniform"):
                normed = group_layer.layer_norm(hidden)
                attended = self._grouped_attention(
                    normed, group_layer.self_attention, layout
                )
                hidden = hidden + group_layer.dropout(attended)
            else:
                transposed = rearrange(hidden, "batch time d -> time batch d")
                normed = group_layer.layer_norm(transposed)
                attention = group_layer.self_attention(
                    normed, mask=static["group_time_mask"]
                )
                transposed = transposed + group_layer.dropout(attention[0])
                hidden = rearrange(transposed, "time batch d -> batch time d")
            ffn_layer = block.layer[2]
            normed = ffn_layer.layer_norm(hidden)
            hidden = hidden + ffn_layer.dropout(ffn_layer.mlp(normed))
        return hidden

    def encode_sequence(
        self,
        hidden: torch.Tensor,
        static: dict[str, Any],
        time_embedding: torch.Tensor,
        *,
        start_block: int = 0,
    ) -> torch.Tensor:
        """Run the native encoder with AdaRMSNorm flow-time modulation."""
        backbone = self.backbone
        encoder = backbone.encoder
        mask = self._modulation_mask(static)
        zero = torch.zeros(
            hidden.shape[0], self.hidden_dim, device=hidden.device, dtype=hidden.dtype
        )
        if start_block == 0:
            hidden = encoder.dropout(hidden)
        if self.adaln_shared is not None:
            shared_modulation = self.adaln_shared(time_embedding)
        for index, block in enumerate(encoder.block):
            if index < start_block:
                continue
            if self.adaln_shared is not None:
                parameters = (
                    shared_modulation + self.adaln_block_offsets[index]
                ).chunk(6, dim=-1)
            else:
                parameters = (zero,) * 6
            time_scale, time_shift, group_scale, group_shift, ffn_scale, ffn_shift = (
                parameters
            )
            time_layer = block.layer[0]
            normed = self._modulate_time_major(
                hidden, time_layer.layer_norm, time_scale, time_shift, mask
            )
            attention = time_layer.self_attention(
                normed,
                position_ids=static["position_ids"],
                mask=static["extended_attention_mask"],
            )
            hidden = hidden + time_layer.dropout(attention[0])
            group_layer = block.layer[1]
            layout = static["group_layout"]
            if layout.get("uniform"):
                normed = self._modulate_time_major(
                    hidden, group_layer.layer_norm, group_scale, group_shift, mask
                )
                attended = self._grouped_attention(
                    normed, group_layer.self_attention, layout
                )
                hidden = hidden + group_layer.dropout(attended)
            else:
                transposed = rearrange(hidden, "batch time d -> time batch d")
                normed = self._modulate_batch_major(
                    transposed, group_layer.layer_norm, group_scale, group_shift, mask
                )
                attention = group_layer.self_attention(
                    normed, mask=static["group_time_mask"]
                )
                transposed = transposed + group_layer.dropout(attention[0])
                hidden = rearrange(transposed, "time batch d -> batch time d")
            ffn_layer = block.layer[2]
            normed = self._modulate_time_major(
                hidden, ffn_layer.layer_norm, ffn_scale, ffn_shift, mask
            )
            hidden = hidden + ffn_layer.dropout(ffn_layer.mlp(normed))
        if self.final_modulation is None:
            final_scale, final_shift = (zero, zero)
        else:
            final_scale, final_shift = self.final_modulation(time_embedding).chunk(
                2, dim=-1
            )
        hidden = self._modulate_time_major(
            hidden, encoder.final_layer_norm, final_scale, final_shift, mask
        )
        return encoder.dropout(hidden)

    def future_hidden_slice(self, num_output_patches: int) -> slice:
        """Positions of the future patch tokens in the encoded sequence."""
        return slice(-num_output_patches, None)

    def encode_prepared(
        self, state: torch.Tensor, flow_time: torch.Tensor, static: dict[str, Any]
    ) -> torch.Tensor:
        """Encode one flow-time batch and return every hidden state."""
        batch_size = static["history_embeds"].shape[0]
        if flow_time.shape not in {(batch_size,), (batch_size, 1)}:
            raise ValueError(
                f"flow_time must have shape ({batch_size},) or ({batch_size}, 1), got {tuple(flow_time.shape)}"
            )
        time_embedding = self.flow_time_embedding(flow_time.reshape(-1, 1))
        gate = self.gate(time_embedding)
        if self.flow_entry_block:
            entry = static["entry_hidden"]
            history_patches = static["num_context_patches"]
            delta = (
                self.future_tokens(state, static, gate) - static["blank_future_embeds"]
            )
            hidden = torch.cat(
                [
                    entry[:, : history_patches + 1],
                    entry[:, history_patches + 1 :] + delta,
                ],
                dim=-2,
            )
            hidden = self.encode_sequence(
                hidden, static, time_embedding, start_block=self.flow_entry_block
            )
            return hidden
        future = self.future_tokens(state, static, gate)
        register = static["register_embeds"]
        pieces = [static["history_embeds"], register, future]
        hidden = torch.cat(pieces, dim=-2)
        hidden = self.encode_sequence(hidden, static, time_embedding)
        return hidden

    def _velocity_readout(self, forecast: torch.Tensor) -> torch.Tensor:
        if self.parallel_knots is not None:
            raise ValueError("This model requires parallel decoding")
        trunk = self.backbone.output_patch_embedding
        latent = trunk.act(trunk.hidden_layer(forecast))
        return trunk.dropout(
            self.clean_output_layer(latent)
        ) + self.clean_residual_layer(forecast)

    def forward_prepared(
        self,
        state: torch.Tensor,
        flow_time: torch.Tensor,
        static: dict[str, Any],
    ) -> torch.Tensor:
        """Return the velocity at one Euler integration point."""
        hidden = self.encode_prepared(state, flow_time, static)
        forecast = hidden[:, self.future_hidden_slice(static["num_output_patches"])]
        return self._velocity_readout(forecast).reshape(
            hidden.shape[0], static["future_length"]
        )

    def forward_fused(
        self,
        state: torch.Tensor,
        flow_time: torch.Tensor,
        static: dict[str, Any],
        *,
        start: int,
        length: int,
    ) -> torch.Tensor:
        """Mean velocity of one parallel block from the fused head."""
        hidden = self.encode_prepared(state, flow_time, static)
        return self.fused_head(
            hidden, static["num_output_patches"], start=start, length=length
        )

    def prepare_packed_draws(
        self, static: dict[str, Any], num_draws: int
    ) -> dict[str, Any]:
        """Pack ``num_draws`` flow states behind one shared history prefix."""
        if num_draws < 1:
            raise ValueError("num_draws must be positive")
        self._require_packable()
        history_patches = static["num_context_patches"]
        appended = 1 + static["num_output_patches"]
        history_embeds = static["history_embeds"]
        batch_size = history_embeds.shape[0]
        device = history_embeds.device
        floating_dtype = history_embeds.dtype
        packed_mask = torch.cat(
            [
                static["attention_mask"][:, :history_patches],
                torch.ones(
                    batch_size,
                    num_draws * appended,
                    dtype=static["attention_mask"].dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )
        sequence_length = history_patches + num_draws * appended
        position_ids = torch.cat(
            [
                torch.arange(history_patches, dtype=torch.long, device=device),
                (
                    torch.arange(appended, dtype=torch.long, device=device)
                    + history_patches
                ).repeat(num_draws),
            ]
        ).unsqueeze(0)
        draw_ids = torch.full((sequence_length,), -1, dtype=torch.long, device=device)
        draw_ids[history_patches:] = torch.arange(
            num_draws, device=device
        ).repeat_interleave(appended)
        group_ids = static["group_ids"]
        layout = self._group_layout(group_ids, packed_mask)
        draw_mask = packed_mask[:, history_patches:]
        draw_layout = self._group_layout(group_ids, draw_mask)
        return {
            **static,
            "attention_mask": packed_mask,
            "extended_attention_mask": self._packed_time_attention_mask(
                packed_mask, draw_ids, floating_dtype
            ),
            "group_time_mask": None
            if layout.get("uniform")
            else Chronos2Encoder._construct_and_invert_group_time_mask(
                group_ids, packed_mask, floating_dtype
            ),
            "position_ids": position_ids,
            "sequence_length": sequence_length,
            "group_layout": layout,
            "draw_group_layout": draw_layout,
            "draw_group_time_mask": None
            if draw_layout.get("uniform")
            else Chronos2Encoder._construct_and_invert_group_time_mask(
                group_ids, draw_mask, floating_dtype
            ),
            "num_draws": num_draws,
            "appended_per_draw": appended,
        }

    def _require_packable(self) -> None:
        """Reject designs whose history tokens could depend on the draws."""

    @staticmethod
    def _packed_time_attention_mask(
        attention_mask: torch.Tensor,
        draw_ids: torch.Tensor,
        floating_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Additive time-attention mask for a packed multi-draw sequence."""
        is_history = draw_ids < 0
        allowed = is_history[:, None] & is_history[None, :] | (~is_history)[:, None] & (
            is_history[None, :] | (draw_ids[:, None] == draw_ids[None, :])
        )
        key_valid = allowed[None] & (attention_mask > 0)[:, None, :]
        empty = ~key_valid.any(dim=-1, keepdim=True)
        key_valid = key_valid | empty & is_history[None, None, :]
        mask = torch.where(
            key_valid,
            torch.zeros((), dtype=floating_dtype, device=draw_ids.device),
            torch.full(
                (),
                torch.finfo(floating_dtype).min,
                dtype=floating_dtype,
                device=draw_ids.device,
            ),
        )
        return mask.unsqueeze(1)

    def encode_history(self, static: dict[str, Any]) -> dict[str, Any]:
        """Encode the history prefix once for reuse across solver evaluations."""
        self._require_packable()
        encoder = self.backbone.encoder
        history_patches = static["num_context_patches"]
        group_ids = static["group_ids"]
        history_mask = static["attention_mask"][:, :history_patches]
        floating_dtype = static["history_embeds"].dtype
        time_mask = static["extended_attention_mask"][
            :, :, :history_patches, :history_patches
        ]
        position_ids = static["position_ids"][:, :history_patches]
        layout = self._group_layout(group_ids, history_mask)
        group_time_mask = (
            None
            if layout.get("uniform")
            else Chronos2Encoder._construct_and_invert_group_time_mask(
                group_ids, history_mask, floating_dtype
            )
        )
        if self.flow_entry_block:
            hidden = static["entry_hidden"][:, :history_patches]
        else:
            hidden = encoder.dropout(static["history_embeds"])
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for index, block in enumerate(encoder.block):
            if index < self.flow_entry_block:
                continue
            time_layer = block.layer[0]
            attended, key, value = self._time_attention(
                time_layer.self_attention,
                time_layer.layer_norm(hidden),
                position_ids,
                time_mask,
            )
            keys.append(key)
            values.append(value)
            hidden = hidden + time_layer.dropout(attended)
            group_layer = block.layer[1]
            if layout.get("uniform"):
                normed = group_layer.layer_norm(hidden)
                attended = self._grouped_attention(
                    normed, group_layer.self_attention, layout
                )
                hidden = hidden + group_layer.dropout(attended)
            else:
                transposed = rearrange(hidden, "batch time d -> time batch d")
                normed = group_layer.layer_norm(transposed)
                attention = group_layer.self_attention(normed, mask=group_time_mask)
                transposed = transposed + group_layer.dropout(attention[0])
                hidden = rearrange(transposed, "time batch d -> batch time d")
            ffn_layer = block.layer[2]
            normed = ffn_layer.layer_norm(hidden)
            hidden = hidden + ffn_layer.dropout(ffn_layer.mlp(normed))
        return {
            "keys": keys,
            "values": values,
            "num_context_patches": history_patches,
            "batch_size": int(hidden.shape[0]),
        }

    def encode_packed_states(
        self,
        states: torch.Tensor,
        flow_times: torch.Tensor,
        packed: dict[str, Any],
        *,
        history_cache: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Encode every draw of one batch behind a shared history prefix."""
        num_draws = packed["num_draws"]
        batch_size = packed["history_embeds"].shape[0]
        future_length = packed["future_length"]
        if states.shape != (num_draws, batch_size, future_length):
            raise ValueError(
                f"states must have shape {(num_draws, batch_size, future_length)}, got {tuple(states.shape)}"
            )
        if flow_times.shape != (num_draws, batch_size):
            raise ValueError(
                f"flow_times must have shape {(num_draws, batch_size)}, got {tuple(flow_times.shape)}"
            )
        time_embeddings = self.flow_time_embedding(flow_times.reshape(-1, 1)).reshape(
            num_draws, batch_size, -1
        )
        history_patches = packed["num_context_patches"]
        if history_cache is not None and (
            history_cache["num_context_patches"] != history_patches
            or history_cache["batch_size"] != batch_size
        ):
            raise ValueError(
                "history_cache was encoded for a different window geometry"
            )
        if self.flow_entry_block:
            entry = packed["entry_hidden"]
            pieces = [entry[:, :history_patches]]
            for index in range(num_draws):
                pieces.append(entry[:, history_patches : history_patches + 1])
                delta = (
                    self.future_tokens(
                        states[index], packed, self.gate(time_embeddings[index])
                    )
                    - packed["blank_future_embeds"]
                )
                pieces.append(entry[:, history_patches + 1 :] + delta)
        else:
            pieces = [packed["history_embeds"]]
            for index in range(num_draws):
                register = packed["register_embeds"]
                pieces.append(register)
                pieces.append(
                    self.future_tokens(
                        states[index], packed, self.gate(time_embeddings[index])
                    )
                )
        if history_cache is not None:
            pieces = pieces[1:]
        hidden = torch.cat(pieces, dim=-2)
        hidden = self._encode_packed(
            hidden, packed, time_embeddings, history_cache=history_cache
        )
        return hidden

    def forward_packed(
        self,
        states: torch.Tensor,
        flow_times: torch.Tensor,
        packed: dict[str, Any],
        *,
        history_cache: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return packed velocities with shape [draws, rows, horizon]."""
        cached = bool(history_cache)
        hidden = self.encode_packed_states(
            states, flow_times, packed, history_cache=history_cache if cached else None
        )
        history_patches = 0 if cached else packed["num_context_patches"]
        forecast = hidden[:, history_patches:].reshape(
            hidden.shape[0], packed["num_draws"], packed["appended_per_draw"], hidden.shape[-1]
        )[:, :, 1:]
        return self._velocity_readout(forecast).reshape(
            hidden.shape[0], packed["num_draws"], packed["future_length"]
        ).permute(1, 0, 2)

    def forward_fused_packed(
        self,
        states: torch.Tensor,
        flow_times: torch.Tensor,
        packed: dict[str, Any],
        *,
        start: int,
        length: int,
        history_cache: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Fused block mean velocity of every draw,
        ``(num_draws, rows, future_length)``."""
        cached = bool(history_cache)
        hidden = self.encode_packed_states(
            states, flow_times, packed, history_cache=history_cache if cached else None
        )
        return self._packed_fused_head(
            hidden,
            packed,
            start=start,
            length=length,
            history_patches=0 if cached else None,
        )

    def _encode_packed(
        self,
        hidden: torch.Tensor,
        packed: dict[str, Any],
        time_embeddings: torch.Tensor,
        *,
        history_cache: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run the native encoder over a packed multi-draw sequence."""
        encoder = self.backbone.encoder
        appended = packed["appended_per_draw"]
        history_patches = packed["num_context_patches"]
        batch_size = hidden.shape[0]
        if history_cache is None:
            position_ids = packed["position_ids"]
            time_mask = packed["extended_attention_mask"]
            layout = packed["group_layout"]
            group_time_mask = packed["group_time_mask"]
        else:
            position_ids = packed["position_ids"][:, history_patches:]
            time_mask = packed["extended_attention_mask"][:, :, history_patches:]
            layout = packed["draw_group_layout"]
            group_time_mask = packed["draw_group_time_mask"]
            history_patches = 0

        def tokens(parameters: torch.Tensor) -> torch.Tensor:
            per_draw = parameters.permute(1, 0, 2).repeat_interleave(appended, dim=1)
            zeros = per_draw.new_zeros(batch_size, history_patches, per_draw.shape[-1])
            return torch.cat([zeros, per_draw], dim=1)

        def modulate(
            values: torch.Tensor,
            norm: nn.Module,
            scale: torch.Tensor | None,
            shift: torch.Tensor | None,
        ) -> torch.Tensor:
            normalized = norm(values)
            if scale is None:
                return normalized
            return normalized * (1.0 + scale.to(normalized.dtype)) + shift.to(
                normalized.dtype
            )

        if self.flow_entry_block == 0:
            hidden = encoder.dropout(hidden)
        shared_modulation = None
        if self.adaln_shared is not None:
            shared_modulation = self.adaln_shared(time_embeddings)

        def block_body(
            block: nn.Module,
            hidden: torch.Tensor,
            time_scale: torch.Tensor | None,
            time_shift: torch.Tensor | None,
            group_scale: torch.Tensor | None,
            group_shift: torch.Tensor | None,
            ffn_scale: torch.Tensor | None,
            ffn_shift: torch.Tensor | None,
            prefix: tuple[torch.Tensor, torch.Tensor] | None,
        ) -> torch.Tensor:
            time_layer = block.layer[0]
            normed = modulate(hidden, time_layer.layer_norm, time_scale, time_shift)
            if prefix is None:
                attention = time_layer.self_attention(
                    normed, position_ids=position_ids, mask=time_mask
                )[0]
            else:
                attention, _, _ = self._time_attention(
                    time_layer.self_attention, normed, position_ids, time_mask, prefix
                )
            hidden = hidden + time_layer.dropout(attention)
            group_layer = block.layer[1]
            if layout.get("uniform"):
                normed = modulate(
                    hidden, group_layer.layer_norm, group_scale, group_shift
                )
                attended = self._grouped_attention(
                    normed, group_layer.self_attention, layout
                )
                hidden = hidden + group_layer.dropout(attended)
            else:
                transposed = rearrange(hidden, "batch time d -> time batch d")
                normed = modulate(
                    transposed,
                    group_layer.layer_norm,
                    group_scale.permute(1, 0, 2) if group_scale is not None else None,
                    group_shift.permute(1, 0, 2) if group_shift is not None else None,
                )
                attention = group_layer.self_attention(normed, mask=group_time_mask)
                transposed = transposed + group_layer.dropout(attention[0])
                hidden = rearrange(transposed, "time batch d -> batch time d")
            ffn_layer = block.layer[2]
            normed = modulate(hidden, ffn_layer.layer_norm, ffn_scale, ffn_shift)
            return hidden + ffn_layer.dropout(ffn_layer.mlp(normed))

        for index, block in enumerate(encoder.block):
            if index < self.flow_entry_block:
                continue
            prefix = None
            if history_cache is not None:
                cached = index - self.flow_entry_block
                prefix = (
                    history_cache["keys"][cached],
                    history_cache["values"][cached],
                )
            if shared_modulation is not None:
                parameters = (
                    shared_modulation + self.adaln_block_offsets[index]
                ).chunk(6, dim=-1)
            else:
                parameters = (None,) * 6
            token_parameters = tuple(
                (
                    tokens(parameter) if parameter is not None else None
                    for parameter in parameters
                )
            )
            if self.training and getattr(self, "gradient_checkpointing", False):
                from torch.utils.checkpoint import checkpoint

                hidden = checkpoint(
                    block_body, block, hidden, *token_parameters, prefix,
                    use_reentrant=False,
                )
            else:
                hidden = block_body(block, hidden, *token_parameters, prefix)
        if self.final_modulation is None:
            final_scale = final_shift = None
        else:
            final_scale, final_shift = (
                tokens(parameter)
                for parameter in self.final_modulation(time_embeddings).chunk(2, dim=-1)
            )
        hidden = modulate(hidden, encoder.final_layer_norm, final_scale, final_shift)
        return encoder.dropout(hidden)

    @property
    def parallel_grid_size(self) -> int | None:
        if self.parallel_knots is None:
            return None
        return len(self.parallel_knots) - 1

    def _require_parallel(self) -> int:
        if self.parallel_knots is None:
            raise ValueError("This model has no parallel decoding heads")
        return len(self.parallel_knots) - 1

    def parallel_block_weights(
        self, start: int, length: int
    ) -> tuple[torch.Tensor, float, float]:
        """Step weights ``Delta_k`` of one block and its start and end time."""
        grid_size = self._require_parallel()
        start = int(start)
        length = int(length)
        if length < 1 or start < 0 or start + length > grid_size:
            raise ValueError(
                f"block [{start}, {start + length}) does not fit the {grid_size}-interval grid"
            )
        knots = self.parallel_knots
        widths = torch.tensor(
            [knots[index + 1] - knots[index] for index in range(start, start + length)],
            dtype=torch.float32,
        )
        span = knots[start + length] - knots[start]
        return (widths / span, knots[start], knots[start + length])

    def fused_readout(self, start: int, length: int) -> dict[str, Any]:
        """The single affine head that reproduces one block step exactly."""
        weights, start_time, end_time = self.parallel_block_weights(start, length)
        grid_size = self._require_parallel()
        output = self.parallel_output_layer
        weights = weights.to(device=output.weight.device, dtype=output.weight.dtype)
        block = slice(start, start + length)
        output_weight = output.weight.reshape(grid_size, self.patch_length, -1)[block]
        output_bias = output.bias.reshape(grid_size, self.patch_length)[block]
        fused = {
            "output_weight": torch.einsum("k,kpi->pi", weights, output_weight),
            "output_bias": torch.einsum("k,kp->p", weights, output_bias),
            "start_time": start_time,
            "end_time": end_time,
        }
        residual = self.parallel_residual_layer
        if residual is not None:
            residual_weight = residual.weight.reshape(grid_size, self.patch_length, -1)[
                block
            ]
            residual_bias = residual.bias.reshape(grid_size, self.patch_length)[block]
            fused["residual_weight"] = torch.einsum(
                "k,kpi->pi", weights, residual_weight
            )
            fused["residual_bias"] = torch.einsum("k,kp->p", weights, residual_bias)
        else:
            fused["residual_weight"] = self.clean_residual_layer.weight
            fused["residual_bias"] = self.clean_residual_layer.bias
        return fused

    def _apply_fused(
        self, forecast: torch.Tensor, fused: dict[str, Any]
    ) -> torch.Tensor:
        trunk = self.backbone.output_patch_embedding
        latent = trunk.act(trunk.hidden_layer(forecast))
        return trunk.dropout(
            F.linear(latent, fused["output_weight"], fused["output_bias"])
        ) + F.linear(forecast, fused["residual_weight"], fused["residual_bias"])

    def fused_head(
        self, hidden: torch.Tensor, num_output_patches: int, *, start: int, length: int
    ) -> torch.Tensor:
        """Block mean velocity ``(rows, future_length)`` from the fused head."""
        forecast = hidden[:, self.future_hidden_slice(num_output_patches)]
        prediction = self._apply_fused(forecast, self.fused_readout(start, length))
        return prediction.reshape(
            hidden.shape[0], num_output_patches * self.patch_length
        )

    def _packed_fused_head(
        self,
        hidden: torch.Tensor,
        packed: dict[str, Any],
        *,
        start: int,
        length: int,
        history_patches: int | None = None,
    ) -> torch.Tensor:
        num_draws = packed["num_draws"]
        appended = packed["appended_per_draw"]
        if history_patches is None:
            history_patches = packed["num_context_patches"]
        batch_size = hidden.shape[0]
        forecast = hidden[:, history_patches:].reshape(
            batch_size, num_draws, appended, hidden.shape[-1]
        )[:, :, 1:]
        prediction = self._apply_fused(forecast, self.fused_readout(start, length))
        return prediction.reshape(
            batch_size, num_draws, packed["future_length"]
        ).permute(1, 0, 2)
