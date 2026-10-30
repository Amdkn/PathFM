"""Load self-contained PathFM safetensors checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.layers import ResidualBlock
from chronos.chronos2.model import Chronos2Model
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .model import PathFM


def build_model(config: dict) -> PathFM:
    if config.get("format_version") != 1 or config.get("model_type") != "pathfm":
        raise ValueError("Expected a PathFM format-version-1 checkpoint")
    backbone_config = Chronos2CoreConfig(**config["backbone"])
    backbone_config._attn_implementation = "sdpa"
    backbone = Chronos2Model(backbone_config)
    embedding_width = int(config["input_embedding_hidden_dim"])
    if embedding_width != backbone_config.d_ff:
        backbone.input_patch_embedding = ResidualBlock(
            in_dim=backbone.chronos_config.input_patch_size * 3,
            h_dim=embedding_width,
            out_dim=backbone_config.d_model,
            act_fn_name=backbone_config.dense_act_fn,
            dropout_p=backbone_config.dropout_rate,
        )
    return PathFM(backbone, **config["architecture"]).eval()


def load_model(
    model_id: str | Path,
    *,
    device: str | torch.device = "cpu",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[PathFM, dict]:
    local = Path(model_id).expanduser()
    if local.is_dir():
        config_path = local / "config.json"
        weights_path = local / "model.safetensors"
    else:
        options = dict(
            repo_id=str(model_id),
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        config_path = Path(hf_hub_download(filename="config.json", **options))
        # Pin the weights to the same resolved commit as the configuration.
        options["revision"] = config_path.parent.name
        weights_path = Path(hf_hub_download(filename="model.safetensors", **options))
    config = json.loads(config_path.read_text())
    model = build_model(config)
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config
