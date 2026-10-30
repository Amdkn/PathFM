"""Reproduce the released PANDA sMAPE scores at 64, 128, and 256 steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pathfm import PathFMPipeline
from pathfm.sampling import denormalized_samples, draw_source, flow_sample

from .common import write_json
from .panda_data import CHANNELS, CONTEXT, WINDOWS, sha256, validate_tasks

HORIZONS = (64, 128, 256)
NUM_SHARDS = 16
NUM_SAMPLES = 16
BATCH_SIZE = 32
MAX_ROWS = 512


def window_smape(
    point: np.ndarray,
    truth: np.ndarray,
    context: np.ndarray,
) -> dict[str, np.ndarray]:
    """PANDA's context-standardized, 0–100 sMAPE per window."""
    if point.shape != truth.shape or point.shape[:2] != context.shape[:2]:
        raise ValueError("Forecasts, truth, and context must align")
    location = np.nanmean(context, axis=-1, keepdims=True)
    scale = np.nanstd(context, axis=-1, keepdims=True)
    scale = np.where(scale < 1e-10, 1e-10, scale)
    predicted, observed = (point - location) / scale, (truth - location) / scale
    scores = {}
    for horizon in HORIZONS:
        x, y = predicted[..., :horizon], observed[..., :horizon]
        scores[f"h{horizon}_smape"] = (
            100.0 * np.abs(x - y) / (np.abs(x) + np.abs(y) + 1e-10)
        ).mean(axis=(1, 2))
    return scores


class PandaPredictor:
    """The release's 16-path protocol with the checkpoint's default decoder."""

    def __init__(self, pipeline: PathFMPipeline, channels: int):
        self.model = pipeline.flow
        self.device = self.model.backbone.device
        self.amp = self.device.type == "cuda"
        self.steps = pipeline.ode_steps
        self.method = pipeline.sampler_method
        self.generator = torch.Generator(device=self.device).manual_seed(
            42 + 1000 * channels
        )
        if self.model.max_horizon != 256:
            raise ValueError("This protocol requires a 256-step PathFM-Chaotic checkpoint")

    @torch.inference_mode()
    def __call__(self, contexts: np.ndarray) -> np.ndarray:
        if contexts.ndim != 3 or contexts.shape[-1] != CONTEXT:
            raise ValueError("Expected [windows, channels, 512] contexts")
        channels = contexts.shape[1]
        location = contexts.mean(axis=-1, keepdims=True)
        scale = contexts.std(axis=-1, keepdims=True)
        scale = np.where(scale > 1e-10, scale, 1e-10)
        standardized = ((contexts - location) / scale).astype(np.float32)
        per_chunk = max(1, MAX_ROWS // (channels * NUM_SAMPLES))
        chunks = []
        counts = []
        for start in range(0, len(contexts), per_chunk):
            block = standardized[start : start + per_chunk]
            counts.append(len(block))
            rows = torch.from_numpy(
                np.ascontiguousarray(
                    block.reshape(len(block) * channels, CONTEXT),
                )
            ).to(self.device)
            groups = torch.arange(len(block), device=self.device).repeat_interleave(
                channels
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.amp,
            ):
                static = self.model.prepare_static(
                    rows,
                    num_output_patches=16,
                    group_ids=groups,
                )
            samples = flow_sample(
                self.model,
                static,
                torch.ones(len(rows), 256, device=self.device),
                num_samples=NUM_SAMPLES,
                ode_steps=self.steps,
                method=self.method,
                source_sampling="stratified",
                generator=self.generator,
                amp=self.amp,
            )
            ordered = (
                denormalized_samples(self.model, static, samples)
                .float()
                .sort(dim=1)
                .values
            )
            point = (ordered[:, 7] + ordered[:, 8]) * 0.5
            chunks.append(point.cpu().numpy().reshape(len(block), channels, 256))
        # The archived evaluator generated a second 256-step block per batch.
        # Advance those unused noise draws to reproduce the next batch's
        # first block exactly, without computing or feeding back that forecast.
        for count in counts:
            draw_source(
                count * channels,
                NUM_SAMPLES,
                256,
                sampling="stratified",
                device=self.device,
                generator=self.generator,
                scale=self.model.source_scale,
            )
        return np.concatenate(chunks) * scale + location


def aggregate(output: Path, *, max_groups: int | None) -> dict:
    """Mean windows, then trajectories, then equally weighted source groups."""
    pieces = []
    for channels in CHANNELS:
        for shard in range(NUM_SHARDS):
            path = output / f"shard{shard:02d}" / f"c{channels}.npz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing shard; finish inference first: {path}"
                )
            with np.load(path, allow_pickle=False) as data:
                frame = pd.DataFrame(
                    {
                        key: data[key]
                        for key in (
                            "row_subset",
                            "system_index",
                            "window_index",
                            "dataset",
                            "source_directory",
                            *(f"h{h}_smape" for h in HORIZONS),
                        )
                    }
                )
            frame["channels"] = channels
            pieces.append(frame)
    windows = pd.concat(pieces, ignore_index=True).sort_values(
        ["channels", "row_subset"]
    )
    metrics = [f"h{h}_smape" for h in HORIZONS]
    # The original evaluator promotes its float32 window scores before pooling.
    windows[metrics] = windows[metrics].astype(np.float64)
    if windows.duplicated(["channels", "row_subset"]).any():
        raise ValueError("Duplicate forecast windows")
    if not np.isfinite(windows[metrics]).all().all():
        raise ValueError("Nonfinite PANDA scores")
    counts = windows.groupby("system_index").size()
    if not (counts == WINDOWS).all():
        raise ValueError("Every evaluated trajectory must contain all six windows")
    identity = ["dataset", "source_directory", "channels"]
    trajectories = windows.groupby([*identity, "system_index"], sort=True)[
        metrics
    ].mean()
    groups = trajectories.groupby(identity, sort=True)[metrics].mean().reset_index()
    complete = max_groups is None
    if complete and (len(windows), len(trajectories), len(groups)) != (
        56082,
        9347,
        480,
    ):
        raise ValueError("The full benchmark must cover 56,082 windows in 480 groups")
    groups.to_csv(output / "groups.csv", index=False)
    summary = {
        "complete": complete,
        "windows": len(windows),
        "trajectories": len(trajectories),
        "source_groups": len(groups),
        "aggregation": "mean across source groups",
        "metrics": {name: float(groups[name].mean()) for name in metrics},
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(args):
    validate_tasks(args.data_root)
    manifest_hash = sha256(args.data_root / "manifest.json")
    protocol_path = args.output / "protocol.json"
    if args.score_only:
        protocol = json.loads(protocol_path.read_text())
        if protocol["tasks_sha256"] != manifest_hash:
            raise ValueError("The results use a different task dataset")
        return aggregate(args.output, max_groups=protocol["max_groups"])
    if not args.model:
        raise ValueError("--model is required for inference")
    torch.manual_seed(42)
    np.random.seed(42)
    pipeline = PathFMPipeline.from_pretrained(args.model, device=args.device)
    config = pipeline.release_config
    identity = config.get("weights_sha256") or config.get("provenance", {}).get(
        "source_checkpoint_sha256"
    )
    if not identity:
        raise ValueError("The model configuration must identify its weights")
    protocol = {
        "name": "panda_short_horizon_v1",
        "model_weights": identity,
        "tasks_sha256": manifest_hash,
        "seed": 42,
        "steps": pipeline.ode_steps,
        "sampler_method": pipeline.sampler_method,
        "samples": NUM_SAMPLES,
        "source_sampling": "stratified",
        "batch_size": BATCH_SIZE,
        "max_rows": MAX_ROWS,
        "shards": NUM_SHARDS,
        "max_groups": args.max_groups,
        "save_forecasts": args.save_forecasts,
        "precision": "bfloat16"
        if pipeline.flow.backbone.device.type == "cuda"
        else "float32",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with protocol_path.open("x") as stream:
            json.dump(protocol, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Use a separate output directory for a different protocol")
    shards = range(NUM_SHARDS) if args.shard_index is None else [args.shard_index]
    for channels in CHANNELS:
        with np.load(args.data_root / f"c{channels}.npz", allow_pickle=False) as data:
            contexts, truth = data["context"], data["target"]
            metadata = {
                key: data[key]
                for key in (
                    "system_index",
                    "window_index",
                    "origin",
                    "dataset",
                    "source_directory",
                )
            }
        groups = {}
        for index, key in enumerate(
            zip(metadata["dataset"], metadata["source_directory"])
        ):
            groups.setdefault(key, []).append(index)
        group_indices = list(groups.values())
        if args.max_groups is not None:
            group_indices = group_indices[: args.max_groups]
        for shard in shards:
            path = args.output / f"shard{shard:02d}" / f"c{channels}.npz"
            if path.exists():
                continue
            predict = PandaPredictor(pipeline, channels)
            selected_rows, points = [], []
            for indices in group_indices:
                indices = np.asarray(indices, dtype=np.int64)
                indices = indices[indices % NUM_SHARDS == shard]
                for start in range(0, len(indices), BATCH_SIZE):
                    selected = indices[start : start + BATCH_SIZE]
                    point = predict(contexts[selected])
                    if not np.isfinite(point).all():
                        raise FloatingPointError("Nonfinite PANDA point forecasts")
                    selected_rows.append(selected)
                    points.append(point)
            selected = (
                np.concatenate(selected_rows)
                if selected_rows
                else np.empty(0, dtype=np.int64)
            )
            point = (
                np.concatenate(points)
                if points
                else np.empty((0, channels, 256), dtype=np.float32)
            )
            arrays = {
                "row_subset": selected,
                **{key: values[selected] for key, values in metadata.items()},
                **window_smape(point, truth[selected, :, :256], contexts[selected]),
            }
            if args.save_forecasts:
                arrays["point_forecast"] = point
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(path)
            print(
                f"shard {shard:02d}, c{channels}: {len(selected)} windows", flush=True
            )
    if args.shard_index is None:
        return aggregate(args.output, max_groups=args.max_groups)
    print(
        "Finish all 16 shards, then run the same command with --score-only.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="PathFM-Chaotic Hub ID or local weights folder")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--shard-index", type=int, choices=range(NUM_SHARDS))
    parser.add_argument(
        "--max-groups",
        type=int,
        help="Smoke check: first N complete groups per channel count",
    )
    parser.add_argument("--save-forecasts", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("--max-groups must be positive")
    run(args)


if __name__ == "__main__":
    main()
