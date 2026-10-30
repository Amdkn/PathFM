"""Pinned PANDA test data and the original six-window sampling protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

CONTEXT = 512
FUTURE = 512
WINDOWS = 6
WINDOW_SEED = 8097
CHANNELS = (3, 4, 5, 6, 10)
SOURCES = {
    "base40": {
        "revision": "31dda63b5e5dcd6eb7f7ee196007a91aa1e82178",
        "sha256": "c64c6a5860e77f7eb863f62b6b0098bb4ecc6577284511c431be5c5a570c17b2",
    },
    "skew40": {
        "revision": "481d79f05aad91a8bc3455a14e48fbbcf90936d9",
        "sha256": "c4d6ea01fc5dac632f0a84ad90111e75ccaa22046ab776700abb58a722a20563",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_source(name: str, root: Path | str) -> Path:
    """Download one original test Parquet and verify its contents."""
    source = SOURCES[name]
    path = Path(
        hf_hub_download(
            repo_id=f"GilpinLab/{name}",
            repo_type="dataset",
            filename="test_zeroshot.parquet",
            revision=source["revision"],
            local_dir=Path(root) / "raw" / name,
        )
    )
    if sha256(path) != source["sha256"]:
        raise ValueError(f"{name}: test Parquet checksum mismatch")
    return path


def source_rows(name: str, path: Path) -> list[dict]:
    table = pq.read_table(
        path,
        columns=[
            "target._np_shape",
            "_source_directory",
            "_source_filename",
        ],
    )
    return [
        {
            "dataset": name,
            "dataset_row_index": index,
            "source_directory": str(directory),
            "source_filename": str(filename),
            "channels": int(shape[0]),
            "timepoints": int(shape[1]),
        }
        for index, (shape, directory, filename) in enumerate(
            zip(
                table["target._np_shape"].to_pylist(),
                table["_source_directory"].to_pylist(),
                table["_source_filename"].to_pylist(),
                strict=True,
            )
        )
    ]


def file_index(row: dict) -> int:
    return int(row["source_filename"].split("_", 1)[0])


def example_trajectory(
    root: Path | str,
    *,
    dataset: str = "base40",
    system: str = "Bouali",
    trajectory: int = 0,
) -> tuple[np.ndarray, dict]:
    """Load one public test trajectory as [channels, time]."""
    path = download_source(dataset, root)
    candidates = sorted(
        (
            row
            for row in source_rows(dataset, path)
            if row["source_directory"] == system
        ),
        key=file_index,
    )
    if not 0 <= trajectory < len(candidates):
        raise ValueError(f"{dataset}/{system} has {len(candidates)} trajectories")
    row = candidates[trajectory]
    parquet = pq.ParquetFile(path)
    offset = 0
    for index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(index).num_rows
        if offset <= row["dataset_row_index"] < offset + count:
            table = parquet.read_row_group(index, columns=["target"])
            values = table["target"][row["dataset_row_index"] - offset].as_py()
            return np.asarray(values, dtype=np.float32).reshape(
                row["channels"],
                row["timepoints"],
            ), row
        offset += count
    raise ValueError("Trajectory row is missing from its Parquet")


def validate_tasks(root: Path | str) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {
        "schema": "panda_paper_forecast_tasks",
        "status": "complete",
        "seed": 42,
        "window_seed": WINDOW_SEED,
        "systems": 9347,
        "windows": 56082,
        "context_length": CONTEXT,
        "prediction_length": FUTURE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"PANDA manifest {key} must be {value!r}")
    for channels in CHANNELS:
        path = root / f"c{channels}.npz"
        if sha256(path) != manifest["shards"][str(channels)]["sha256"]:
            raise ValueError(f"PANDA task checksum mismatch: {path.name}")
    return manifest


def prepare_dataset(root: Path | str) -> Path:
    """Export the 56,082 fixed windows, preserving the original row order."""
    root = Path(root)
    if (root / "manifest.json").is_file():
        validate_tasks(root)
        return root
    if list(root.glob("c*.npz")):
        raise FileExistsError(
            f"{root} contains task shards without a complete manifest"
        )
    sources = {name: download_source(name, root) for name in SOURCES}
    groups = {}
    for name, path in sources.items():
        for row in source_rows(name, path):
            groups.setdefault((name, row["source_directory"]), []).append(row)
    ordered = []
    for group in groups.values():
        generator = np.random.default_rng(WINDOW_SEED)
        for row in sorted(group, key=file_index):
            row["system_index"] = len(ordered)
            # Keep the original 512-step target when sampling origins, even
            # when only its 64/128/256-step prefixes are scored.
            row["origins"] = generator.integers(
                CONTEXT,
                row["timepoints"] - FUTURE + 1,
                size=WINDOWS,
            )
            ordered.append(row)
    if len(ordered) != 9347 or len(groups) != 480:
        raise ValueError("The test data must contain 9,347 trajectories in 480 groups")

    shards = {}
    for channels in CHANNELS:
        rows = [row for row in ordered if row["channels"] == channels]
        count = len(rows) * WINDOWS
        arrays = {
            "context": np.empty((count, channels, CONTEXT), dtype=np.float32),
            "target": np.empty((count, channels, FUTURE), dtype=np.float32),
            "system_index": np.repeat(
                [row["system_index"] for row in rows], WINDOWS
            ).astype(np.int64),
            "window_index": np.tile(np.arange(WINDOWS, dtype=np.int64), len(rows)),
            "dataset": np.repeat([row["dataset"] for row in rows], WINDOWS),
            "dataset_row_index": np.repeat(
                [row["dataset_row_index"] for row in rows], WINDOWS
            ).astype(np.int64),
            "source_directory": np.repeat(
                [row["source_directory"] for row in rows], WINDOWS
            ),
            "source_filename": np.repeat(
                [row["source_filename"] for row in rows], WINDOWS
            ),
            "origin": np.concatenate([row["origins"] for row in rows]),
        }
        for index, row in enumerate(rows):
            row["offset"] = index * WINDOWS
        shards[channels] = arrays
    lookup = {(row["dataset"], row["dataset_row_index"]): row for row in ordered}
    for name, path in sources.items():
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=32, columns=["target"]
        ):
            for flattened in batch.column(0).to_pylist():
                row = lookup[(name, row_index)]
                values = np.asarray(flattened, dtype=np.float32).reshape(
                    row["channels"],
                    row["timepoints"],
                )
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite test trajectory: {name}/{row_index}")
                arrays = shards[row["channels"]]
                for window, origin in enumerate(row["origins"]):
                    index = row["offset"] + window
                    arrays["context"][index] = values[:, origin - CONTEXT : origin]
                    arrays["target"][index] = values[:, origin : origin + FUTURE]
                row_index += 1

    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "complete",
        "schema": "panda_paper_forecast_tasks",
        "schema_version": 1,
        "seed": 42,
        "window_seed": WINDOW_SEED,
        "systems": len(ordered),
        "source_groups": len(groups),
        "windows": len(ordered) * WINDOWS,
        "windows_per_system": WINDOWS,
        "context_length": CONTEXT,
        "prediction_length": FUTURE,
        "channel_counts": dict(
            sorted(Counter(row["channels"] for row in ordered).items())
        ),
        "shards": {},
    }
    for channels, arrays in shards.items():
        path = root / f"c{channels}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        manifest["shards"][str(channels)] = {
            "sha256": sha256(path),
            "windows": len(arrays["context"]),
        }
        print(f"PANDA c{channels}: {len(arrays['context'])} windows", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root
