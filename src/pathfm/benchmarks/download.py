"""Download the exact dataset snapshot used by a release benchmark."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .common import PROTOCOLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("fev", "gift", "time", "panda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "panda":
        from .panda_data import prepare_dataset

        print(prepare_dataset(args.output).resolve())
        return
    config = json.loads((PROTOCOLS / "datasets.json").read_text())[args.suite]
    snapshot_download(
        **config,
        repo_type="dataset",
        local_dir=args.output,
        max_workers=4,
    )
    print(f"{args.suite}: {args.output.resolve()} at revision {config['revision']}")


if __name__ == "__main__":
    main()
