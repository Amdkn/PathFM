# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
# Modified by TIME for flexible prediction lengths and YAML config support.

import math
import os
import warnings
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import datasets
import pyarrow.compute as pc
import yaml
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose

# --- Constants from Gift-Eval---
# Default prediction length by frequency (used when no config provided)
M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 12,
    "W": 8,
    "B": 21,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

# Local change: use the bundled protocol and omit unused split accessors.
DEFAULT_CONFIG_PATH = Path(__file__).parent / "protocols" / "time-datasets.yaml"


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15
        return 1


def load_dataset_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load dataset configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_dataset_settings(
    name: str,
    term: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get prediction_length, test_length, and val_length for a specific dataset and term.

    Returns dict with keys: 'prediction_length', 'test_length', 'val_length'
    """
    datasets_config = config.get("datasets", {})

    if name in datasets_config:
        dataset_config = datasets_config[name]
        term_config = dataset_config.get(term, {})
        return {
            "prediction_length": term_config.get("prediction_length"),
            "test_length": dataset_config.get("test_length"),
            "val_length": dataset_config.get("val_length"),
        }
    else:
        raise ValueError(f"Dataset '{name}' not found in configuration.")


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


class Dataset:
    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = False,
        prediction_length: Optional[int] = None,
        test_length: int = None,
        val_length: int = 0,
        storage_env_var: str = "TIME_DATASET",
        storage_path: Optional[str | Path] = None,
    ):
        """
        Initialize a TimeBench Dataset.

        Parameters
        ----------
        name : str
            Dataset name (path relative to storage, e.g., "bitbrains_rnd/5T")
        term : Term or str
            Forecast horizon term: "short", "medium", or "long"
        to_univariate : bool
            Convert multivariate to univariate
        prediction_length : int, optional
            Prediction length. If None, use default calculation based on freq and term.
        test_length : int
            Length of test set (in time steps). Required parameter.
            Windows is auto-calculated: ceil(test_length / prediction_length)
        val_length : int
            Length of validation set (in time steps). Required parameter.
            Val windows is auto-calculated: ceil(val_length / prediction_length)
        storage_env_var : str
            Environment variable name for dataset storage path (used if storage_path is None).
        storage_path : str or Path, optional
            Direct path to dataset storage. If provided, overrides storage_env_var.
        """
        # Use direct storage_path if provided, otherwise fall back to environment variable
        if storage_path is not None:
            resolved_storage_path = Path(storage_path)
        else:
            load_dotenv()
            env_path = os.getenv(storage_env_var)
            if not env_path:
                raise ValueError(
                    f"Environment variable '{storage_env_var}' not set. "
                    "Either set the environment variable or provide storage_path parameter."
                )
            resolved_storage_path = Path(env_path)

        dataset_path = resolved_storage_path / name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

        self.hf_dataset = datasets.load_from_disk(str(dataset_path)).with_format(
            "numpy"
        )

        self.term = Term(term) if isinstance(term, str) else term
        self.name = name
        self._custom_prediction_length = prediction_length
        self._test_length = test_length
        self._val_length = val_length

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

    @cached_property
    def prediction_length(self) -> int:
        """Get prediction length. Uses explicit value if provided, else default calculation."""
        if self._custom_prediction_length is not None:
            return self._custom_prediction_length

        # Default calculation based on frequency and term
        freq = norm_freq_str(to_offset(self.freq).name)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if len(target.shape) > 1 else 1

    def _validate_lengths(self):
        """Validate test_length and val_length against min_series_length."""
        min_len = self._min_series_length

        # Check test_length
        if self._test_length > min_len:
            raise ValueError(
                f"test_length ({self._test_length}) exceeds min_series_length ({min_len}). "
                f"Please reduce test_length."
            )
        if self._test_length > 0.5 * min_len:
            warnings.warn(
                f"test_length ({self._test_length}) exceeds 50% of min_series_length ({min_len}). "
                f"This may leave insufficient data for training.",
                UserWarning,
            )

        # Check val_length
        if self._val_length > min_len:
            raise ValueError(
                f"val_length ({self._val_length}) exceeds min_series_length ({min_len}). "
                f"Please reduce val_length."
            )
        if self._val_length > 0.5 * min_len:
            warnings.warn(
                f"val_length ({self._val_length}) exceeds 50% of min_series_length ({min_len}). "
                f"This may leave insufficient data for training.",
                UserWarning,
            )

    @cached_property
    def windows(self) -> int:
        """
        Get number of test windows.
        Auto-calculated: floor(test_length / prediction_length)
        """
        self._validate_lengths()
        w = math.floor(self._test_length / self.prediction_length)
        return max(1, w)

    @cached_property
    def _series_lengths(self):
        """Get array of all series lengths."""
        target_col = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            # Multivariate: get length of inner list
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(target_col, 0, 1))
            )
        else:
            lengths = pc.list_value_length(target_col)
        return lengths.to_numpy()

    @cached_property
    def _min_series_length(self) -> int:
        return int(min(self._series_lengths))

    @property
    def test_data(self) -> TestData:
        _, test_template = split(self.gluonts_dataset, offset=-self._test_length)
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data
