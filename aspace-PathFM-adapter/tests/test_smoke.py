"""Smoke tests de l'adapter PathFM (D-N053 pas A).

Roules avec le venv du pas : `python -m pytest aspace-PathFM-adapter/tests/test_smoke.py -v`.
Ne touchent pas l'amont, ne font aucun appel reseau.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aspace_pathfm_adapter as adapter  # noqa: E402


def test_constants():
    assert re.fullmatch(r"[0-9a-f]{40}", adapter.PINNED_UPSTREAM_COMMIT)
    assert adapter.FIBER_ADDRESS == "fiber:a0:PathFM"
    assert adapter.LICENSE == "CC BY-NC 4.0"
    assert adapter.FORK == "Amdkn/PathFM"
    assert adapter.UPSTREAM_REPO == "autogluon/PathFM"
    assert adapter.ADAPTER_VERSION == "0.1.0"


def test_validate_keys_and_files():
    v = adapter.validate()
    assert v["pinned_commit_is_40_hex"] is True
    assert v["license_file_present"] is True
    assert v["upstream_package_dir_present"] is True
    assert v["adapter_network_calls"] == 0
    assert v["confinement"]


def test_upstream_importable():
    v = adapter.validate()
    assert v["upstream_importable"] is True, v["upstream_import_error"]


def test_doctor_all_pass():
    checks = adapter.doctor()
    assert checks, "doctor() vide"
    failed = [c for c in checks if not c["ok"]]
    assert not failed, failed


def test_no_network_import_in_adapter():
    src = (Path(adapter.__file__).parent / "__init__.py").read_text()
    for banned in ("requests", "httpx", "urllib", "socket", "huggingface_hub", "aiohttp"):
        assert banned not in src, banned


def test_describe():
    d = adapter.describe()
    assert d["fiber"] == "fiber:a0:PathFM"
    assert d["pinned_upstream_commit"] == adapter.PINNED_UPSTREAM_COMMIT
    assert d["license"] == "CC BY-NC 4.0"
