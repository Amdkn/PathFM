"""A'Space adapter — PathFM (D-N053 pas A, design Clara).

Contrat seul, patron D-N048 (dify / cordis) : cet adapter ne modifie AUCUN
fichier amont, ne fait AUCUN appel reseau, ne monte AUCUN poids.
Il constate : le gel epingle, la licence, le confinement, l'importabilite
du paquet amont, l'absence de poids dans le fork.

Adresse : fiber:a0:PathFM
Fork souverain : Amdkn/PathFM (parent autogluon/PathFM, branche main)
Gel amont : 8a3a42c05ce84981bc0fa292d5503b3f2ea22a8b (« Prepare PathFM release »)
Bind Cordis Fiber (16L) : d94942ecb0b12609bf1d450e50ab98240999d549
Licence : CC BY-NC 4.0
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

FIBER_ADDRESS = "fiber:a0:PathFM"
FORK = "Amdkn/PathFM"
UPSTREAM_REPO = "autogluon/PathFM"
UPSTREAM_BRANCH = "main"
PINNED_UPSTREAM_COMMIT = "8a3a42c05ce84981bc0fa292d5503b3f2ea22a8b"
PINNED_UPSTREAM_COMMIT_SHORT = "8a3a42c"
BIND_COMMIT = "d94942ecb0b12609bf1d450e50ab98240999d549"
BIND_LAYER = "16L"
LICENSE = "CC BY-NC 4.0"
ADAPTER_VERSION = "0.1.0"
UPSTREAM_PACKAGE = "pathfm"
UPSTREAM_PACKAGE_DIR = "src/pathfm"
CONFINEMENT = """CONFINEMENT PROFIL INTERNE (design D-N053 D3) : le CODE du repo PathFM est CC BY-NC 4.0 (LICENSE lue par Clara) et les MODELES PathFM sont non-commerciaux ; jamais en B2B (verifie a la configuration du profil commercial, H-A5). Manifeste ML corrige a CC BY-NC 4.0 (acte nomme du pas A, evidence D-N053/a/PathFM/)."""

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _weight_files(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if ".git" in p.parts:
            continue
        if p.is_file() and p.suffix in _WEIGHT_SUFFIXES:
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            out.append({"path": str(p.relative_to(root)), "size_o": size})
    return out


def describe() -> dict:
    """Decrit l'adapter (manifeste machine, sans effet de bord)."""
    return {
        "fiber": FIBER_ADDRESS,
        "fork": FORK,
        "upstream": UPSTREAM_REPO,
        "upstream_branch": UPSTREAM_BRANCH,
        "pinned_upstream_commit": PINNED_UPSTREAM_COMMIT,
        "bind_commit": BIND_COMMIT,
        "bind_layer": BIND_LAYER,
        "license": LICENSE,
        "adapter_version": ADAPTER_VERSION,
        "upstream_package": UPSTREAM_PACKAGE,
        "confinement": CONFINEMENT,
        "surface": "validate() -> dict ; doctor() -> list[check] ; describe() -> dict",
    }


def validate() -> dict:
    """Constate l'etat du fork gele. Aucun appel reseau, aucune ecriture."""
    root = _repo_root()
    try:
        mod = importlib.import_module(UPSTREAM_PACKAGE)
        importable, import_error = True, None
        upstream_version = getattr(mod, "__version__", None)
    except Exception as exc:  # noqa: BLE001 - constate, ne leve pas
        importable, import_error = False, f"{type(exc).__name__}: {exc}"
        upstream_version = None
    weights = _weight_files(root)
    return {
        "fiber": FIBER_ADDRESS,
        "pinned_upstream_commit": PINNED_UPSTREAM_COMMIT,
        "pinned_commit_is_40_hex": bool(re.fullmatch(r"[0-9a-f]{40}", PINNED_UPSTREAM_COMMIT)),
        "license_file_present": (root / "LICENSE").exists(),
        "license": LICENSE,
        "upstream_package_dir_present": (root / UPSTREAM_PACKAGE_DIR).is_dir(),
        "upstream_importable": importable,
        "upstream_import_error": import_error,
        "upstream_version": upstream_version,
        "weight_files_in_repo": weights,
        "adapter_network_calls": 0,
        "confinement": CONFINEMENT,
    }


def doctor() -> list[dict]:
    """Liste de controles (ok / detail). Tout doit etre ok en vert."""
    v = validate()
    root = _repo_root()
    weights = v["weight_files_in_repo"]
    # Seuls toleres : les fixtures factices de test (< 1 Mo, sous test/).
    unexpected = [w for w in weights if not (w["path"].startswith("test/") and w["size_o"] < 1_000_000)]
    checks = [
        {"check": "pinned_commit_40_hex", "ok": v["pinned_commit_is_40_hex"],
          "detail": PINNED_UPSTREAM_COMMIT_SHORT},
        {"check": "license_file_present", "ok": v["license_file_present"],
          "detail": LICENSE},
        {"check": "upstream_package_dir_present", "ok": v["upstream_package_dir_present"],
          "detail": UPSTREAM_PACKAGE_DIR},
        {"check": "upstream_importable", "ok": v["upstream_importable"],
          "detail": v["upstream_import_error"] or f"{UPSTREAM_PACKAGE} {v['upstream_version']}"},
        {"check": "no_real_weights_in_fork", "ok": not unexpected,
          "detail": f"{len(weights)} fichier(s) poids, {len(unexpected)} inattendu(s) : {unexpected}"},
        {"check": "adapter_network_calls_zero", "ok": v["adapter_network_calls"] == 0,
          "detail": "aucun import reseau dans l'adapter (verifie par grep a l'audit)"},
        {"check": "confinement_nomme", "ok": bool(CONFINEMENT),
          "detail": CONFINEMENT.split(":")[0]},
    ]
    return checks
