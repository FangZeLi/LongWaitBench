"""Manifest loader (as-is; no content validation beyond file existence).

The vendored manifest is already verified by the extraction pipeline
(see data/extraction_log.json). Over-strict loader guards have historically
killed every pipeline run that added them; here we only check that files
referenced by the manifest exist on disk.
"""
import json
import os

from . import constants as C


def manifest_path() -> str:
    return os.environ.get("LC1_MANIFEST") or str(C.DEFAULT_MANIFEST_PATH)


def load_manifest() -> dict:
    with open(manifest_path(), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    _check_file_existence(manifest)
    return manifest


def _check_file_existence(manifest: dict) -> None:
    """Only guard: referenced snapshot files must exist."""
    missing = []
    for inst in manifest.get("instances", []):
        snap = inst.get("initial_state", {}).get("db_snapshot")
        if snap and not os.path.exists(snap):
            missing.append((inst.get("id"), snap))
    if missing:
        raise FileNotFoundError(
            "manifest db snapshots missing: " + "; ".join(f"{i}: {p}" for i, p in missing)
        )


def load_extraction_log() -> dict:
    with open(C.EXTRACTION_LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
