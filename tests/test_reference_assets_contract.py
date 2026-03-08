from __future__ import annotations

import json
import re
from pathlib import Path

from sam_3d_body.reference_assets import RENDER_ASSET_SCHEMA_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT
    / "harness"
    / "contracts"
    / "fixtures"
    / "technique-reference-assets.v2.sample.json"
)


def _is_absolute_machine_path(value: str) -> bool:
    return value.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def test_shared_metadata_fixture_uses_expected_schema_and_relative_paths() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["schemaVersion"] == "technique_reference_assets.v2"
    assert fixture["renderAssetSchemaVersion"] == RENDER_ASSET_SCHEMA_VERSION
    assert fixture["assetCount"] == len(fixture["assets"])

    asset = fixture["assets"][0]
    assert not _is_absolute_machine_path(asset["sourceVideoPath"])
    assert not _is_absolute_machine_path(asset["skeletonPath"])
    assert not _is_absolute_machine_path(asset["renderAssetPath"])
