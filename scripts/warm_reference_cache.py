#!/usr/bin/env python3
"""Warm registered standard references from a private, short-lived URL manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.config import load_api_settings
from api.reference_cache import prepare_reference_asset
from poc.alignment_3d_viewer.export_assets import _load_render_asset
from sam_3d_body.technique_alignment import load_skeleton_sequence_npz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    root = args.cache_root or Path(load_api_settings().reference_cache_root)
    entries = json.loads(args.manifest.read_text())
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        parser.error("manifest must be an array of reference objects")
    failed = 0
    for entry in entries:
        try:
            version = entry["version"]
            if not isinstance(version, str) or not version.strip():
                raise ValueError("missing version")
            for kind, key, validate in (
                ("skeleton", "skeletonUrl", load_skeleton_sequence_npz),
                ("render", "renderUrl", _load_render_asset),
            ):
                with prepare_reference_asset(
                    entry[key], cache_root=root, version=version,
                    kind=kind, validate=validate,
                ):
                    pass
            print(json.dumps({"id": entry.get("id"), "status": "ready"}), flush=True)
        except Exception as exc:
            failed += 1
            # Never print upstream messages, signed URLs, or the input manifest.
            print(json.dumps({"id": entry.get("id"), "status": "failed", "errorType": type(exc).__name__}), flush=True)
    print(json.dumps({"ready": len(entries) - failed, "failed": failed}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
