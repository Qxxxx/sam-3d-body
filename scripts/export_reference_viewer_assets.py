#!/usr/bin/env python3
"""Export the reference-side viewer buffers that never change per task.

The 3D comparison viewer loads four reference buffers that are identical for
every task using the same reference asset version:

* ``reference.camera.positions`` - the full smoothed reference vertex sequence
  (the camera overlay indexes it per aligned frame, so no DTW input is needed)
* ``reference.camera.translation`` / ``reference.camera.intrinsics`` - the
  reference camera parameters
* ``mesh.indices`` - the fixed mesh topology

The remaining reference buffers (``raw.positions`` and ``basis``) depend on the
user's alignment path and stay per task. Buffers produced here are uploaded by
``cf-backend/scripts/technique/warm-reference-viewer-assets.mjs``, which also
records their URLs on the reference row, so a task references them instead of
exporting and uploading another ~26 MB copy.

The writer helpers come from the shipped exporter so the bytes match what the
per-task path produced before this cache existed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.reference_cache import prepare_reference_asset
from poc.alignment_3d_viewer.export_assets import (
    _load_render_asset,
    _write_float_buffer,
    _write_index_buffer,
    temporal_smooth_frames,
)

# Manifest name -> exported file name, mirroring TECHNIQUE_VIEWER_SHARED_FILES.
SHARED_BUFFERS = {
    "referenceCameraPositions": "reference.camera.positions.f32.bin",
    "referenceCameraTranslation": "reference.camera.translation.f32.bin",
    "referenceCameraIntrinsics": "reference.camera.intrinsics.f32.bin",
    "indices": "mesh.indices.u32.bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-npz", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--reference-version")
    parser.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print the generated file list as JSON instead of a summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the versioned local cache so re-running the warm step never downloads
    # the same render asset twice.
    if args.render_npz.startswith(("http://", "https://")):
        if not args.cache_root or not args.reference_version:
            raise SystemExit(
                "--cache-root and --reference-version are required for URLs"
            )
        with prepare_reference_asset(
            args.render_npz,
            cache_root=args.cache_root,
            version=args.reference_version,
            kind="render",
            validate=_load_render_asset,
        ) as resolved:
            asset = _load_render_asset(resolved)
    else:
        asset = _load_render_asset(args.render_npz)

    reference_vertices = temporal_smooth_frames(asset["vertices_3d"])
    _write_float_buffer(
        out_dir / SHARED_BUFFERS["referenceCameraPositions"], reference_vertices
    )
    _write_float_buffer(
        out_dir / SHARED_BUFFERS["referenceCameraTranslation"], asset["cam_t"]
    )
    _write_float_buffer(
        out_dir / SHARED_BUFFERS["referenceCameraIntrinsics"], asset["cam_intrinsics"]
    )
    _write_index_buffer(out_dir / SHARED_BUFFERS["indices"], asset["faces"])

    files = [
        {
            "manifestKey": key,
            "fileName": name,
            "sizeBytes": (out_dir / name).stat().st_size,
        }
        for key, name in SHARED_BUFFERS.items()
    ]
    if args.print_metadata:
        print(
            json.dumps(
                {"files": files, "frameCount": int(reference_vertices.shape[0])}
            )
        )
    else:
        for entry in files:
            print(f"{entry['fileName']}: {entry['sizeBytes']} bytes")
        print(f"frames: {int(reference_vertices.shape[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
