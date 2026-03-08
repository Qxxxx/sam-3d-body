# Copyright (c) Meta Platforms, Inc. and affiliates.
__version__ = "1.0.0"

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
    "VideoExtractionConfig",
    "extract_skeleton_sequence_from_video",
    "DEFAULT_METADATA_FILENAME",
    "RENDER_ASSET_SCHEMA_VERSION",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "ReferenceAssetBundle",
    "ReferenceVideoEntry",
    "build_reference_asset_bundle",
    "discover_reference_videos",
    "load_reference_manifest",
    "save_render_asset_npz",
    "save_reference_assets_metadata",
    "save_skeleton_sequence_npz",
    "build_reference_assets_metadata",
    "build_reference_assets",
]


def __getattr__(name: str):
    if name == "SAM3DBodyEstimator":
        from .sam_3d_body_estimator import SAM3DBodyEstimator

        return SAM3DBodyEstimator
    if name in {"load_sam_3d_body", "load_sam_3d_body_hf"}:
        from .build_models import load_sam_3d_body, load_sam_3d_body_hf

        return {
            "load_sam_3d_body": load_sam_3d_body,
            "load_sam_3d_body_hf": load_sam_3d_body_hf,
        }[name]
    if name in {"VideoExtractionConfig", "extract_skeleton_sequence_from_video"}:
        from .video_processor import VideoExtractionConfig, extract_skeleton_sequence_from_video

        return {
            "VideoExtractionConfig": VideoExtractionConfig,
            "extract_skeleton_sequence_from_video": extract_skeleton_sequence_from_video,
        }[name]
    if name in {
        "DEFAULT_METADATA_FILENAME",
        "RENDER_ASSET_SCHEMA_VERSION",
        "SUPPORTED_VIDEO_EXTENSIONS",
        "ReferenceAssetBundle",
        "ReferenceVideoEntry",
        "build_reference_asset_bundle",
        "discover_reference_videos",
        "load_reference_manifest",
        "save_render_asset_npz",
        "save_reference_assets_metadata",
        "save_skeleton_sequence_npz",
        "build_reference_assets_metadata",
        "build_reference_assets",
    }:
        from .reference_assets import (
            DEFAULT_METADATA_FILENAME,
            RENDER_ASSET_SCHEMA_VERSION,
            SUPPORTED_VIDEO_EXTENSIONS,
            ReferenceAssetBundle,
            ReferenceVideoEntry,
            build_reference_asset_bundle,
            build_reference_assets,
            build_reference_assets_metadata,
            discover_reference_videos,
            load_reference_manifest,
            save_render_asset_npz,
            save_reference_assets_metadata,
            save_skeleton_sequence_npz,
        )

        return {
            "DEFAULT_METADATA_FILENAME": DEFAULT_METADATA_FILENAME,
            "RENDER_ASSET_SCHEMA_VERSION": RENDER_ASSET_SCHEMA_VERSION,
            "SUPPORTED_VIDEO_EXTENSIONS": SUPPORTED_VIDEO_EXTENSIONS,
            "ReferenceAssetBundle": ReferenceAssetBundle,
            "ReferenceVideoEntry": ReferenceVideoEntry,
            "build_reference_asset_bundle": build_reference_asset_bundle,
            "discover_reference_videos": discover_reference_videos,
            "load_reference_manifest": load_reference_manifest,
            "save_render_asset_npz": save_render_asset_npz,
            "save_reference_assets_metadata": save_reference_assets_metadata,
            "save_skeleton_sequence_npz": save_skeleton_sequence_npz,
            "build_reference_assets_metadata": build_reference_assets_metadata,
            "build_reference_assets": build_reference_assets,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
