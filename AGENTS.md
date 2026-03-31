# sam-3d-body Agent Guide

Scope: everything under `sam-3d-body/`.

This fork should stay easy to rebase on upstream. Prefer narrow changes that
improve local harnessing without restructuring the model code.

Branching and landing:

- Land `sam-3d-body` changes directly on `custom/main`.
- Do not open a separate `sam-3d-body` PR from this workspace for routine fork
  maintenance; commit on `custom/main`, push it, then update the root
  repository's submodule pointer.

Project shape:

- Python package in `sam_3d_body/`
- FastAPI service in `api/`
- CLI helpers in `scripts/`
- deterministic tests in `tests/`

Runtime boundary:

- `sam-3d-body` is Linux-only for local development, tests, and inference.
- The expected development environment is `conda activate sam_3d_body`.
- Run all Python scripts, CLIs, tests, and service entrypoints for this module inside the `sam_3d_body` conda environment. Do not rely on the system Python for `sam-3d-body` tasks; use `conda activate sam_3d_body` or `conda run -n sam_3d_body ...`.

Validation lanes:

- Fast/strict baseline: `./scripts/run_linux_pytest.sh tests/test_api_main.py tests/test_build_reference_assets_script.py tests/test_reference_assets.py tests/test_technique_alignment.py tests/test_video_processor.py`
- Deterministic smoke: `./scripts/run_linux_pytest.sh tests/test_reference_assets_contract.py`
- Real model smoke stays opt-in and should not block normal PRs.

Harness expectations:

- Keep public reference-asset outputs schema-driven and versioned.
- Treat `metadata.json` as a public contract consumed by other modules.
- Do not add repo-tracked samples with machine-local absolute paths.
- Favor lightweight test/runtime dependencies for CI-compatible paths.
