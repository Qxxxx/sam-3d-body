# sam-3d-body Agent Guide

Scope: everything under `sam-3d-body/`.

This fork should stay easy to rebase on upstream. Prefer narrow changes that
improve local harnessing without restructuring the model code.

Project shape:

- Python package in `sam_3d_body/`
- FastAPI service in `api/`
- CLI helpers in `scripts/`
- deterministic tests in `tests/`

Runtime boundary:

- `sam-3d-body` is Linux-only for local development, tests, and inference.
- The expected development environment is `conda activate sam_3d_body`.

Validation lanes:

- Fast/strict baseline: `./scripts/run_linux_pytest.sh tests/test_api_main.py tests/test_build_reference_assets_script.py tests/test_reference_assets.py tests/test_technique_alignment.py tests/test_video_processor.py`
- Deterministic smoke: `./scripts/run_linux_pytest.sh tests/test_reference_assets_contract.py`
- Real model smoke stays opt-in and should not block normal PRs.

Harness expectations:

- Keep public reference-asset outputs schema-driven and versioned.
- Treat `metadata.json` as a public contract consumed by other modules.
- Do not add repo-tracked samples with machine-local absolute paths.
- Favor lightweight test/runtime dependencies for CI-compatible paths.
