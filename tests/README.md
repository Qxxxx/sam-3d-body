# SAM 3D Body Tests

Test suite for the SAM 3D Body video inference API.

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-asyncio httpx
pip install -r requirements-api.txt
```

### Run all tests

```bash
cd sam-3d-body
python -m pytest tests/ -v
```

### Run specific test file

```bash
python -m pytest tests/test_video_processor.py -v
python -m pytest tests/test_skeleton_processing.py -v
python -m pytest tests/test_alignment_scoring.py -v
python -m pytest tests/test_api.py -v
```

### Run with coverage

```bash
python -m pytest tests/ --cov=sam_3d_body --cov=api --cov-report=html
```

## Test Structure

- `conftest.py` - Shared fixtures (test videos, skeletons, etc.)
- `test_video_processor.py` - Video frame extraction and subject tracking tests
- `test_skeleton_processing.py` - Skeleton normalization and smoothing tests
- `test_alignment_scoring.py` - FastDTW alignment and scoring tests
- `test_api.py` - FastAPI endpoint tests

## Test Coverage

### VideoFrameExtractor
- ✅ Frame extraction at correct FPS
- ✅ Time range cropping
- ✅ Retry logic
- ✅ URL download and cleanup

### SubjectTracker
- ✅ IoU calculation accuracy
- ✅ Temporal consistency tracking
- ✅ Multi-detection selection

### SkeletonNormalizer
- ✅ Translation normalization (pelvis centering)
- ✅ Scale normalization (unit height)
- ✅ Orientation normalization (facing direction)
- ✅ Serialization/deserialization

### TemporalSmoother
- ✅ Gaussian smoothing
- ✅ Savitzky-Golay smoothing
- ✅ Short sequence handling

### FastDTWAligner
- ✅ Sequence alignment
- ✅ Different length sequences
- ✅ Valid alignment path generation

### PoseScorer
- ✅ Overall score calculation
- ✅ Joint error analysis
- ✅ Severity classification
- ✅ Tip generation

### API Endpoints
- ✅ Health check
- ✅ Feature flags
- ✅ Video inference (mocked)
- ✅ Alignment endpoint
- ✅ Error handling

## Notes

- Tests that require the actual model use mocking to avoid GPU dependencies
- Integration tests require test video fixtures (auto-generated)
- API tests use FastAPI's TestClient for in-process testing
