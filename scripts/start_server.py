#!/usr/bin/env python3
"""
Startup script for SAM 3D Body inference API server.

Usage:
    python scripts/start_server.py
    python scripts/start_server.py --port 8000 --host 0.0.0.0

Environment Variables:
    SAM3DB_API_PORT: API server port (default: 8000)
    SAM3DB_API_HOST: API server host (default: 0.0.0.0)
    SAM3DB_MODEL_CHECKPOINT_PATH: Path to model checkpoint
    SAM3DB_DEVICE: Device to use (cuda or cpu)
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []

    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")

    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")

    try:
        import httpx
    except ImportError:
        missing.append("httpx")

    try:
        import fastdtw
    except ImportError:
        missing.append("fastdtw")

    if missing:
        print("Error: Missing required dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements-api.txt")
        sys.exit(1)


def check_model_checkpoints():
    """Check if model checkpoints are available."""
    checkpoint_path = os.environ.get(
        "SAM3DB_MODEL_CHECKPOINT_PATH",
        "./checkpoints/sam-3d-body-dinov3/model.ckpt"
    )

    if not Path(checkpoint_path).exists():
        print("Warning: Model checkpoint not found!")
        print(f"  Expected at: {checkpoint_path}")
        print("\nDownload with:")
        print("  hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3")
        print("\nServer will start but inference endpoints will return errors.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Start SAM 3D Body inference API server"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SAM3DB_API_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SAM3DB_API_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (for development)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Check model checkpoints
    has_checkpoints = check_model_checkpoints()

    # Import after dependency check
    import uvicorn

    print("=" * 60)
    print("SAM 3D Body Inference API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print(f"Model loaded: {has_checkpoints}")
    print("=" * 60)
    print("Endpoints:")
    print(f"  Health:     http://{args.host}:{args.port}/health")
    print(f"  Video:      http://{args.host}:{args.port}/infer/video")
    print(f"  Alignment:  http://{args.host}:{args.port}/infer/alignment")
    print(f"  Docs:       http://{args.host}:{args.port}/docs")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()

    # Start server
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
