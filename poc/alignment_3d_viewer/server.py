#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlsplit


POC_ROOT = Path(__file__).resolve().parent
WEB_ROOT = POC_ROOT / "web"
NODE_MODULES_ROOT = POC_ROOT / "node_modules"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the SAM-3D-Body 3D viewer POC.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--data-dir", default=str(POC_ROOT / "data"))
    return parser.parse_args()


def _safe_child(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes the configured root")
    return candidate


def _parse_byte_range(value: str, size: int) -> tuple[int, int]:
    """Parse one HTTP byte range and return its inclusive bounds."""
    if size <= 0 or not value.startswith("bytes=") or "," in value:
        raise ValueError("Unsupported byte range")
    start_text, separator, end_text = value.removeprefix("bytes=").partition("-")
    if not separator:
        raise ValueError("Malformed byte range")
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - suffix_length), size - 1
    start = int(start_text)
    end = min(int(end_text), size - 1) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Range outside file")
    return start, end


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".bin": "application/octet-stream",
    }

    def __init__(self, *args: object, data_root: Path, **kwargs: object) -> None:
        self.data_root = data_root.resolve()
        self._response_range: tuple[int, int] | None = None
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def send_head(self) -> BinaryIO | None:
        self._response_range = None
        range_header = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if range_header and path.is_file():
            size = path.stat().st_size
            try:
                start, end = _parse_byte_range(range_header, size)
            except (TypeError, ValueError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            file = path.open("rb")
            self._response_range = (start, end)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
            self.end_headers()
            file.seek(start)
            return file
        return super().send_head()

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        if self._response_range is None:
            super().copyfile(source, outputfile)
            return
        start, end = self._response_range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if request_path.startswith("/data/"):
            try:
                return str(_safe_child(self.data_root, request_path.removeprefix("/data/")))
            except ValueError:
                return str(self.data_root / "__invalid_path__")

        vendor_files = {
            "/vendor/three.module.js": NODE_MODULES_ROOT / "three" / "build" / "three.module.js",
            "/vendor/three.core.js": NODE_MODULES_ROOT / "three" / "build" / "three.core.js",
        }
        if request_path in vendor_files:
            return str(vendor_files[request_path])
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        request_path = urlsplit(self.path).path
        if request_path.endswith((".mp4", ".mov")):
            self.send_header("Accept-Ranges", "bytes")
        if request_path.startswith("/data/") or request_path.endswith(
            ("/", ".html", ".css", ".js")
        ):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir).expanduser().resolve()
    metadata_path = data_root / "viewer-data.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run export_assets.py before starting the viewer."
        )
    if not (NODE_MODULES_ROOT / "three" / "build" / "three.module.js").exists():
        raise FileNotFoundError("Three.js is missing. Run `npm install` in the POC directory.")

    handler = partial(ViewerRequestHandler, data_root=data_root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"SAM-3D-Body 3D viewer: http://{args.host}:{args.port}")
    print(f"Data directory: {data_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
