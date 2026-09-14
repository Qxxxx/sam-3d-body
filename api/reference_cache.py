"""Persistent, versioned cache for registered reference assets (never user media)."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sam_3d_body.technique_alignment import resolve_npz_file


def _cache_key(source: str, version: str, kind: str) -> str:
    parts = urlsplit(source)
    # Only remove S3 signing parameters. Preserve object versionId and any other
    # query parameters that can change the resource, plus the bucket/host/path.
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                   if not k.lower().startswith("x-amz-"))
    identity = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    return hashlib.sha256(json.dumps([identity, version, kind]).encode()).hexdigest()


def _fingerprint(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


@contextmanager
def prepare_reference_asset(
    source: str,
    *,
    cache_root: Path,
    version: str | None,
    kind: str,
    validate: Callable[[Path], object],
) -> Iterator[Path]:
    """Resolve and validate before inference; publish only complete cache entries.

    Old callers without a version still prefetch, but cannot safely reuse a
    mutable URL forever. Their temporary file lives until this context exits.
    """
    if not version or urlsplit(source).scheme not in {"http", "https"}:
        with resolve_npz_file(source) as path:
            validate(path)
            yield path
        return

    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = _cache_key(source, version, kind)
    destination = cache_root / f"{key}.npz"
    receipt = cache_root / f"{key}.json"
    # A separate open per caller gives both threads and processes an exclusive
    # lock. Keep lock files: unlinking a held lock would allow a second writer.
    with (cache_root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            valid = destination.is_file() and json.loads(receipt.read_text()) == _fingerprint(destination)
        except (OSError, ValueError):
            valid = False
        if not valid:
            # The downloader retries transient failures with fresh temp files.
            # Validation failures never publish a partial/corrupt cache entry.
            with resolve_npz_file(source) as downloaded:
                validate(downloaded)
                with tempfile.NamedTemporaryFile(dir=cache_root, suffix=".part", delete=False) as stream:
                    pending = Path(stream.name)
                pending_receipt = pending.with_suffix(".json.part")
                try:
                    shutil.copyfile(downloaded, pending)
                    metadata = _fingerprint(pending)
                    pending_receipt.write_text(json.dumps(metadata))
                    os.replace(pending, destination)
                    os.replace(pending_receipt, receipt)
                finally:
                    pending.unlink(missing_ok=True)
                    pending_receipt.unlink(missing_ok=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
    yield destination
