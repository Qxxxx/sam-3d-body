"""Private account erasure for backend-owned technique jobs, not reference assets."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import fcntl
from pathlib import Path
import re
import shutil
from threading import RLock


class AccountErasure:
    def __init__(self, artifact_root: str) -> None:
        self.root = Path(artifact_root).expanduser().resolve()
        self.lock = RLock()
        self.active: dict[str, int] = {}

    def _receipt(self, user_id: str) -> Path:
        digest = sha256(user_id.encode()).hexdigest()
        return self.root / "account-erasure" / f"{digest}.json"

    def is_deleted(self, user_id: str | None) -> bool:
        return bool(user_id and self._receipt(user_id).exists())

    @contextmanager
    def processing(self, user_id: str | None):
        if not user_id:
            yield
            return
        lock_path = self._receipt(user_id).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lease:
            # A shared filesystem lease also protects multiple service processes
            # using the same artifact root. No GPU thread cancellation is assumed.
            fcntl.flock(lease, fcntl.LOCK_SH)
            with self.lock:
                if self.is_deleted(user_id):
                    raise ValueError("account_deleted")
                self.active[user_id] = self.active.get(user_id, 0) + 1
            try:
                yield
            finally:
                with self.lock:
                    self.active[user_id] -= 1
                    if self.active[user_id] == 0:
                        del self.active[user_id]

    def erase(self, user_id: str) -> bool:
        with self.lock:
            receipt = self._receipt(user_id)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            # A tombstone is installed before checking active work. New/queued
            # jobs cannot start while a running inference thread winds down.
            receipt.touch(exist_ok=True)
            if self.active.get(user_id, 0):
                return False
            with receipt.with_suffix(".lock").open("a") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                return self._erase_files(user_id)

    def _erase_files(self, user_id: str) -> bool:
        records: list[tuple[Path, dict]] = []
        for path in (self.root / "jobs").glob("*/job.json"):
            # Unknown/corrupt job ownership requires repair, never deletion
            # by guessing a path or an unconditional success response.
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("invalid_job_record")
            records.append((path, record))
        pending = False
        for path, record in records:
            if record.get("traceContext", {}).get("userId") != user_id:
                continue
            payload = record.get("request", {})
            storage = record.get("erasureStorage") or payload.get("storage", {})
            prefix = storage.get("prefix", "")
            # Public references and custom output directories are outside
            # this endpoint's authority and must be handled explicitly.
            if storage.get("managed") is False or storage.get("outputDir") or storage.get("output_dir") or not re.fullmatch(r"technique-analysis/[A-Za-z0-9_-]+", prefix):
                pending = True
                continue
            if any(other.get("traceContext", {}).get("userId") != user_id
                   and (other.get("erasureStorage") or other.get("request", {}).get("storage", {})).get("prefix") == prefix
                   for _, other in records):
                pending = True
                continue
            output = self.root / prefix
            job_directory = path.parent
            if output.resolve() != output or job_directory.resolve() != job_directory:
                pending = True
                continue
            if output.exists():
                shutil.rmtree(output)
            # Remove request URLs, pose results and callback state only
            # after the associated output removal succeeded.
            shutil.rmtree(job_directory)
        return not pending
