# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pytorch_lightning.utilities import rank_zero_only
except ModuleNotFoundError:
    # Lightweight CI paths import this module without model dependencies.
    def rank_zero_only(fn):  # type: ignore[no-redef]
        return fn


_LOGGER_NAMES = ("sam3d", "sam_3d_body")
_HANDLER_NAME = "sam3d-json-stream"
_RESERVED_LOG_RECORD_KEYS = frozenset(logging.makeLogRecord({}).__dict__) | {
    "asctime",
    "message",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _coerce_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = str(level).strip().upper()
    resolved = logging.getLevelName(normalized)
    return resolved if isinstance(resolved, int) else logging.INFO


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
        }

        event_payload = getattr(record, "event_payload", None)
        if isinstance(event_payload, dict):
            payload.update(_json_safe(event_payload))

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_KEYS or key == "event_payload":
                continue
            payload[key] = _json_safe(value)

        if "message" not in payload:
            payload["message"] = record.getMessage()

        if record.exc_info:
            payload["excInfo"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stackInfo"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str | int = logging.INFO) -> None:
    resolved_level = _coerce_log_level(level)

    for logger_name in _LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger.setLevel(resolved_level)
        logger.propagate = False

        handler = next(
            (
                item
                for item in logger.handlers
                if item.get_name() == _HANDLER_NAME
            ),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler()
            handler.set_name(_HANDLER_NAME)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        handler.setLevel(resolved_level)


def log_event(
    logger: logging.Logger,
    level: str,
    payload: dict[str, Any],
) -> None:
    message = str(payload.get("message") or payload.get("event") or "log_event")
    level_name = level.lower()
    method_name = {
        "debug": "debug",
        "warn": "warning",
        "warning": "warning",
        "error": "error",
    }.get(level_name, "info")
    getattr(logger, method_name)(
        message,
        extra={"event_payload": _json_safe(payload)},
    )


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
