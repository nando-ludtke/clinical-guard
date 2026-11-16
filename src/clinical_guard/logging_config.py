"""Structured JSON logging configuration with privacy-safe defaults."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a LogRecord and its extra structured fields to JSON."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def redact_query_metadata(query: str) -> dict[str, int]:
    """Return non-sensitive metadata for a query without exposing raw text."""
    return {"query_char_length": len(query)}


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with structured JSON output on stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
