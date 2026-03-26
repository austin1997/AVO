"""Structured logging configuration for AVO."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for trajectory analysis."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__
            and k not in ("message", "msg", "args", "exc_info", "exc_text", "stack_info")
        }
        if extra:
            entry["extra"] = extra
        return json.dumps(entry)


def setup_logging(
    log_dir: Path | str | None = None,
    level: str = "INFO",
    json_logs: bool = True,
) -> None:
    """Configure logging for an AVO run.

    Logs go to both stderr (human-readable) and an optional JSON file.
    """
    root = logging.getLogger("avo")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(console)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "avo.log")
        if json_logs:
            fh.setFormatter(JSONFormatter())
        else:
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
        root.addHandler(fh)
