"""Structured per-run observability for document processing."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from processing_manifest import atomic_write_text


DEFAULT_RUN_REPORT = ".dataminder-last-run.json"


class PipelineRunReport:
    def __init__(self, source_dir: str, dest_dir: str, config=None):
        self._started_monotonic = time.perf_counter()
        self.data = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": os.path.abspath(source_dir),
            "destination_dir": os.path.abspath(dest_dir),
            "configuration": dict(config or {}),
            "documents": [],
        }

    def set_configuration(self, config: dict[str, Any]) -> None:
        self.data["configuration"] = config

    def add_document(self, **result) -> None:
        self.data["documents"].append(result)

    def finish(self) -> dict[str, Any]:
        documents = self.data["documents"]
        counts = {
            status: sum(item.get("status") == status for item in documents)
            for status in ("success", "skipped", "failed")
        }
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.data["elapsed_seconds"] = round(
            time.perf_counter() - self._started_monotonic, 6
        )
        self.data["summary"] = {
            "document_count": len(documents),
            **counts,
            "total_characters": sum(
                item.get("char_count", 0)
                for item in documents
                if item.get("status") == "success"
            ),
            "total_chunks": sum(
                item.get("chunk_count", 0)
                for item in documents
                if item.get("status") == "success"
            ),
        }
        return self.data

    def write(self, path: str) -> dict[str, Any]:
        report = self.finish()
        atomic_write_text(
            path,
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return report
