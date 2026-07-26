"""Content-aware resume manifest and atomic output helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any


MANIFEST_FILENAME = ".dataminder-manifest.json"


def stable_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relative_source_id(path: str, source_dir: str) -> str:
    return os.path.relpath(path, source_dir).replace(os.sep, "/")


def output_paths(path: str, source_dir: str, dest_dir: str) -> dict[str, str]:
    """Return collision-free outputs while preserving the source hierarchy."""
    relative_path = os.path.relpath(path, source_dir)
    # Keep the source extension as part of the sidecar name: ``report.pdf`` and
    # ``report.docx`` must remain distinct even when they share a directory.
    base = os.path.join(dest_dir, relative_path)
    return {
        "markdown": f"{base}.md",
        "chunks": f"{base}.chunks.jsonl",
        "document": f"{base}.document.json",
    }


def atomic_write_text(path: str, content: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".dataminder-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


class ProcessingManifest:
    """Small JSON manifest keyed by source-relative path."""

    def __init__(self, dest_dir: str):
        self.path = os.path.join(dest_dir, MANIFEST_FILENAME)
        self.data = {"schema_version": 1, "documents": {}}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as source:
                loaded = json.load(source)
            if loaded.get("schema_version") == 1 and isinstance(
                loaded.get("documents"), dict
            ):
                self.data = loaded
        except (OSError, ValueError):
            # A damaged manifest should trigger safe reprocessing, never data loss.
            self.data = {"schema_version": 1, "documents": {}}

    def is_current(
        self,
        source_id: str,
        *,
        source_sha256: str,
        pipeline_fingerprint: str,
        outputs: dict[str, str],
    ) -> bool:
        entry = self.data["documents"].get(source_id)
        if not entry or entry.get("status") != "success":
            return False
        if entry.get("source_sha256") != source_sha256:
            return False
        if entry.get("pipeline_fingerprint") != pipeline_fingerprint:
            return False
        return all(os.path.exists(path) for path in outputs.values())

    def mark_success(
        self,
        source_id: str,
        *,
        source_sha256: str,
        pipeline_fingerprint: str,
        outputs: dict[str, str],
        document_id: str,
    ) -> None:
        self.data["documents"][source_id] = {
            "status": "success",
            "source_sha256": source_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "document_id": document_id,
            "outputs": {
                name: os.path.relpath(path, os.path.dirname(self.path)).replace(
                    os.sep, "/"
                )
                for name, path in outputs.items()
            },
        }
        self.save()

    def save(self) -> None:
        atomic_write_text(
            self.path,
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
