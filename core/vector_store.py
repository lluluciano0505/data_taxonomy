"""core/vector_store.py — Lightweight in-memory similarity index for L3 cross-file context.

Ranking uses token overlap over the L2 classification text (labels, keywords,
and summary). The lightweight index is intentionally in-memory and has no
external service or persistent project data.

Designed for 200–2000 files; no external server, no persistence.
"""
from __future__ import annotations

import re
import logging
import threading

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Incremental in-memory similarity index.

    add(text, meta, key=None)  — store text with a metadata dict.
                                 Provide key to allow later update() calls.
    update(key, updates)       — merge updates into an existing entry's metadata.
    search(query, k=5,
           exclude_key=None)   — token-overlap top-k.
                                 Pass exclude_key to skip the file being scored.
    """

    def __init__(self) -> None:
        self._texts: list[str] = []      # lowercased classification text
        self._meta:  list[dict] = []     # metadata dicts, mutable
        self._index: dict[str, int] = {} # key → position in the lists
        self._lock   = threading.Lock()  # guards update() for parallel Phase 2

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._meta)

    def add(self, text: str, meta: dict, key: str | None = None) -> None:
        pos = len(self._meta)
        self._texts.append(text.lower())
        self._meta.append(dict(meta))
        if key is not None:
            self._index[key] = pos

    def update(self, key: str, updates: dict) -> None:
        with self._lock:
            pos = self._index.get(key)
            if pos is not None:
                self._meta[pos].update(updates)

    def search(
        self,
        query: str,
        k: int = 5,
        exclude_key: str | None = None,
    ) -> list[dict]:
        if not self._meta:
            return []
        excl_pos = self._index.get(exclude_key) if exclude_key else None
        ranked   = self._keyword(query)
        results  = []
        for pos, _ in ranked:
            if pos == excl_pos:
                continue
            results.append(self._meta[pos])
            if len(results) == k:
                break
        return results

    # ── Internal ─────────────────────────────────────────────────────────────

    def _keyword(self, query: str) -> list[tuple[int, float]]:
        tokens = {t for t in re.split(r"[\s,;]+", query.lower()) if t}
        def sc(text: str) -> float:
            return sum(1 for t in tokens if t in text) / max(len(tokens), 1)
        return sorted(enumerate(self._texts), key=lambda x: sc(x[1]), reverse=True)
