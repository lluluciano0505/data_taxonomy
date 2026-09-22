"""core/output.py — CSV schema, sanitization, and read/write helpers.

Single source of truth for the pipeline's output format.
"""

from __future__ import annotations

import csv
from pathlib import Path


# ── CSV column schema ─────────────────────────────────────────────────────────

FIELDNAMES: list[str] = [
    "filename", "format", "file_path", "size_kb",
    "page_count", "folder",
    # Layer 1 extraction
    "extraction_coverage", "extraction_method", "year_confidence",
    "drawing_mode",
    "is_data_asset", "info_value_hint",
    # Layer 2 classification
    "information_type", "year", "domain", "scale", "lifecycle",
    "domain_candidates",
    "asset_type", "asset_type_candidates", "short_summary", "keywords",
    # Layer 4 trust / risk
    "governance", "confidentiality", "confidentiality_reason",
    "certainty", "certainty_score", "certainty_reason",
    "age_warning", "review_priority", "action", "review_reasons",
    # Layer 3 design decision priority (5 rubric-graded dimensions; files rank
    # lexicographically by authority > scope > coverage — no composite score)
    "authority", "authority_reason",
    "scope", "scope_reason",
    "urgency", "urgency_reason",
    "coverage", "coverage_reason",
    "accessibility", "accessibility_reason",
    "decision_priority_reason",
    # Audit
    "_reasoning", "llm_status", "processed_at",
]


# ── Row sanitization ──────────────────────────────────────────────────────────

def sanitize_row(row: dict) -> dict:
    """Strip newlines/tabs from all string values so CSV cells don't break."""
    def _clean(v):
        if isinstance(v, str):
            return v.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
        return v
    return {k: _clean(v) for k, v in row.items()}


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def write_rows(rows: list[dict], path: Path, mode: str = "w") -> None:
    """Write rows to CSV using FIELDNAMES schema. mode='w' writes header; 'a' appends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path: Path) -> list[dict]:
    """Read all rows from a CSV file. Returns empty list if file doesn't exist."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_extraction_failed(coverage: str) -> bool:
    """Return True when coverage indicates no usable content was produced."""
    c = str(coverage).lower()
    return "extraction failed" in c or "no readable content" in c or "unsupported format" in c


def read_processed_paths(path: Path) -> set[str]:
    """Return the set of file_path values already recorded in an existing CSV."""
    paths: set[str] = set()
    if not path.exists():
        return paths
    try:
        for row in read_rows(path):
            fp = (row.get("file_path") or "").strip()
            if fp:
                paths.add(fp)
    except Exception:
        pass
    return paths
