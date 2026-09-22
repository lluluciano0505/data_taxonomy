"""
Project Intelligence — scans the input directory for meeting minutes, RFI logs,
and decision documents, then extracts open decisions using one LLM call.

Called once at pipeline start; result is passed to every layer3_priority() call
to ground the urgency dimension in real project context rather than phase heuristics.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ClientT = Any

# Filename patterns that suggest meeting / decision documents
_MEETING_PATTERNS = [
    r"(?i)(meeting|minutes|mom|mot\b)",
    r"(?i)(rfi.?log|decision.?log|action.?item|issue.?log|open.?item)",
    r"(?i)(weekly|progress.?report|status.?report)",
    r"(?i)(workshop|design.?review|coordination.?meeting)",
    r"(?i)(beslut|protokol|referat|sitzung|procès.?verbal)",   # Scandinavian / DE / FR
    r"(?i)(site.?meeting|project.?update|handover|kick.?off)",
]

_CANDIDATE_EXTS = {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"}
_MAX_FILES  = 8     # analyse at most this many meeting files
_MAX_CHARS  = 2000  # content budget per file


def _is_meeting_file(path: Path) -> bool:
    return any(re.search(pat, path.name) for pat in _MEETING_PATTERNS)


def _read_text(path: Path) -> str:
    """Best-effort text extraction — no LLM, no routing."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            text = ""
            for page in list(reader.pages)[:6]:
                text += (page.extract_text() or "")
                if len(text) >= _MAX_CHARS:
                    break
            return text[:_MAX_CHARS]

        if ext in (".docx", ".doc"):
            import docx  # type: ignore
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs[:40] if p.text.strip())[:_MAX_CHARS]

        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")[:_MAX_CHARS]

        if ext in (".xlsx", ".xls"):
            import pandas as pd  # type: ignore
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            df = pd.read_excel(str(path), nrows=25, engine=engine)
            return df.to_string()[:_MAX_CHARS]

    except Exception:
        pass
    return ""


_INTEL_PROMPT = """\
You are analysing recent project documents to identify what design decisions
are currently open and what the team urgently needs right now.

PROJECT DOCUMENTS:
{files_block}

Return ONLY valid JSON — no markdown, no commentary:
{{
  "open_decisions": [
    "<specific open decision — state the discipline, the decision topic, and what information is needed to close it>"
  ],
  "active_design_questions": [
    "<unresolved technical or design question from the documents>"
  ],
  "urgency_context": "<2–3 sentences: what are the most time-sensitive issues right now, and what types of project files would be most useful to the team this week>",
  "phase_signal": "<1 sentence: what project phase/stage these documents suggest the project is currently in>"
}}

Rules:
- Be specific: use actual names, disciplines, topics, and numbers from the documents.
- Max 6 open_decisions, max 4 active_design_questions.
- If documents are too sparse, return best effort with what is available.
"""

_EMPTY_INTEL: dict = {
    "open_decisions": [],
    "active_design_questions": [],
    "urgency_context": (
        "No meeting or decision documents found — "
        "urgency will be scored from classification signals only."
    ),
    "phase_signal": "Unknown.",
    "sources": [],
}


def extract_project_intelligence(
    input_path: Path,
    client: _ClientT,
    model: str,
    api_timeout: int = 30,
) -> dict:
    """
    Scan input_path for meeting/decision documents and extract open decisions.

    Called once at pipeline start.  Returns a dict that is injected into every
    layer3_priority() call to calibrate the urgency dimension.
    """
    candidates: list[Path] = []
    try:
        candidates = [
            f for f in input_path.rglob("*")
            if f.is_file()
            and f.suffix.lower() in _CANDIDATE_EXTS
            and _is_meeting_file(f)
        ]
    except Exception as exc:
        logger.warning("project_intel: directory scan failed: %s", exc)
        return dict(_EMPTY_INTEL)

    if not candidates:
        logger.info("project_intel: no meeting/decision files found in %s", input_path)
        return dict(_EMPTY_INTEL)

    # Most recent first; take top N
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    selected = candidates[:_MAX_FILES]
    logger.info(
        "project_intel: scanning %d file(s): %s",
        len(selected), [f.name for f in selected],
    )

    files_block_parts: list[str] = []
    sources: list[str] = []
    for f in selected:
        text = _read_text(f)
        if text.strip():
            files_block_parts.append(f"[{f.name}]\n{text}")
            sources.append(f.name)

    if not files_block_parts:
        return {**dict(_EMPTY_INTEL), "sources": [f.name for f in selected]}

    prompt = _INTEL_PROMPT.format(files_block="\n\n---\n\n".join(files_block_parts))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a project intelligence analyst. "
                        "Respond ONLY with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=api_timeout,
        )
        raw     = resp.choices[0].message.content or ""
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        result: dict = json.loads(match.group(0)) if match else json.loads(cleaned)

        result.setdefault("open_decisions",          [])
        result.setdefault("active_design_questions", [])
        result.setdefault("urgency_context",         _EMPTY_INTEL["urgency_context"])
        result.setdefault("phase_signal",            "Unknown.")
        result["sources"] = sources
        return result

    except Exception as exc:
        logger.warning("project_intel: LLM extraction failed: %s", exc)
        return {**dict(_EMPTY_INTEL), "sources": sources}


def format_for_prompt(intel: dict | None) -> str:
    """
    Format project intelligence as a compact text block for injection into prompts.
    Returns a ready-to-embed string (no leading/trailing newlines).
    """
    if not intel:
        return "  (no project intelligence available)"

    lines: list[str] = []

    if decisions := intel.get("open_decisions"):
        lines.append("  OPEN DECISIONS:")
        for d in decisions[:6]:
            lines.append(f"    • {d}")

    if questions := intel.get("active_design_questions"):
        lines.append("  ACTIVE DESIGN QUESTIONS:")
        for q in questions[:4]:
            lines.append(f"    • {q}")

    if ctx := intel.get("urgency_context", "").strip():
        lines.append(f"  URGENCY CONTEXT: {ctx}")

    if phase := intel.get("phase_signal", "").strip():
        lines.append(f"  PHASE SIGNAL: {phase}")

    if sources := intel.get("sources"):
        label = ", ".join(sources[:3]) + ("…" if len(sources) > 3 else "")
        lines.append(f"  (Sources: {label})")

    return "\n".join(lines) if lines else "  (meeting documents found but no decisions extracted)"
