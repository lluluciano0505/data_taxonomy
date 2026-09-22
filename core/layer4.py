"""Layer 4 — Rule-based trust & risk assessment.  No LLM call.

Also applies two deterministic post-classification guards that correct known
LLM biases before the results are used downstream:

  Guard 1 — Confidentiality:
      Drawing-like files are forced to Not Confidential unless a hard
      sensitive-term signal is found in the filename or content.
      Email files are always Confidential.

  Guard 2 — Data asset:
      Files in tabular/spatial data formats (csv, xlsx, json, shp…) are
      forced to asset_type=Data when the LLM returned a generic label
      (Document, Unknown).
"""

from datetime import datetime
from pathlib import Path
import logging
import re
from .output import is_extraction_failed

logger = logging.getLogger(__name__)

_BINARY_FORMAT_EXEMPT = {
    # CAD — binary, no readable text layer
    ".dwg", ".rvt", ".nwd", ".dwf",
    # Video
    ".mp4", ".mov",
    # Archive formats not supported by layer1
    ".7z", ".rar",
    # Outlook proprietary binary
    ".msg",
}

# Formats that are definitively data/tabular regardless of LLM label.
_DATA_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".shp", ".geojson", ".kml", ".gpkg"}

# Formats that are definitively drawing/CAD — never confidential.
_DRAWING_FORMATS = {".dwg", ".dxf", ".dwf", ".ifc", ".rvt", ".nwd"}

# Hard evidence of sensitive content — override any LLM/prescreen label.
_SENSITIVE_TERMS = {
    # Financial — personal or firm-level only (not generic project cost docs)
    "payroll", "salary", "bank account", "account number",
    "balance sheet", "income statement", "profit and loss",
    "financial statement", "financial record",
    # Legal
    "legal advice", "legal counsel", "without prejudice", "privileged",
    "litigation", "settlement", "non-disclosure", "nda",
    # Personnel / HR
    "personnel record", "staff record", "disciplinary", "performance review",
    "termination", "employment record", "hr record", "grievance",
    "medical record", "health insurance",
    # Security / access
    "security clearance", "classified", "clearance level",
    "evacuation plan", "emergency protocol",
    # Explicit markers
    "strictly confidential", "private and confidential",
    "internal only", "not for distribution", "for internal use",
    "eyes only", "secret", "top secret",
}

# Word-boundary-anchored patterns for the terms above. A plain `term in text`
# substring check false-positives badly on ordinary AEC vocabulary — "nda"
# (meant to catch the "NDA" acronym) matches inside "age NDA", "seco NDA ry",
# "bou NDA ry" — so every filename containing "agenda", "secondary", or
# "boundary" was silently tripping the confidentiality guard. Matching is
# anchored on non-alphanumeric boundaries (not just \b) so separators like
# "_" and "-" — common in filenames — still count as word breaks.
_SENSITIVE_TERM_PATTERNS = [
    re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")
    for term in _SENSITIVE_TERMS
]


# ── Guard helpers ─────────────────────────────────────────────────────────────

def _normalize_confidentiality(value: str) -> str:
    v = str(value or "").strip().lower()
    if v == "confidential":
        return "Confidential"
    return "Not Confidential"


def _is_drawing_like(file_path: Path, layer2: dict) -> bool:
    ext       = file_path.suffix.lower()
    info_type = str(layer2.get("information_type", "")).lower()
    asset     = str(layer2.get("asset_type", "")).strip().lower()
    if ext in _DRAWING_FORMATS:
        return True
    if asset == "drawing":
        return True
    return info_type in {"technical", "bim"}


def _apply_confidentiality_guard(
    file_path: Path,
    meta: dict,
    layer2: dict,
) -> dict:
    """Detect guard signals for confidentiality review (no modifications).

    Returns dict with:
      confidentiality: LLM's judgment (unchanged)
      confidentiality_reason: LLM's reason + guard signal
      guard_signal: human-readable description of what the guard detected
    """
    conf   = _normalize_confidentiality(layer2.get("confidentiality", "Not Confidential"))
    reason = str(layer2.get("confidentiality_reason", "")).strip()
    signal = ""

    # Email files always need review if LLM missed it
    if file_path.suffix.lower() in {".eml", ".msg", ".mbox"}:
        if conf != "Confidential":
            signal = "Email file should always be Confidential — review needed"

    # Detect sensitive terms — important for review if LLM disagreed
    filename    = str(meta.get("filename", "")).lower()
    content     = str(meta.get("content_sample", "")).lower()
    signal_blob = f"{filename}\n{content[:3000]}"
    has_sensitive_term = any(p.search(signal_blob) for p in _SENSITIVE_TERM_PATTERNS)
    if has_sensitive_term and conf != "Confidential":
        signal = "Sensitive term detected — LLM judgment conflicts with content signal"

    # Detect drawing-like files — important for review if LLM said Confidential
    is_drawing_like = _is_drawing_like(file_path, layer2)
    if is_drawing_like and conf == "Confidential" and not has_sensitive_term:
        signal = "Drawing-like file marked Confidential without sensitive term — review needed"

    return {
        "confidentiality": conf,
        "confidentiality_reason": reason,
        "guard_signal": signal,
    }


def _apply_data_asset_guard(
    file_path: Path,
    meta: dict,
    layer2: dict,
    settings: dict | None = None,
) -> dict:
    """Detect guard signals for asset type review (no modifications).

    Returns dict with:
      asset_type: LLM's judgment (unchanged)
      is_data_asset: based on LLM's asset_type
      guard_signal: human-readable description of what the guard detected
    """
    asset = str(layer2.get("asset_type", "Unknown")).strip()

    data_formats = _DATA_FORMATS
    if settings:
        raw = settings.get("data_formats")
        if raw:
            data_formats = {str(f).lower() if f.startswith(".") else f".{f.lower()}" for f in raw}

    ext = file_path.suffix.lower()
    prescreen = layer2.get("_prescreen") or {}
    pre_likelihood = str(prescreen.get("data_likelihood", "")).strip().lower()

    is_data_format = ext in data_formats or pre_likelihood in {"likely", "possible"}

    signal = ""
    # Detect conflict: format suggests data but LLM said generic label
    if is_data_format and asset.lower() in {"", "unknown", "document", "other"}:
        signal = f"Format {ext} suggests data, but LLM classified as {asset} — review recommended"

    is_data = "Yes" if asset.lower() == "data" else "No"
    return {
        "asset_type": asset,
        "is_data_asset": is_data,
        "guard_signal": signal,
    }




def _is_binary_exempt(file_path: Path) -> bool:
    return file_path.suffix.lower() in _BINARY_FORMAT_EXEMPT


def _age_warning(
    file_year: int,
    project_start: int,
    project_end: int,
    current_year: int,
    min_valid_year: int = 2000,
    warn_predates_years: int = 10,
    warn_postproject_years: int = 3,
    year_is_reliable: bool = True,
) -> str:
    """Return an age warning string, or '' if no warning is needed.

    Severity:
      ''         — no anomaly
      'NOTICE: ' — informational
      'WARNING:' — significant anomaly
    """
    if file_year < min_valid_year or file_year > current_year:
        return ""

    if file_year < project_start:
        years_before = project_start - file_year
        if years_before > warn_predates_years:
            msg = (f"dated {file_year} — {years_before} years before "
                   f"project start ({project_start}), verify relevance")
            if year_is_reliable:
                return f"WARNING: {msg}"
            return f"NOTICE: {msg} (year from OS mtime — low confidence)"
        if not year_is_reliable:
            return ""
        return f"NOTICE: dated {file_year} — predates project start ({project_start})"

    if file_year <= project_end:
        return ""

    years_after = file_year - project_end
    if years_after > warn_postproject_years and year_is_reliable:
        return f"NOTICE: dated {file_year} — {years_after} years after project end ({project_end})"
    return ""


def layer4_trust(
    file_path: Path,
    meta: dict,
    layer2: dict,
    project: dict,
    settings: dict | None = None,
    layer3: dict | None = None,
) -> dict:
    """Rule-based trust and risk assessment.  No LLM call.

    Parameters
    ----------
    file_path : Path to the file (used for extension-based format checks).
    meta      : Layer 1 output dict.
    layer2    : Layer 2 output dict (raw LLM classification — guard signals detected here).
    project   : Project config dict with year_range and optional age_analysis.
    settings  : Optional layer2_settings dict (used for data_formats config).
    layer3    : Optional Layer 3 output dict (used to factor design impact into review_priority).

    Returns
    -------
    dict with keys:
      asset_type, is_data_asset,          (LLM judgment, not modified by guards)
      governance, confidentiality,         (LLM judgment, not modified by guards)
      confidentiality_reason, age_warning,
      review_priority  ("Critical" | "High" | "Medium" | "Low"),
      action           (human-readable recommended next step),
      review_reasons   (comma-joined string),
    """
    # ── Unpack LLM judgment ──────────────────────────────────────────────
    domain          = str(layer2.get("domain",     "Unknown")).strip()
    governance      = str(layer2.get("governance", "Unknown")).strip()
    lifecycle       = str(layer2.get("lifecycle",  "Unknown")).strip()
    certainty_label = str(layer2.get("certainty", layer2.get("confidence", "Low"))).strip()

    # ── Guard signals (detect conflicts, no modifications) ────────────────
    conf_guard = _apply_confidentiality_guard(file_path, meta, layer2)
    asset_guard = _apply_data_asset_guard(file_path, meta, layer2, settings=settings)

    confidentiality = conf_guard["confidentiality"]
    confidentiality_reason = conf_guard["confidentiality_reason"]
    conf_guard_signal = conf_guard["guard_signal"]

    asset_type = asset_guard["asset_type"]
    is_data_asset = asset_guard["is_data_asset"]
    asset_guard_signal = asset_guard["guard_signal"]
    certainty_reason = str(layer2.get("certainty_reason", layer2.get("confidence_reason", ""))).strip()
    vlm_confidence = str(layer2.get("vlm_confidence", "")).strip()
    vlm_is_strong = vlm_confidence in {"High", "Medium"}

    llm_ok = not str(layer2.get("llm", "ok")).startswith("error")

    coverage          = str(meta.get("extraction_coverage", "")).strip()
    year_confidence   = str(meta.get("year_confidence",   "unknown")).strip()
    extraction_method = str(meta.get("extraction_method", "unknown")).strip()
    drawing_mode      = str(meta.get("drawing_mode", "")).strip()
    text_signal_len   = len(re.sub(r"\s+", "", str(meta.get("content_sample", ""))))

    # ── Project timeline ──────────────────────────────────────────────────
    yr            = project.get("year_range", []) if isinstance(project, dict) else []
    project_start = int(yr[0]) if yr else 2000
    project_end   = int(yr[1]) if len(yr) > 1 else datetime.now().year
    current_year  = datetime.now().year

    age_cfg = project.get("age_analysis", {}) if isinstance(project, dict) else {}

    # ── Age warning ───────────────────────────────────────────────────────
    age_warning = ""
    try:
        raw_year  = layer2.get("year") or meta.get("year")
        file_year = int(str(raw_year).split()[0]) if raw_year else 0
        age_warning = _age_warning(
            file_year              = file_year,
            project_start          = project_start,
            project_end            = project_end,
            current_year           = current_year,
            min_valid_year         = int(age_cfg.get("min_valid_year",         2000)),
            warn_predates_years    = int(age_cfg.get("warn_predates_years",      10)),
            warn_postproject_years = int(age_cfg.get("warn_postproject_years",    3)),
            year_is_reliable       = year_confidence in {"high", "medium"},
        )
    except Exception:
        pass

    # ── Signals (direct booleans, no counting) ────────────────────────────
    # Keep extraction failures visible for diagnostics, but do not treat them as
    # standalone flag escalators because scanned drawings often trigger false positives.
    is_unreadable    = is_extraction_failed(coverage) and not _is_binary_exempt(file_path)
    is_metadata_only = (
        extraction_method == "metadata"
        and not _is_binary_exempt(file_path)
        and not is_unreadable
    )
    is_metadata_only_risky = is_metadata_only and not vlm_is_strong
    is_drawing_weak_signal = (
        drawing_mode == "single_page_drawing"
        and text_signal_len < 120
        and not vlm_is_strong
    )
    is_unknown_src   = governance == "Unknown"
    is_low_certainty = certainty_label == "Low" and not vlm_is_strong

    # Guard signals — indicators that LLM judgment conflicts with format/content signals
    has_conf_guard_signal = bool(conf_guard_signal)
    has_asset_guard_signal = bool(asset_guard_signal)

    # ── Review priority ───────────────────────────────────────────────────
    if not llm_ok and not vlm_is_strong:
        review_priority = "Critical"
    elif domain == "Unknown" or lifecycle == "Unknown":
        review_priority = "High"
    elif is_low_certainty or is_metadata_only_risky:
        review_priority = "Medium"
    else:
        review_priority = "Low"

    # ── Actions ───────────────────────────────────────────────────────────
    actions: list[str] = []
    if (not llm_ok and not vlm_is_strong) or domain == "Unknown":
        actions.append("Manual classification")
    if is_low_certainty:
        actions.append(f"Manual review ({certainty_reason})" if certainty_reason else "Manual review")
    if is_unknown_src:
        actions.append("Verify source")
    if is_metadata_only_risky:
        actions.append("Verify file accessibility")
    if has_conf_guard_signal:
        actions.append("Review confidentiality judgment")
    if has_asset_guard_signal:
        actions.append("Review asset type classification")

    action = " → ".join(actions) if actions else "Auto-process"

    # ── Reasons (human-readable, for CSV / UI display) ────────────────────
    reasons: list[str] = []
    if not llm_ok and not vlm_is_strong: reasons.append("LLM classification failed")
    if is_unknown_src:    reasons.append("unknown source")
    if domain == "Unknown": reasons.append("unknown domain")
    if is_metadata_only:  reasons.append("metadata only — no content extracted")
    if is_low_certainty:
        reasons.append(f"low certainty{': ' + certainty_reason if certainty_reason else ''}")
    if lifecycle == "Unknown": reasons.append("unknown lifecycle")
    if has_conf_guard_signal: reasons.append("confidentiality signal conflict")
    if has_asset_guard_signal: reasons.append("asset type signal conflict")
    # Keep age_warning as a separate field; do not include it in model-quality review reasons.

    return {
        "asset_type":             asset_type,
        "is_data_asset":          is_data_asset,
        "governance":             governance,
        "confidentiality":        confidentiality,
        "confidentiality_reason": confidentiality_reason,
        "age_warning":            age_warning,
        "review_priority":        review_priority,
        "action":                 action,
        "review_reasons":         ", ".join(reasons),
    }
