"""Layer 2 — LLM Domain Classification.

Design philosophy
-----------------
* All classification is done by the LLM.  Deterministic post-processing guards
  exist only to correct two well-known systematic biases:
    1. Drawings mis-labelled Confidential (guard: trust prescreen over main call)
    2. Data formats not labelled as Data asset (guard: deterministic from extension)
* The LLM returns its own certainty_score (0–100) alongside the certainty label.
  No separate scoring system is needed.
* Taxonomy lives in taxonomy.yaml / taxonomy.json.
* routing_note from Layer 1 is surfaced in the prompt so the LLM understands
  why the content_sample looks the way it does.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import subprocess
import tempfile
import fnmatch
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ClientT = Any

try:
    from PIL import Image as _PILImg
    _PILImg.MAX_IMAGE_PIXELS = None
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Language detection (factual unicode ranges — not heuristic)
# ---------------------------------------------------------------------------

_LANGUAGE_PATTERNS = {
    "chinese":  {"name": "Chinese (Simplified/Traditional)", "regex": r"[\u4E00-\u9FFF]",          "code": "zh"},
    "japanese": {"name": "Japanese",                         "regex": r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", "code": "ja"},
    "korean":   {"name": "Korean",                           "regex": r"[\uAC00-\uD7AF]",            "code": "ko"},
    "arabic":   {"name": "Arabic",                           "regex": r"[\u0600-\u06FF]",            "code": "ar"},
    "russian":  {"name": "Russian",                          "regex": r"[\u0400-\u04FF]",            "code": "ru"},
    "thai":     {"name": "Thai",                             "regex": r"[\u0E00-\u0E7F]",            "code": "th"},
}


def detect_text_language(text: str) -> dict:
    """Detect dominant language in text via unicode script ranges."""
    if not text:
        return {"language": "Unknown", "code": "unknown", "confidence": 0.0,
                "detected_scripts": [], "is_multilingual": False}

    sample = text[:3000]
    script_counts: dict[str, int] = {}
    for lang_key, info in _LANGUAGE_PATTERNS.items():
        n = len(re.findall(info["regex"], sample))
        if n > 0:
            script_counts[lang_key] = n

    ascii_letters = len(re.findall(r"[a-zA-Z]", sample))

    if not script_counts:
        return {"language": "English", "code": "en",
                "confidence": 1.0 if ascii_letters > 0 else 0.5,
                "detected_scripts": ["ASCII"], "is_multilingual": False}

    primary = max(script_counts, key=lambda k: script_counts[k])
    confidence = script_counts[primary] / (ascii_letters + sum(script_counts.values()))
    detected   = list(script_counts.keys())

    return {
        "language":         _LANGUAGE_PATTERNS[primary]["name"],
        "code":             _LANGUAGE_PATTERNS[primary]["code"],
        "confidence":       min(1.0, confidence),
        "detected_scripts": detected,
        "is_multilingual":  len(detected) > 1 or (ascii_letters > 0 and len(detected) > 0),
    }


def _detect_language_in_content(content: str, filename: str = "") -> dict:
    return detect_text_language(f"{filename} {content}")


# ---------------------------------------------------------------------------
# Taxonomy loader
# ---------------------------------------------------------------------------

def load_taxonomy(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {p}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML required: pip install pyyaml") from exc
        with p.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    if p.suffix.lower() == ".json":
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported taxonomy format: {p.suffix}")


def _get_default_taxonomy() -> dict:
    # Search order: project root first (user-visible), then core/ (bundled generic default)
    search_dirs = [Path(__file__).parent.parent, Path(__file__).parent]
    for directory in search_dirs:
        for name in ("taxonomy.yml", "taxonomy.yaml", "taxonomy.json"):
            candidate = directory / name
            if candidate.exists():
                return load_taxonomy(candidate)
    raise FileNotFoundError("No taxonomy file found in project root or core/.")


def _effective_taxonomy(custom: dict | None, fallback: dict | None = None) -> dict:
    if custom and all(custom.get(k) for k in ("domains", "scales", "lifecycle_stages")):
        return custom
    if fallback:
        return fallback
    return _get_default_taxonomy()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required keys in LLM output; missing ones trigger a retry
_REQUIRED_OUTPUT_KEYS = {
    "_reasoning", "domain", "scale", "information_type",
    "lifecycle", "governance", "confidentiality", "confidentiality_reason", "asset_type",
    "certainty", "certainty_score", "short_summary", "keywords",
    "year", "year_confidence",
}


# ---------------------------------------------------------------------------
# Pre-screen prompt
# ---------------------------------------------------------------------------

_PRESCREEN_PROMPT = """\
You are reviewing a single file's metadata and a content extract.
Answer three questions concisely and precisely.

⚠️  LANGUAGE NOTE: This file may contain non-English content.
    Detect and classify using domain knowledge, not language-specific heuristics.

FILE METADATA
  Filename          : {filename}
  Format            : {format}
  Folder path       : {folder_chain}
  Size              : {size_kb} KB
  Extraction method : {extraction_method}
  Year (confidence) : {year_info}
  Content language  : {content_language}

EXTRACTION NOTES
{extraction_notes}

CONTENT EXTRACT
{content_sample}

QUESTION 1 — Structured data presence
Does this file contain structured / tabular data (tables, CSV-like rows,
measurement records, GIS attributes, survey results, cost schedules, inventories)?

Answer: Likely | Possible | Unlikely
Reason: one sentence citing the strongest signal.

QUESTION 1B — Information value signal (not limited to tabular data)
Could this file still provide distinctive, reusable project information
(named entities, decisions, commitments, quantities, constraints, references,
or other facts) even if it is not structured/semi-structured data?

Answer: High | Medium | Low
Reason: one sentence citing the strongest signal.

Return ONLY this JSON (no markdown, no extra keys):
{{
  "data_likelihood": "Likely|Possible|Unlikely",
  "data_reason":     "<one sentence>",
  "info_value_signal": "High|Medium|Low",
  "info_value_reason": "<one sentence>",
  "detected_language": "{content_language}"
}}
"""


def _build_extraction_notes(meta: dict) -> str:
    """Human-readable extraction quality note for the LLM prompt."""
    lines = []
    cov    = str(meta.get("extraction_coverage", "")).strip()
    method = str(meta.get("extraction_method", "")).strip()
    yconf  = str(meta.get("year_confidence", "")).strip()
    rnote  = str(meta.get("routing_note", "")).strip()

    if cov:
        lines.append(f"  Coverage : {cov}")
    if method:
        method_label = {
            "text":      "native text extraction",
            "ocr":       "OCR (image recognition) — text accuracy may be lower",
            "text+ocr":  "mixed: native text on some pages, OCR on image-only pages",
            "converted": "converted from source format (e.g. DWG→PDF) then extracted",
            "metadata":  "no content extractable — filename and folder signals only",
        }.get(method, method)
        lines.append(f"  Method   : {method_label}")
    if yconf:
        yconf_label = {
            "high":    "high — year found in filename or repeated 3+ times in content",
            "medium":  "medium — year appears in content but only once or twice",
            "low":     "low — year inferred from file modification time (OS metadata only)",
            "unknown": "unknown — no year signal found",
        }.get(yconf, yconf)
        lines.append(f"  Year src : {yconf_label}")
    if rnote:
        lines.append(f"  Routing  : {rnote}")

    return "\n".join(lines) if lines else "  (no extraction notes)"


def _prescreen(
    meta: dict,
    folder_chain: str,
    client: _ClientT,
    model: str,
    api_timeout: int,
) -> dict:
    fallback = {
        "data_likelihood": "Unlikely",
        "data_reason":     "Pre-screen failed — using neutral default.",
        "info_value_signal": "Low",
        "info_value_reason": "Pre-screen failed — using neutral default.",
        "detected_language": "Unknown",
    }

    sample    = (meta.get("content_sample") or "")[:3000]
    lang_info = _detect_language_in_content(sample, meta.get("filename", ""))
    lang_display = f"{lang_info['code']} ({lang_info['language']})"
    if lang_info.get("is_multilingual"):
        lang_display += f" [mixed: {', '.join(lang_info.get('detected_scripts', []))}]"

    content_indented = (
        "\n".join("  " + ln for ln in sample.splitlines()) if sample else "  (not extractable)"
    )
    year_val  = meta.get("year") or "unknown"
    year_conf = meta.get("year_confidence", "unknown")

    prompt = _PRESCREEN_PROMPT.format(
        filename          = meta.get("filename", ""),
        format            = meta.get("format", ""),
        folder_chain      = folder_chain or "(root)",
        size_kb           = meta.get("size_kb", "?"),
        extraction_method = meta.get("extraction_method", "unknown"),
        year_info         = f"{year_val} ({year_conf} confidence)",
        content_language  = lang_display,
        extraction_notes  = _build_extraction_notes(meta),
        content_sample    = content_indented,
    )

    try:
        resp   = _llm_call(client, model,
                           system="You are a file classification assistant. Respond ONLY with valid JSON.",
                           user=prompt, temperature=0, api_timeout=api_timeout)
        result = _parse_json(resp.choices[0].message.content or "")
        for key in fallback:
            result.setdefault(key, fallback[key])
        ivs = str(result.get("info_value_signal", "")).strip().lower()
        if ivs in {"high", "strong"}:
            result["info_value_signal"] = "High"
        elif ivs in {"medium", "moderate", "mid"}:
            result["info_value_signal"] = "Medium"
        else:
            result["info_value_signal"] = "Low"
        result["detected_language"] = lang_display
        return result
    except Exception as exc:
        logger.warning("layer2 pre-screen failed for %s [%s: %s]",
                       meta.get("filename"), type(exc).__name__, exc)
        fallback["detected_language"] = lang_display
        return dict(fallback)


def _build_vlm_block(vlm_hint: dict | None) -> str:
    if not vlm_hint:
        return "  (not available for this file)"
    return "\n".join([
        f"  Visual file type   : {vlm_hint.get('visual_file_type', 'Unknown')}",
        f"  Visual discipline  : {vlm_hint.get('visual_discipline', 'Unknown')}",
        f"  Visual confidence  : {vlm_hint.get('visual_confidence', 'Low')}",
        f"  Visual note        : {vlm_hint.get('visual_note', '')}",
    ])


def _has_readable_text(meta: dict) -> bool:
    """True when layer1 produced meaningful readable text content."""
    sample = str(meta.get("content_sample", "") or "")
    density = len(re.sub(r"\s+", "", sample))
    method = str(meta.get("extraction_method", "")).strip().lower()
    if method in {"metadata", ""}:
        return False
    return density >= 80


def _pdf_first_page_data_url(file_path: Path) -> str | None:
    """Render first PDF page to a JPEG data URL for VLM calls (max 1600px long edge)."""
    try:
        import fitz  # type: ignore
        from PIL import Image as _PILImage
        import io as _io
        doc = fitz.open(str(file_path))
        try:
            if len(doc) < 1:
                return None
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            img = _PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
        max_side = 1600
        if max(img.width, img.height) > max_side:
            ratio = max_side / max(img.width, img.height)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _image_file_data_url(file_path: Path) -> str | None:
    """Encode an image file as a data URL for VLM calls (max 1600px long edge, JPEG)."""
    if file_path.suffix.lower().lstrip(".") not in {"jpg", "jpeg", "png", "webp"}:
        return None
    try:
        from PIL import Image as _PILImage
        import io as _io
        img = _PILImage.open(file_path).convert("RGB")
        max_side = 1600
        if max(img.width, img.height) > max_side:
            ratio = max_side / max(img.width, img.height)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _vlm_needs_visual_first(meta: dict, prescreen: dict | None = None) -> bool:
    """Decide whether to run a VLM visual pass.

    Agent gate (new): if prescreen confirms good readable text with high info
    value, skip VLM — the text path already provides sufficient signal.
    Otherwise fall back to format/extraction heuristics.

    Skip always when Layer 1 already performed a vision extraction — the
    content_sample already contains the visual description.
    """
    extraction_method = str(meta.get("extraction_method", "")).strip().lower()

    # L1 already ran vision — redundant to run VLM again
    if extraction_method == "vision":
        return False

    # Agent gate: prescreen confirms high information value and text is readable.
    # VLM would add nothing over what the text layer already provides.
    if (
        prescreen
        and _has_readable_text(meta)
        and extraction_method == "text"
        and prescreen.get("info_value_signal") == "High"
    ):
        return False

    # CAD formats — always visual-first (no readable text layer)
    if str(meta.get("format", "")).strip().lower() == "dwg":
        return True

    # Single-page drawing PDF detected by L1
    if str(meta.get("drawing_mode", "")).strip() == "single_page_drawing":
        return True

    # No readable text extracted at all
    if not _has_readable_text(meta):
        return True

    coverage = str(meta.get("extraction_coverage", "")).lower()
    return "no readable content" in coverage or "extraction failed" in coverage


def _dwg_preview_data_url(file_path: Path) -> str | None:
    """Convert DWG to temporary PDF and render first page for VLM."""
    lo = shutil.which("soffice") or shutil.which("libreoffice")
    if not lo:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            result = subprocess.run(
                [
                    lo,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(out_dir),
                    str(file_path),
                ],
                capture_output=True,
                timeout=45,
            )
            if result.returncode != 0:
                return None
            pdf_path = out_dir / f"{file_path.stem}.pdf"
            if not pdf_path.exists():
                return None
            return _pdf_first_page_data_url(pdf_path)
    except Exception:
        return None


def _vlm_hint_visual_first(
    meta: dict,
    client: _ClientT,
    model: str,
    api_timeout: int,
    prescreen: dict | None = None,
) -> dict:
    """Use a multimodal pass when visual analysis is needed.

    Priority:
      1) single-page drawing mode
      2) no readable text / metadata-only extraction

    Falls back silently when model/provider does not support images.
    Passes prescreen to _vlm_needs_visual_first so the agent gate can skip
    the VLM call for text-rich, high-value documents.
    """
    if not _vlm_needs_visual_first(meta, prescreen=prescreen):
        return {}

    file_path_raw = str(meta.get("file_path", "")).strip()
    if not file_path_raw:
        return {}
    file_path = Path(file_path_raw)
    fmt = str(meta.get("format", "")).strip().lower()

    if fmt == "pdf":
        data_url = _pdf_first_page_data_url(file_path)
    elif fmt == "dwg":
        data_url = _dwg_preview_data_url(file_path)
    else:
        data_url = _image_file_data_url(file_path)

    if not data_url:
        return {}

    prompt = (
        "You are reviewing one AEC visual file (drawing/render/photo/screenshot). "
        "Return only JSON with keys: visual_file_type, visual_discipline, "
        "visual_confidence, visual_note.\n"
        "visual_file_type should be specific when possible: Drawing, Rendering, "
        "Photo, Map, Table/Chart, Diagram, Document Scan, Unknown.\n"
        "visual_confidence must be High/Medium/Low.\n"
        f"Filename: {meta.get('filename', '')}\n"
        f"Format: {meta.get('format', '')}\n"
        f"Extraction method: {meta.get('extraction_method', '')}\n"
        f"Folder: {' / '.join(meta.get('path_segments') or [])}\n"
        "Keep visual_note to one short sentence describing the key visual evidence."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise AEC visual classifier. Respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0,
            timeout=int(api_timeout),
        )
        result = _parse_json(resp.choices[0].message.content or "")
        result.setdefault("visual_file_type", "Unknown")
        result.setdefault("visual_discipline", "Unknown")
        result.setdefault("visual_confidence", "Low")
        result.setdefault("visual_note", "")
        return result
    except Exception as exc:
        logger.warning(
            "VLM vision call failed for %s (model may not support vision or provider error): %s",
            meta.get("filename"), type(exc).__name__,
        )
        logger.debug("VLM hint unavailable for %s: %s", meta.get("filename"), exc)
        return {}


# ---------------------------------------------------------------------------
# Main classification prompt
# ---------------------------------------------------------------------------

def _taxonomy_block(items: list[dict]) -> str:
    if not items:
        return "  (none configured)"
    width = max(len(d.get("name", "")) for d in items)
    return "\n".join(
        f"  {d.get('name', ''):{width}}  — {d.get('description', '')}"
        for d in items
    )


# Fallback disambiguation cues, used only when the active taxonomy defines no
# per-domain `cues` fields. Project taxonomies should carry their own cues so
# custom domains (Heritage, Religion, Timber Engineering, …) get equal
# treatment — see _build_domain_hints_block.
_LEGACY_DOMAIN_HINTS = """\
    Structural Engineering:
        foundation, footing, pile, slab, beam, column, rebar, steel frame, load, structural calc.
    MEP - HVAC:
        HVAC, AHU, FCU, duct, diffuser, ventilation, cooling load, heating load, psychrometric.
    MEP - Plumbing:
        plumbing, sanitary, drainage, stormwater, wastewater, pipe schedule, manhole, invert level.
    MEP - Electrical:
        single line diagram, SLD, panel, breaker, cable tray, lighting circuit, ELV, earthing.
    Civil & Infrastructure:
        road, pavement, bridge, tunnel, utility network, sewer line, grading, profile/section chainage.
    QS & Commercial:
        BOQ, quantity takeoff, estimate, valuation, payment certificate, variation order, commercial claim.
    Administrative & Legal:
        contract clauses, legal notice, permit letter, statutory approval correspondence.
    Project Management:
        schedule, programme, RFI log, meeting minutes, transmittal, tracker."""


def _build_domain_hints_block(taxonomy: dict) -> str:
    """Build the disambiguation-hints block from per-domain `cues` in the taxonomy.

    Falls back to the legacy hardcoded hints when the taxonomy defines no cues,
    so older taxonomy files keep working unchanged.
    """
    lines = []
    for d in (taxonomy or {}).get("domains", []):
        cues = str(d.get("cues", "") or "").strip()
        name = str(d.get("name", "") or "").strip()
        if cues and name:
            lines.append(f"    {name}:\n        {cues}")
    return "\n".join(lines) if lines else _LEGACY_DOMAIN_HINTS


_PROMPT_TEMPLATE = """\
You are a senior AEC data archivist. Classify the file described below.
Use your domain knowledge; do not ask for more information.

⚠️  LANGUAGE NOTE ⚠️
This file may contain non-English content ({content_language}).
Classify using universal concepts, not language-specific rules.

═══ PROJECT CONTEXT ═══════════════════════════════════════════════════════════
{project_context}

═══ FILE ═══════════════════════════════════════════════════════════════════════
  Filename  : {filename}
  Format    : {format}  |  Size: {size_kb} KB ({size_category})
  Pages     : {page_count}
  Year      : {year_info}

═══ FOLDER PATH  (root → parent) ══════════════════════════════════════════════
  {folder_chain}

═══ FILENAME STRUCTURAL TOKENS ════════════════════════════════════════════════
{filename_signals_block}

═══ EXTRACTION QUALITY ════════════════════════════════════════════════════════
{extraction_notes}

  Weight the content sample accordingly:
  - "native text extraction" → high reliability
  - "OCR" or "text+ocr"     → moderate; trust structure over individual words
  - "converted" (DWG→PDF)   → treat like native text but note CAD origin
  - "metadata only"         → no content available; rely on filename + folder

═══ PRE-SCREEN  (focused LLM pass on raw content) ═════════════════════════════
  Data likelihood  : {data_likelihood}  — {data_reason}
  Info value signal: {info_value_signal}  — {info_value_reason}
  Language context : {detected_language}

  These come from direct content analysis.
  Override only if folder path or filename clearly contradicts them.

═══ FOLDER SIBLINGS  (other files already classified in this directory) ════════
{folder_context_block}

  Calibration hint: when 3+ siblings share a domain or lifecycle, treat that
  as strong prior evidence for this file too.  Override only when filename or
  content clearly contradicts the consensus.

{layer2_rules_section}═══ VISUAL HINT  (VLM pass for drawing-like files) ═════════════════════════════
{vlm_block}

═══ CONTENT SAMPLE  ({information_type}) ══════════════════════════════════════
{content_sample}

═══ CLASSIFICATION TASK ════════════════════════════════════════════════════════
Write a 2–4 sentence _reasoning chain first, then fill every field.
Signal priority: folder path > filename tokens > pre-screen > content sample.
Drawing mode: {drawing_mode_note}

DOMAIN DISAMBIGUATION HINTS (use strongest matching cue set):
{domain_hints_block}

SCALE MAPPING HINTS:
    Component / Room: detail callouts, room IDs, interior fit-out packages, component schedules.
    Floor / Level: one-level plans (e.g., L01/L02/B1), floor coordination packages.
    Plot / Block: parcel/red-line scope, specific block/lot development boundary.
    Neighborhood / District: precinct or district packages across multiple plots.
    City / Municipal: citywide strategies, municipal plans.
    Metropolitan: cross-city metro corridors/region-wide systems.

LIFECYCLE MAPPING HINTS:
    Competition: competition boards/reports, jury submissions, RFP responses, bid-stage design.
    Concept / Schematic: visioning, feasibility, massing studies, concept and SD packages.
    Design Development: discipline-coordinated developed design and technical refinement.
    Construction Documents: permit sets, IFC/CD packages, final specs, tender/procurement docs.
    Construction Administration: site diary/logs, method statements, RFIs, submittals, VO/RVO/change notices.
    As-Built / Handover: as-built drawings, completion certificates, O&M manuals, FM handover records.

UNKNOWN USAGE RULE:
    Use Unknown only when available evidence is genuinely insufficient or contradictory.
    If at least one strong cue exists, choose the best-fit non-Unknown label.
    Prefer specific domains (Structural/MEP/Civil/QS) over broad buckets.

MULTI-DOMAIN RULE:
        Some files can span multiple domains.
        - Put the BEST single primary domain in "domain".
        - Also return up to 3 ranked candidates in "domain_candidates"
            (primary first, then secondary domains if relevant).
        - If only one clear domain exists, return just ["<primary>"] for domain_candidates.

DOMAIN options:
{domain_options}

SCALE options:
{scale_options}

LIFECYCLE options:
{lifecycle_options}

CONFIDENTIALITY RULE — mark Confidential ONLY for:
  1. Contracts (service agreement, engagement letter, binding legal agreement)
  2. Financial records (invoice, payroll, salary, bank statement, financial statement/report)
  3. Email communications (.eml/.msg file or email chain with From/To/Subject headers)
  4. Files explicitly labelled "Confidential", "Strictly Confidential", or "Private and Confidential" — a label anywhere in the FILENAME or content counts, and this rule OVERRIDES the exclusions below.
  DO NOT mark Confidential for drawings, reports, specs, budgets, cost plans, tenders, or meeting minutes — UNLESS rule 4 applies (an explicit confidentiality label is present in the filename or content).

CONFIDENTIALITY options:
{confidentiality_options}

ASSET TYPE options:
{asset_type_options}

  A single file can serve multiple asset-type roles (e.g. a Drawing that is also
  a Model, or a Report that is also Data). Put the BEST single primary type in
  "asset_type" and list up to 3 ranked candidates (primary first) in
  "asset_type_candidates". If only one type applies, repeat it: ["<primary>"].

INFORMATION TYPE — pick one:
  Schematic / Technical  |  Quantitative / Tabular  |  Narrative / Textual
  Spatial / Cartographic  |  Visual / Media  |  Archive  |  Unknown

GOVERNANCE options:
{governance_options}

YEAR  — You are the sole authority for year determination. Ignore the Layer 1 hint
        below if content gives better evidence. Evaluate in this order:
          1. Explicit dated field in content: "Issued:", "Date:", "Rev. Date:",
             "Prepared:", title-block date, stamp, header/footer date → high
          2. Natural-language date in content: "March 2024", "Q1 2022",
             "Winter 2023", "15.03.2022" → high or medium depending on clarity
          3. Year appearing 3+ times in content (excluding AutoCAD version strings,
             copyright notices, drawing codes) → high
          4. Year in filename or folder (Layer 1 hint) → medium
          5. Single year mention in content → medium
          6. No credible signal → null
        Output integer YYYY or null.
        Also output "year_confidence": "high" | "medium" | "low" | "unknown"
          high    — explicit dated field, or 3+ consistent content occurrences
          medium  — filename/folder hint, or 1–2 content occurrences
          low     — single weak/indirect mention
          unknown — no signal found

CERTAINTY — your assessment of how reliable this classification is:
  High   — 3+ consistent signals, no contradictions
  Medium — 1–2 clear signals, rest generic or mildly inconsistent
  Low    — generic filename AND generic folder, or signals conflict

CERTAINTY_SCORE — integer 0–100 matching your certainty assessment.
  High = 75–100, Medium = 45–74, Low = 0–44.
  Deduct for: unknown domain/lifecycle, metadata-only extraction,
  mtime-only year, contradicting signals.

SHORT_SUMMARY — 2 sentences, 30–45 words. STRUCTURE: sentence 1 = what it contains
  (specific facts, values, codes, visible elements); sentence 2 = what design decision
  or question this file helps answer or supports.

  ── FORBIDDEN PHRASES (auto-fail if any appear) ───────────────────────────────
  NEVER use these words or phrases anywhere in the summary:
  "This file", "This document", "This drawing", "contains information about",
  "related to", "pertaining to", "regarding", "is a document that",
  "provides information on", "showcases", "illustrates", "depicts",
  "reflects modern", "reflects the", "captures a", "part of the masterplan",
  "part of the project", "likely contains", "serves as visual reference",
  "supports design decisions", "supports planning decisions",
  "supports decisions on layout", "informs design decisions",
  "within the masterplan", "within the project", "within the overall",
  "design elements", "spatial relationships", "design coordination",
  "visual coordination", "project development".
  These are all empty filler. Sentence 2 MUST name a SPECIFIC decision type
  (e.g. "foundation strategy", "shading detail", "MEP routing", "facade system",
  "path hierarchy", "golf course grading") — not a generic category.

  Every summary must contain at least ONE of:
    • a named entity (person, firm, place, zone/block name, feature name)
    • a specific quantity or measurement (depth, load, area, scale, count)
    • a drawing/document code or reference number
    • a named decision, constraint, or conclusion
    • a date or version identifier

  ── WHEN CONTENT IS READABLE ──────────────────────────────────────────────────
  Sentence 1: State the most specific fact, finding, measurement, or named element
  visible in the content. Quote numbers, codes, or named entities directly.
  Sentence 2: Name the SPECIFIC decision type this file informs — e.g.
  "Informs grading strategy for the golf course fairways" not "supports design".

  ── WHEN EXTRACTION IS METADATA-ONLY (documents/drawings) ────────────────────
  Begin sentence 1 with: "Based on filename analysis: ..."
  Decode the filename tokens — expand codes, name the discipline, drawing type,
  and date token if present. Do NOT write "likely contains".
  Sentence 2: Name the specific decision or coordination task this series handles.

  ── FOR IMAGES (Media asset_type) ────────────────────────────────────────────
  Sentence 1: Describe the most specific visible element: space type, material,
  structure, scale cue, or notable detail (e.g. "column-free hall, ~30m span,
  top-lit clerestory, exposed concrete walls").
  Sentence 2: Name the precise design question it answers — e.g.
  "Precedent for long-span roof structure at the main cultural building" or
  "Reference for pedestrian shading detail along the site promenade."
  NEVER end with "for the project" or "in landscape design" alone.

  ── SELF-CHECK before writing ─────────────────────────────────────────────────
  Ask: "Could sentence 2 apply to 50 other files?" If yes, it is too generic.
  Add the specific system, zone, or decision name that makes it unique.

  ✓ EXCELLENT (content readable):
  • "Geotechnical report: bearing capacity 200 kPa at 3 m depth, Zone A; raft
    foundation recommended for Block 3, April 2023. Informs foundation type
    selection and structural grid spacing for the eastern residential parcels."
  • "Minutes record approval of 4.2 m floor-to-floor height and deferral of
    facade system to DD stage, March 2024. Locks structural slab depth and
    triggers facade procurement scope revision."

  ✓ EXCELLENT (image):
  • "Dense urban park: ~80% canopy cover, linear stone-paved paths, central
    reflecting pool. Precedent for canopy density targets and path material
    selection in the project landscape zone."
  • "Column-free exhibition hall, top-lit clerestory, exposed concrete, ~30 m
    span. Reference for a long-span roof structural strategy."

  ✓ EXCELLENT (metadata only):
  • "Based on filename analysis: UT-400-016, utility corridor drawing for
    irrigation and electrical systems, Construction Documents phase, 2022.
    Informs site-wide MEP routing and trench-coordination for Phase 1."

  ✗ BAD — auto-rejected:
  • "An image showcasing modern landscape design principles for an urban
    public-realm project."
  • "Supports design decisions regarding spatial relationships within the masterplan."
  • "Based on filename analysis: landscape drawing from the design phase." ← still generic
  • "Useful as a precedent for architectural design decisions in the project." ← forbidden

KEYWORDS — 5–8 lowercase terms for archive search.
  Rules: ≥3 keywords must appear verbatim in short_summary.
  Prefer specific codes, materials, system names, and named parties over
  generic category words like "design", "document", "report".

Return ONLY this JSON object (no markdown fences, no extra commentary):
{{
  "_reasoning":       "<2–4 sentence reasoning chain>",
  "domain":           "<value>",
    "domain_candidates": ["<primary>", "<secondary?>", "<secondary?>"],
  "scale":            "<value>",
  "information_type": "<value>",
  "lifecycle":        "<value>",
  "governance":             "<value>",
  "confidentiality":        "<value>",
  "confidentiality_reason": "<one sentence citing the specific signal that determined this>",
  "asset_type":            "<primary value>",
  "asset_type_candidates": ["<primary>", "<secondary?>"],
  "year":             <YYYY or null>,
  "year_confidence":  "<high|medium|low|unknown>",
  "certainty":        "<High|Medium|Low>",
  "certainty_score":  <integer 0–100>,
    "short_summary":    "<1–2 sentences, ~30 words>",
  "keywords":         ["<term1>", "<term2>", ...]
}}
"""


# ---------------------------------------------------------------------------
# Summary critique prompt (second-order evaluation)
# ---------------------------------------------------------------------------

_SUMMARY_CRITIQUE_PROMPT = """\
You are an expert AEC content reviewer. Evaluate the quality of a short_summary.

ORIGINAL SUMMARY:
  "{short_summary}"

CONTEXT (for reference):
  Domain: {domain}
  Asset Type: {asset_type}
  File: {filename}

EVALUATION CRITERIA:
  1. Specificity (0–100): Does it cite specific facts, codes, entities, or measurements?
                          0 = all generic, 100 = highly specific
  2. Actionability (0–100): Does it name a SPECIFIC decision/constraint/finding?
                            0 = vague, 100 = clear and specific

QUALITY ASSESSMENT:
  Return JSON with:
    "quality_score": <average of specificity + actionability, 0–100>,
    "specificity": <0–100>,
    "actionability": <0–100>,
    "is_good": <true if quality_score ≥ 70>,
    "issues": [<list of specific issues found>],
    "suggestion": <improved version (keep original meaning, add specifics) OR null>

If score < 70:
  Provide a suggestion that:
  - Keeps the original meaning and sentence structure
  - Adds at least one specific fact/code/named entity/measurement
  - Names a SPECIFIC decision type (not generic "design decisions")
  - Avoids generic phrases like "supports design", "related to", "provides information about"

If score ≥ 70:
  Set suggestion to null.
"""


# ---------------------------------------------------------------------------
# Summary critique (second-order evaluation)
# ---------------------------------------------------------------------------

def _critique_summary(
    short_summary: str,
    filename: str,
    domain: str,
    asset_type: str,
    client: _ClientT,
    model: str,
    api_timeout: int = 30,
) -> dict:
    """
    Second-order evaluation: LLM critiques its own short_summary.

    Returns:
      {
        "quality_score": 0–100,
        "specificity": 0–100,
        "actionability": 0–100,
        "is_good": boolean,
        "issues": [list],
        "suggestion": str or None,
        "improved_summary": str  # original or improved, based on quality
      }
    """
    if not short_summary or not short_summary.strip():
        return {
            "quality_score": 0,
            "specificity": 0,
            "actionability": 0,
            "is_good": False,
            "issues": ["Empty summary"],
            "suggestion": None,
            "improved_summary": short_summary,
        }

    prompt = _SUMMARY_CRITIQUE_PROMPT.format(
        short_summary=short_summary,
        filename=filename,
        domain=domain,
        asset_type=asset_type,
    )

    try:
        resp = _llm_call(
            client, model,
            system="You are a content quality reviewer. Respond ONLY with valid JSON.",
            user=prompt,
            temperature=0,
            api_timeout=api_timeout,
        )
        critique = _parse_json(resp.choices[0].message.content or "")

        # Ensure required fields
        quality_score = int(critique.get("quality_score", 50))
        specificity = int(critique.get("specificity", 50))
        actionability = int(critique.get("actionability", 50))
        is_good = critique.get("is_good", quality_score >= 70)
        issues = critique.get("issues", [])
        suggestion = critique.get("suggestion")

        # Decide: use suggestion if quality is low and suggestion exists
        improved = suggestion if (not is_good and suggestion) else short_summary

        return {
            "quality_score": min(100, max(0, quality_score)),
            "specificity": min(100, max(0, specificity)),
            "actionability": min(100, max(0, actionability)),
            "is_good": is_good,
            "issues": issues if isinstance(issues, list) else [str(issues)],
            "suggestion": suggestion,
            "improved_summary": improved,
        }

    except Exception as exc:
        logger.warning(
            "summary critique failed for %s: %s",
            filename, exc
        )
        # Graceful fallback: return original summary with no critique
        return {
            "quality_score": 50,
            "specificity": 50,
            "actionability": 50,
            "is_good": False,
            "issues": [f"Critique failed: {str(exc)}"],
            "suggestion": None,
            "improved_summary": short_summary,
        }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_folder_context_block(folder_context: list[dict] | None) -> str:
    """Format already-classified siblings as a compact calibration block."""
    if not folder_context:
        return "  (no siblings classified yet in this folder)"
    width = 42
    lines = []
    for fc in folder_context[-6:]:   # show up to 6 most recent
        fn   = str(fc.get("filename", "?"))[:width]
        dom  = fc.get("domain",    "?")
        lc   = fc.get("lifecycle", "?")
        at   = fc.get("asset_type","?")
        cert = fc.get("certainty", "?")
        lines.append(f"  {fn:<{width}}  {dom} | {lc} | {at} | cert={cert}")
    return "\n".join(lines)


def _build_folder_chain(meta: dict, input_path: Path) -> str:
    segments = meta.get("path_segments")
    if segments:
        return " / ".join(segments)
    try:
        rel   = Path(meta["file_path"]).parent.relative_to(input_path)
        parts = [p for p in rel.parts if p not in (".", "")]
        return " / ".join(parts) if parts else meta.get("folder", "")
    except (ValueError, KeyError):
        return meta.get("folder", "")


def _build_filename_signals_block(meta: dict) -> str:
    signals = meta.get("filename_signals") or {}
    lines   = []
    if signals.get("raw_stem"):
        lines.append(f"  Full stem        : {signals['raw_stem']}")
    if signals.get("code_tokens"):
        lines.append(f"  Code tokens      : {' | '.join(signals['code_tokens'])}")
    if signals.get("drawing_number"):
        lines.append(f"  Drawing ref      : {signals['drawing_number']}")
    if signals.get("version"):
        lines.append(f"  Version/revision : {signals['version']}")
    if signals.get("has_date_in_name"):
        lines.append("  Date in name     : yes")
    if signals.get("numeric_tokens"):
        lines.append(f"  Numeric tokens   : {', '.join(signals['numeric_tokens'][:6])}")
    return "\n".join(lines) if lines else "  (none detected)"


def _build_year_info(meta: dict) -> str:
    """Build the year hint passed to Layer 2. This is regex-extracted only — LLM overrides."""
    year      = meta.get("year")
    year_conf = str(meta.get("year_confidence", "unknown")).strip()
    if not year:
        return "no regex signal found — determine from content"
    label = {
        "high":    "regex hint — from filename or repeated in content (you may override)",
        "medium":  "regex hint — found in content (you may override)",
        "unknown": "regex hint — source unclear",
    }.get(year_conf, year_conf)
    return f"{year}  ({label})"


def _build_layer2_rules_hint(file_path: Path, rules_config: dict | None = None) -> str:
    """Generate prompt hints from folder/filename rules.

    Returns a formatted text block of hints to be included in the prompt.
    Rules act as suggestions, not hard constraints — LLM can override based on content.
    """
    if not rules_config:
        return "(no layer2 rules configured)"

    hints = []
    file_path_str = str(file_path)
    filename = file_path.name

    # Check folder patterns
    for folder_rule in rules_config.get("folder_rules") or []:
        pattern = folder_rule.get("pattern", "").strip()
        if not pattern:
            continue
        # Match against full path and parent directory names
        if fnmatch.fnmatch(file_path_str, f"*{pattern}*"):
            if domain := folder_rule.get("domain_hint", "").strip():
                hints.append(f"  Folder path suggests domain → {domain}")
            if gov := folder_rule.get("governance_hint", "").strip():
                hints.append(f"  Folder path suggests governance → {gov}")

    # Check filename patterns
    for filename_rule in rules_config.get("filename_rules") or []:
        pattern = filename_rule.get("pattern", "").strip()
        if not pattern:
            continue
        try:
            if re.match(pattern, filename, re.IGNORECASE):
                if asset := filename_rule.get("asset_type_hint", "").strip():
                    hints.append(f"  Filename pattern suggests asset_type → {asset}")
                if lifecycle := filename_rule.get("lifecycle_hint", "").strip():
                    hints.append(f"  Filename pattern suggests lifecycle → {lifecycle}")
        except re.error:
            logger.debug("Invalid regex pattern in layer2_rules: %s", pattern)
            continue

    # Add project phase guidance if present
    project_phase = rules_config.get("project_phase") or {}
    if current_stage := project_phase.get("current_stage", "").strip():
        hints.append(f"  Project context: currently in '{current_stage}' phase")
        if guidance := project_phase.get("priority_guidance", "").strip():
            for line in guidance.split("\n"):
                if line.strip():
                    hints.append(f"    {line.strip()}")

    return "\n".join(hints) if hints else "  (no matching rules)"


def _build_layer2_rules_section(file_path: Path, rules_config: dict | None) -> str:
    """Return the full LAYER2 RULES HINTS section, or empty string if no rules configured."""
    if not rules_config:
        return ""
    hints = _build_layer2_rules_hint(file_path, rules_config)
    if not hints.strip() or hints.strip() == "(no matching rules)":
        return ""
    return (
        "═══ LAYER2 RULES HINTS  (human-defined patterns from project config) ═══════════\n"
        f"{hints}\n\n"
    )


def _build_prompt(
    meta: dict,
    input_path: Path,
    project_context: str,
    taxonomy: dict,
    folder_chain: str,
    prescreen: dict,
    vlm_hint: dict | None = None,
    rules_config: dict | None = None,
    folder_context: list[dict] | None = None,
) -> str:
    sample           = (meta.get("content_sample") or "")[:12000]
    content_indented = (
        "\n".join("  " + ln for ln in sample.splitlines()) if sample else "  (not extractable)"
    )
    detected_lang = prescreen.get("detected_language", "Unknown")
    file_path = Path(meta.get("file_path", "")) if meta.get("file_path") else Path("unknown")

    return _PROMPT_TEMPLATE.format(
        project_context         = project_context.strip() or "(no project context provided)",
        filename                = meta.get("filename", ""),
        format                  = meta.get("format", ""),
        size_kb                 = meta.get("size_kb", "?"),
        size_category           = meta.get("size_category", "unknown"),
        page_count              = meta.get("page_count") or "n/a",
        year_info               = _build_year_info(meta),
        folder_chain            = folder_chain or "(root — no folder path)",
        filename_signals_block  = _build_filename_signals_block(meta),
        extraction_notes        = _build_extraction_notes(meta),
        data_likelihood         = prescreen.get("data_likelihood", "Unknown"),
        data_reason             = prescreen.get("data_reason", ""),
        info_value_signal       = prescreen.get("info_value_signal", "Low"),
        info_value_reason       = prescreen.get("info_value_reason", ""),
        detected_language       = detected_lang,
        content_language        = detected_lang,
        folder_context_block    = _build_folder_context_block(folder_context),
        layer2_rules_section    = _build_layer2_rules_section(file_path, rules_config),
        vlm_block               = _build_vlm_block(vlm_hint),
        drawing_mode_note       = (
            "single-page drawing mode is active; text signal may be weak — do not use High certainty"
            if str(meta.get("drawing_mode", "")).strip() == "single_page_drawing"
            else (
                "visual-first mode is active (no readable text); prioritize VLM evidence over folder-only assumptions"
                if not _has_readable_text(meta)
                else "standard mode"
            )
        ),
        information_type        = meta.get("format_type", meta.get("information_type", "Unknown")),
        content_sample          = content_indented,
        domain_hints_block      = _build_domain_hints_block(taxonomy),
        domain_options          = _taxonomy_block(taxonomy.get("domains", [])),
        scale_options           = _taxonomy_block(taxonomy.get("scales", [])),
        lifecycle_options       = _taxonomy_block(taxonomy.get("lifecycle_stages", [])),
        confidentiality_options = _taxonomy_block(taxonomy.get("confidentiality_levels", [])),
        asset_type_options      = _taxonomy_block(taxonomy.get("asset_types", [])),
        governance_options      = _taxonomy_block(taxonomy.get("governance_sources", [])),
    )


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _llm_call(
    client: _ClientT,
    model: str,
    system: str,
    user: str,
    temperature: float,
    api_timeout: int,
) -> Any:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    kwargs: dict = dict(model=model, messages=messages,
                        temperature=temperature, timeout=int(api_timeout))
    try:
        return client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except Exception as e:
        if any(kw in str(e).lower() for kw in ("response_format", "json_object", "unsupported")):
            logger.debug("Provider does not support response_format=json_object; retrying without.")
            return client.chat.completions.create(**kwargs)
        raise


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    # Strip trailing commas before closing braces/brackets (common LLM output error)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        parsed = json.loads(cleaned)
        # If LLM returned an array, take the first element
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fallback: find first { ... } block
    lines = cleaned.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("{")), 0)
    cleaned = "\n".join(lines[start:])
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed:
            return parsed[0] if isinstance(parsed[0], dict) else {}
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}  raw={raw[:200]!r}") from exc


def _validate_output(result: dict) -> list[str]:
    return [k for k in _REQUIRED_OUTPUT_KEYS if k not in result]


def _coerce_domain_candidates(raw: Any, primary: str, taxonomy: dict | None = None) -> list[str]:
    """Normalize multi-domain candidates to known taxonomy labels.

    Keeps primary first, removes duplicates, caps to top 3.
    """
    primary_clean = str(primary or "").strip() or "Unknown"
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        items = [s.strip() for s in re.split(r"[,;/|]", raw) if s.strip()]
    else:
        items = []

    tx_domains = (taxonomy or {}).get("domains", []) if isinstance(taxonomy, dict) else []
    allowed = [str(d.get("name", "")).strip() for d in tx_domains if str(d.get("name", "")).strip()]
    allowed_map = {name.lower(): name for name in allowed}

    normalized: list[str] = []
    for name in items:
        if not name:
            continue
        mapped = allowed_map.get(name.lower(), name)
        if mapped not in normalized:
            normalized.append(mapped)

    primary_mapped = allowed_map.get(primary_clean.lower(), primary_clean)
    if not normalized:
        normalized = [primary_mapped]
    elif primary_mapped in normalized:
        normalized.remove(primary_mapped)
        normalized.insert(0, primary_mapped)
    else:
        normalized.insert(0, primary_mapped)

    # Keep only taxonomy-listed values when taxonomy is available.
    if allowed:
        normalized = [d for d in normalized if d in allowed]
        if not normalized:
            normalized = [primary_mapped if primary_mapped in allowed else "Unknown"]

    # Prefer concrete domains over Unknown when alternatives exist.
    if len(normalized) > 1:
        normalized = [d for d in normalized if d != "Unknown"] or ["Unknown"]

    return normalized[:3]


def _coerce_asset_type_candidates(raw: Any, primary: str, taxonomy: dict | None = None) -> list[str]:
    """Normalize multi-asset-type candidates to known taxonomy labels.

    Keeps primary first, removes duplicates, caps to top 3.
    """
    primary_clean = str(primary or "").strip() or "Unknown"
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        items = [s.strip() for s in re.split(r"[,;/|]", raw) if s.strip()]
    else:
        items = []

    tx_types = (taxonomy or {}).get("asset_types", []) if isinstance(taxonomy, dict) else []
    allowed  = [str(t.get("name", "")).strip() for t in tx_types if str(t.get("name", "")).strip()]
    allowed_map = {name.lower(): name for name in allowed}

    normalized: list[str] = []
    for name in items:
        if not name:
            continue
        mapped = allowed_map.get(name.lower(), name)
        if mapped not in normalized:
            normalized.append(mapped)

    primary_mapped = allowed_map.get(primary_clean.lower(), primary_clean)
    if not normalized:
        normalized = [primary_mapped]
    elif primary_mapped in normalized:
        normalized.remove(primary_mapped)
        normalized.insert(0, primary_mapped)
    else:
        normalized.insert(0, primary_mapped)

    if allowed:
        normalized = [t for t in normalized if t in allowed]
        if not normalized:
            normalized = [primary_mapped if primary_mapped in allowed else "Unknown"]

    if len(normalized) > 1:
        normalized = [t for t in normalized if t != "Unknown"] or ["Unknown"]

    return normalized[:3]


def _canon_key(value: str) -> str:
    """Punctuation/spacing-insensitive comparison key ("As-Built / Handover" -> "as built handover")."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


# Common phase-name variants -> canon key of the six-stage vocabulary.
_LIFECYCLE_ALIASES = {
    "bid": "competition", "competition entry": "competition",
    "concept": "concept schematic", "concept design": "concept schematic",
    "conceptual design": "concept schematic", "schematic": "concept schematic",
    "schematic design": "concept schematic", "sd": "concept schematic",
    "feasibility": "concept schematic", "masterplan": "concept schematic",
    "dd": "design development", "detailed design": "design development",
    "developed design": "design development",
    "cd": "construction documents", "construction documentation": "construction documents",
    "working drawings": "construction documents", "permit set": "construction documents",
    "tender": "construction documents", "technical design": "construction documents",
    "ca": "construction administration", "construction": "construction administration",
    "construction phase": "construction administration", "site support": "construction administration",
    "as built": "as built handover", "handover": "as built handover",
    "closeout": "as built handover", "operation and maintenance": "as built handover",
}


def _snap_to_vocab(value: str, taxonomy: dict | None, key: str,
                   aliases: dict[str, str] | None = None) -> str:
    """Snap a single-valued LLM label onto the taxonomy vocabulary for `key`.

    Exact names pass through; matching ignores case, punctuation and spacing;
    optional aliases map known variants. Anything unmatched becomes Unknown so
    off-vocabulary labels never reach the CSV (seen in production: lifecycle
    "Existing conditions assessment", "Planning/Discussion").
    """
    entries = (taxonomy or {}).get(key, []) if isinstance(taxonomy, dict) else []
    allowed = {_canon_key(e.get("name", "")): str(e.get("name", "")).strip()
               for e in entries if str(e.get("name", "")).strip()}
    if not allowed:
        return str(value or "").strip() or "Unknown"
    ck = _canon_key(value)
    if aliases:
        ck = aliases.get(ck, ck)
    return allowed.get(ck, "Unknown")


def _map_visual_discipline_to_domain(visual_discipline: str, taxonomy: dict | None = None) -> str:
    """Map VLM visual discipline text to a taxonomy domain label."""
    vd = str(visual_discipline or "").strip().lower()
    if not vd:
        return "Unknown"

    # Canonical mapping from common VLM outputs to taxonomy names.
    mapping = [
        ("Architecture & Buildings", ["architect", "building", "interior", "facade"]),
        ("Landscape & Public Realm", ["landscape", "public realm", "streetscape"]),
        ("Structural Engineering", ["structur", "foundation", "rebar", "beam", "column"]),
        ("MEP - HVAC", ["hvac", "duct", "air conditioning", "ventilation", "ahu", "fcu"]),
        ("MEP - Plumbing", ["plumb", "drain", "sanitary", "stormwater", "wastewater", "pipe"]),
        ("MEP - Electrical", ["electr", "lighting", "single line", "sld", "panel", "cable"]),
        ("Civil & Infrastructure", ["civil", "infrastructure", "road", "bridge", "grading", "utility"]),
        ("Mobility & Transport", ["transport", "mobility", "traffic", "transit", "parking"]),
        ("Environment & Climate", ["environment", "climate", "ecology", "sustainab", "hydrology"]),
        ("QS & Commercial", ["qs", "quantity", "boq", "cost", "commercial", "estimate"]),
        ("Administrative & Legal", ["legal", "contract", "permit", "approval", "statutory"]),
        ("Project Management", ["project management", "rfi", "meeting", "transmittal", "schedule"]),
        ("Reference & Research", ["reference", "research", "benchmark", "standard"]),
    ]

    for target, needles in mapping:
        if any(n in vd for n in needles):
            return target

    # Taxonomy-aware passthrough if the model already returned a close label.
    tx_domains = (taxonomy or {}).get("domains", []) if isinstance(taxonomy, dict) else []
    for d in tx_domains:
        name = str(d.get("name", "")).strip()
        if name and name.lower() == vd:
            return name

    return "Unknown"


def _map_visual_type_to_asset_type(visual_file_type: str) -> str:
    """Map VLM file-type hints to asset_type labels used by taxonomy."""
    v = str(visual_file_type or "").strip().lower()
    if not v:
        return "Unknown"
    if any(k in v for k in ["drawing", "diagram", "map"]):
        return "Drawing"
    if any(k in v for k in ["photo", "render", "image"]):
        return "Media"
    if any(k in v for k in ["table", "chart"]):
        return "Data"
    if "document" in v:
        return "Document"
    return "Unknown"


# ---------------------------------------------------------------------------
# Post-processing guards
# ---------------------------------------------------------------------------

def _normalize_confidentiality_label(value: str) -> str:
    v = str(value or "").strip().lower()
    if v == "confidential":
        return "Confidential"
    if v in {"sensitive", "standard", "no issue", "no_issue", "public", "normal",
             "not confidential", "non-confidential", "not_confidential",
             "unclassified", "open", "unrestricted"}:
        return "Not Confidential"
    if not v:
        return "Not Confidential"
    logger.debug("Unrecognised confidentiality label %r — defaulting to Not Confidential", value)
    return "Not Confidential"


def _normalize_confidence_label(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"high", "strong"}:
        return "High"
    if v in {"medium", "moderate", "mid"}:
        return "Medium"
    return "Low"


def _is_vlm_strong(vlm_hint: dict | None) -> bool:
    if not vlm_hint:
        return False
    vconf = _normalize_confidence_label(str(vlm_hint.get("visual_confidence", "Low")))
    vtype = str(vlm_hint.get("visual_file_type", "")).strip().lower()
    return vconf in {"High", "Medium"} and vtype not in {"", "unknown"}


def _apply_post_guards(
    result: dict,
    meta: dict,
    prescreen: dict,
    vlm_hint: dict | None = None,
    taxonomy: dict | None = None,
) -> dict:
    """Populate certainty fields and VLM overrides from the LLM's own assessment.

    Note: confidentiality and data-asset guards have moved to Layer 3,
    which runs after Layer 2 and applies deterministic corrections there.
    """
    out = dict(result)
    # Normalize confidentiality label from raw LLM output.
    out["confidentiality"] = _normalize_confidentiality_label(str(out.get("confidentiality", "")))

    # Use the LLM's own certainty assessment directly — no separate scoring system
    llm_label = _normalize_confidence_label(str(out.get("certainty", "Low")))
    try:
        llm_score = max(0, min(100, int(out.get("certainty_score") or 0)))
    except (ValueError, TypeError):
        llm_score = {"High": 80, "Medium": 55, "Low": 25}.get(llm_label, 25)

    drawing_mode = str(meta.get("drawing_mode", "")).strip()
    file_format = str(meta.get("format", "")).strip().lower()
    extraction_method = str(meta.get("extraction_method", "")).strip().lower()
    vlm_conf = _normalize_confidence_label(str((vlm_hint or {}).get("visual_confidence", "Low")))
    vlm_strong = _is_vlm_strong(vlm_hint)

    # Visual-first override for drawing-like PDFs with weak text extraction.
    # In this mode, trust VLM domain/asset signals over text-only ambiguity.
    text_weak = (not _has_readable_text(meta)) or extraction_method in {"metadata", "ocr"}
    visual_first_pdf = file_format == "pdf" and vlm_strong and (drawing_mode == "single_page_drawing" or text_weak)
    if visual_first_pdf:
        vd = str((vlm_hint or {}).get("visual_discipline", "")).strip()
        vf = str((vlm_hint or {}).get("visual_file_type", "")).strip()
        mapped_domain = _map_visual_discipline_to_domain(vd, taxonomy=taxonomy)
        mapped_asset = _map_visual_type_to_asset_type(vf)

        if mapped_domain != "Unknown":
            out["domain"] = mapped_domain
        if mapped_asset != "Unknown":
            out["asset_type"] = mapped_asset
            out["is_data_asset"] = "Yes" if mapped_asset == "Data" else "No"

        if llm_label == "Low":
            llm_label = "Medium"
            llm_score = max(llm_score, 55)

    # If text-first certainty is weak but VLM sees a plausible visual signal,
    # avoid over-penalizing the file as low-confidence.
    if vlm_strong and llm_label == "Low":
        llm_label = "Medium"
        llm_score = max(llm_score, 55)

    if drawing_mode == "single_page_drawing" and llm_label == "High":
        llm_label = "Medium"
        llm_score = min(llm_score, 74)

    out["certainty"]        = llm_label
    out["certainty_score"]  = llm_score
    out["certainty_reason"] = str(out.get("_reasoning", ""))[:120]
    out["certainty_llm"]    = llm_label

    # Backward-compatible aliases
    out["confidence"]        = llm_label
    out["confidence_score"]  = llm_score
    out["confidence_reason"] = out["certainty_reason"]
    out["confidence_llm"]    = llm_label

    # Multi-domain normalization (backward compatible with single-domain output).
    # The normalized primary is written back so the `domain` field can never
    # carry an off-vocabulary label (e.g. "Mobility & Transport" when the
    # taxonomy defines "Mobility & Transit").
    primary_domain = str(out.get("domain", "Unknown")).strip() or "Unknown"
    domains = _coerce_domain_candidates(out.get("domain_candidates"), primary_domain, taxonomy=taxonomy)
    out["domain_candidates"] = domains
    out["domain"] = domains[0]
    out["secondary_domains"] = ", ".join([d for d in domains[1:] if d and d != "Unknown"])

    # Multi-asset-type normalization (mirrors domain_candidates logic).
    primary_asset = str(out.get("asset_type", "Unknown")).strip() or "Unknown"
    asset_types = _coerce_asset_type_candidates(out.get("asset_type_candidates"), primary_asset, taxonomy=taxonomy)
    out["asset_type_candidates"] = asset_types
    out["asset_type"] = asset_types[0]

    # Single-valued dimensions get the same off-vocabulary protection.
    out["lifecycle"] = _snap_to_vocab(out.get("lifecycle"), taxonomy, "lifecycle_stages",
                                      aliases=_LIFECYCLE_ALIASES)
    out["scale"] = _snap_to_vocab(out.get("scale"), taxonomy, "scales")
    out["information_type"] = _snap_to_vocab(out.get("information_type"), taxonomy, "information_types")
    out["governance"] = _snap_to_vocab(out.get("governance"), taxonomy, "governance_sources")

    # Expose VLM audit signals for downstream risk logic.
    out["vlm_confidence"] = vlm_conf if vlm_hint else ""
    out["vlm_note"]       = str((vlm_hint or {}).get("visual_note", "")).strip()
    out["vlm_file_type"]  = str((vlm_hint or {}).get("visual_file_type", "")).strip()

    # Confidentiality reason comes from the main LLM classification, not prescreen.
    out["confidentiality_reason"] = str(out.get("confidentiality_reason", "")).strip()

    # Prefix summary when certainty is Low or extraction was metadata-only.
    # If VLM is strong, explicitly mark visual-based inference instead of
    # filename-only inference.
    if llm_label == "Low" or extraction_method == "metadata":
        summary = str(out.get("short_summary", "")).strip()
        if not summary.lower().startswith(("based on", "inferred from", "according to", "assuming")):
            prefix = ("Based on visual analysis: "
                      if (extraction_method == "metadata" and vlm_strong) or visual_first_pdf
                      else "Based on filename and folder analysis: "
                      if extraction_method == "metadata"
                      else "Low confidence assessment: ")
            out["short_summary"] = prefix + summary

    if drawing_mode == "single_page_drawing":
        summary = str(out.get("short_summary", "")).strip()
        if summary and not summary.lower().startswith("Drawing-mode assessment:"):
            out["short_summary"] = "Drawing-mode assessment: " + summary

    return out


# ---------------------------------------------------------------------------
# Fallback result
# ---------------------------------------------------------------------------

def _fallback(
    meta: dict,
    error_msg: str,
    prescreen: dict | None = None,
    vlm_hint: dict | None = None,
) -> dict:
    logger.error("layer2: classification failed for %s — %s", meta.get("filename"), error_msg)
    vlm_conf = _normalize_confidence_label(str((vlm_hint or {}).get("visual_confidence", "Low")))
    vlm_strong = _is_vlm_strong(vlm_hint)
    certainty = "Medium" if vlm_strong else "Low"
    certainty_score = 55 if vlm_strong else 0
    certainty_reason = (
        "LLM failed; VLM provided usable visual signal"
        if vlm_strong
        else "fallback due to classification error"
    )

    return {
        "_reasoning":             "Classification failed — see llm field.",
        "domain":                 "Unknown",
        "domain_candidates":      ["Unknown"],
        "secondary_domains":      "",
        "scale":                  "Non-spatial",
        "information_type":       meta.get("information_type", "Unknown"),
        "lifecycle":              "Unknown",
        "governance":             "Unknown",
        "confidentiality":        "Not Confidential",
        "confidentiality_reason": "fallback due to classification error",
        "asset_type":             "Unknown",
        "is_data_asset":          "No",
        "year":                   meta.get("year"),
        "year_confidence":        "unknown",
        "certainty":              certainty,
        "certainty_score":        certainty_score,
        "certainty_reason":       certainty_reason,
        "certainty_llm":          certainty,
        "confidence":             certainty,
        "confidence_score":       certainty_score,
        "confidence_reason":      certainty_reason,
        "confidence_llm":         certainty,
        "vlm_confidence":         (vlm_conf if vlm_hint else ""),
        "vlm_note":               str((vlm_hint or {}).get("visual_note", "")).strip(),
        "vlm_file_type":          str((vlm_hint or {}).get("visual_file_type", "")).strip(),
        "short_summary":          "Classification failed.",
        "keywords":               [],
        "llm":                    f"error: {error_msg}",
        "_prescreen":             prescreen or {},
        "_vlm":                   vlm_hint or {},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def layer2_domain(
    meta: dict,
    client: _ClientT,
    model: str,
    input_path: Path,
    project_context: str,
    temperature: float = 0,
    taxonomy: dict | None = None,
    settings: dict | None = None,
    layer2_rules: dict | None = None,
    api_timeout: int = 30,
    folder_context: list[dict] | None = None,
) -> dict:
    """Classify a file using two sequential LLM calls.

    Call 1 (pre-screen): cheap — judges data likelihood and info value signal.
    Call 2 (main):       full classification using taxonomy + pre-screen results.
                         Confidentiality is decided entirely by the main prompt,
                         which receives the full taxonomy confidentiality definitions.

    folder_context — list of already-classified siblings in the same directory,
                     each a dict with filename/domain/lifecycle/asset_type/certainty.
                     Used to calibrate the classification via RAG-style injection.

    Returns dict with all classification fields plus audit keys
    (_reasoning, _prescreen, certainty_score, certainty_reason, llm).
    """
    tx           = _effective_taxonomy(taxonomy)
    folder_chain = _build_folder_chain(meta, input_path)
    prescreen    = _prescreen(meta, folder_chain, client, model, api_timeout)
    vlm_hint     = _vlm_hint_visual_first(meta, client, model, api_timeout, prescreen=prescreen)
    prompt       = _build_prompt(meta, input_path, project_context,
                                 tx, folder_chain, prescreen, vlm_hint, layer2_rules,
                                 folder_context=folder_context)

    def _call(temp: float) -> dict:
        resp    = _llm_call(
            client, model,
            system=(
                "You are an expert AEC data classifier. "
                "Respond ONLY with a single valid JSON object — "
                "no markdown fences, no commentary before or after."
            ),
            user=prompt, temperature=temp, api_timeout=api_timeout,
        )
        result  = _parse_json(resp.choices[0].message.content or "")
        missing = _validate_output(result)
        if missing:
            raise ValueError(f"LLM output missing required keys: {missing}")
        return result

    # Attempt 1 — deterministic
    try:
        result               = _call(temperature)
        result               = _apply_post_guards(result, meta, prescreen, vlm_hint=vlm_hint, taxonomy=tx)

        # Second-order evaluation: critique summary and potentially improve it
        critique = _critique_summary(
            short_summary=result.get("short_summary", ""),
            filename=meta.get("filename", ""),
            domain=result.get("domain", "Unknown"),
            asset_type=result.get("asset_type", "Unknown"),
            client=client,
            model=model,
            api_timeout=api_timeout,
        )
        result["short_summary"]   = critique["improved_summary"]
        result["summary_quality"] = critique["quality_score"]
        result["summary_issues"]  = critique["issues"]

        result["llm"]        = "ok"
        result["_prescreen"] = prescreen
        result["_vlm"]       = vlm_hint
        return result

    # Attempt 2 — retry with slight temperature boost
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("layer2: parse/validation failed for %s (%s: %s), retrying.",
                       meta.get("filename"), type(exc).__name__, exc)
        try:
            result               = _call(max(temperature + 0.3, 0.3))
            result               = _apply_post_guards(result, meta, prescreen, vlm_hint=vlm_hint, taxonomy=tx)

            # Second-order evaluation: critique summary and potentially improve it
            critique = _critique_summary(
                short_summary=result.get("short_summary", ""),
                filename=meta.get("filename", ""),
                domain=result.get("domain", "Unknown"),
                asset_type=result.get("asset_type", "Unknown"),
                client=client,
                model=model,
                api_timeout=api_timeout,
            )
            result["short_summary"]   = critique["improved_summary"]
            result["summary_quality"] = critique["quality_score"]
            result["summary_issues"]  = critique["issues"]

            result["llm"]        = f"ok (retry after {type(exc).__name__})"
            result["_prescreen"] = prescreen
            result["_vlm"]       = vlm_hint
            return result
        except Exception as exc2:
            return _fallback(meta, f"retry failed: {exc2}", prescreen, vlm_hint=vlm_hint)

    except Exception as exc:
        return _fallback(meta, str(exc), prescreen, vlm_hint=vlm_hint)
