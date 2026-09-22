"""Layer 1 — Technical metadata + content extraction.

Architecture
------------
Extraction is split into three phases:

Scout  (_scout)
    Cheap, format-agnostic first pass: file metadata + a short raw preview.
    No interpretation, no parsing decisions — pure I/O.

Route  (_routing_agent)
    One LLM call that receives the scout data and returns explicit extraction
    parameters: which PDF pages to read, whether to OCR, content budget, and
    which files to sample inside an archive.  All sampling decisions live here.

Extract  (_extract_content)
    Format-specific I/O executors.  They receive routing parameters and
    execute without any internal heuristics or scoring.

When no LLM client is provided the extractor falls back to safe defaults
(sequential read, OCR enabled, 16000-char budget).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ClientT = Any

# Suppress PIL decompression-bomb warning for large project images (maps, renders)
try:
    from PIL import Image as _PILImg
    _PILImg.MAX_IMAGE_PIXELS = None
except ImportError:
    pass

# On Windows, Tesseract is installed to a fixed path that pytesseract doesn't auto-detect
from .output import is_extraction_failed
import sys as _sys, os as _os
if _sys.platform == "win32":
    try:
        import pytesseract as _tess
        _candidates = [
            # bundled with installer: {app}\tesseract\tesseract.exe
            _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "tesseract", "tesseract.exe"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            _os.path.join(_os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        ]
        for _p in _candidates:
            if _os.path.isfile(_p):
                _tess.pytesseract.tesseract_cmd = _p
                break
    except ImportError:
        pass

# ── Info type map ─────────────────────────────────────────────────────────
INFO_TYPE_MAP = {
    ".pdf":     "Document",
    ".docx":    "Textual",
    ".doc":     "Textual",
    ".txt":     "Textual",
    ".pptx":    "Presentation",
    ".xlsx":    "Tabular",
    ".xls":     "Tabular",
    ".csv":     "Tabular",
    ".json":    "Tabular",
    ".dwg":     "Technical",
    ".dxf":     "Technical",
    ".dwf":     "Technical",
    ".ifc":     "BIM",
    ".rvt":     "BIM",
    ".nwd":     "BIM",
    ".jpg":     "Image",
    ".jpeg":    "Image",
    ".png":     "Image",
    ".tiff":    "Image",
    ".tif":     "Image",
    ".mp4":     "Video",
    ".mov":     "Video",
    ".shp":     "Spatial",
    ".geojson": "Spatial",
    ".kml":     "Spatial",
    ".gpkg":    "Spatial",
    ".eml":     "Email",
    ".msg":     "Email",
    ".zip":     "Archive",
    ".7z":      "Archive",
    ".rar":     "Archive",
    # Design apps
    ".ai":      "Visual",
    ".psd":     "Image",
    ".indd":    "Document",
    # Other CAD / 3D
    ".dgn":     "Technical",
    ".3dm":     "Technical",
    # Parametric / plain text
    ".gh":      "Technical",
    ".lst":     "Textual",
}

# ── DWG lookup tables (factual, not heuristic) ────────────────────────────
_DWG_VERSION_MAP = {
    "AC1032": "AutoCAD 2018–2021",
    "AC1027": "AutoCAD 2013–2017",
    "AC1024": "AutoCAD 2010–2012",
    "AC1021": "AutoCAD 2007–2009",
    "AC1018": "AutoCAD 2004–2006",
    "AC1015": "AutoCAD 2000–2002",
    "AC1014": "AutoCAD R14",
    "AC1012": "AutoCAD R13",
    "AC1009": "AutoCAD R12",
}
_DWG_DISCIPLINES = {
    "A": "Architecture", "S": "Structural", "M": "Mechanical",
    "E": "Electrical",   "L": "Landscape",  "C": "Civil",
    "P": "Plumbing",     "F": "Fire",        "I": "Interiors",
}
_DWG_PHASES = {
    "SK": "Sketch",      "SD": "Concept / Schematic",  "DD": "Design Development",
    "CD": "Construction Documents", "CA": "Construction Administration", "AB": "As-Built / Handover",
    "IFC": "Issued for Construction",
}

_SIZE_CATEGORIES = [
    (0,      50,     "tiny"),
    (50,     500,    "small"),
    (500,    5_000,  "medium"),
    (5_000,  50_000, "large"),
    (50_000, None,   "xlarge"),
]

PDF_OCR_CHAR_THRESHOLD = 40
PDF_OCR_MAX_PAGES      = 5
PDF_OCR_DPI            = 200
PDF_OCR_MAX_SIDE       = 2200
DWG_CONVERT_TIMEOUT    = 45
DEFAULT_MAX_CHARS      = 1600

# ── Format categories for routing defaults ────────────────────────────────
_VISUAL_FORMATS  = {".dwg", ".dxf", ".dwf", ".rvt", ".nwd",
                    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pptx",
                    ".ai", ".psd", ".indd", ".dgn", ".3dm"}
_TABULAR_FORMATS = {".csv", ".xlsx", ".xls", ".json",
                    ".shp", ".geojson", ".kml", ".gpkg"}
_TEXT_FORMATS    = {".docx", ".doc", ".txt", ".eml", ".msg", ".ifc", ".gh", ".lst"}


def _default_mode(ext: str) -> str:
    if ext in _VISUAL_FORMATS:  return "vision"
    if ext in _TABULAR_FORMATS: return "tabular"
    if ext == ".pdf":           return "pdf"
    return "text"


_DEFAULT_ROUTING: dict = {
    "mode":         None,   # resolved via _default_mode at extract time
    "pages":        None,
    "ocr":          True,
    "max_chars":    16000,
    "archive_pick": None,
    "note":         "default",
}


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — Scout
# ══════════════════════════════════════════════════════════════════════════

def _scout(file_path: Path) -> dict:
    """Cheap first pass: file metadata + raw preview.  No interpretation."""
    ext     = file_path.suffix.lower()
    size_kb = file_path.stat().st_size // 1024
    scout: dict = {
        "filename":       file_path.name,
        "extension":      ext,
        "size_kb":        size_kb,
        "suggested_mode": _default_mode(ext),
        "preview":        "",
    }

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
            reader              = PdfReader(str(file_path))
            scout["page_count"] = len(reader.pages)
            scout["preview"]    = (reader.pages[0].extract_text() or "")[:500]
        except Exception:
            pass

    elif ext in (".xlsx", ".xls", ".csv"):
        try:
            import pandas as pd  # type: ignore
            if ext == ".csv":
                df = pd.read_csv(file_path, nrows=3, encoding="utf-8", errors="replace")
            else:
                engine = "openpyxl" if ext == ".xlsx" else "xlrd"
                xl     = pd.ExcelFile(str(file_path), engine=engine)
                scout["sheets"] = xl.sheet_names
                df = pd.read_excel(file_path, sheet_name=xl.sheet_names[0], nrows=3, engine=engine)
            scout["columns"] = list(df.columns)
            scout["preview"] = df.head(3).to_string()
        except Exception:
            pass

    elif ext == ".zip":
        try:
            import zipfile
            with zipfile.ZipFile(file_path) as z:
                scout["archive_files"] = z.namelist()[:30]
        except Exception:
            pass

    elif ext not in _VISUAL_FORMATS:
        try:
            scout["preview"] = file_path.read_text(encoding="utf-8", errors="replace")[:500]
        except Exception:
            pass

    return scout


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — Route
# ══════════════════════════════════════════════════════════════════════════

_ROUTING_PROMPT = """\
You decide how to extract content from a file for document classification.

Scout data:
{scout_json}

Return JSON only — no markdown, no prose:
{{
  "mode":         "vision" | "tabular" | "text" | "pdf",
  "pages":        [0, 1, 2] or null,
  "ocr":          true or false,
  "max_chars":    <integer 3000–20000>,
  "archive_pick": ["exact_filename"] or null,
  "note":         "<one sentence rationale>"
}}

mode
  "vision"  — rasterize to image, described by a vision model.
              Use for: drawings, CAD (DWG/DXF), images, single-page drawing PDFs.
  "tabular" — structured data read by pandas.
              Use for: CSV, Excel, JSON, spatial files.
  "text"    — plain text extraction.
              Use for: DOCX, TXT, email, IFC, multi-page report/specification PDFs.
  "pdf"     — PDF with mixed text and images (text layer + OCR fallback).
              Use when the PDF is neither clearly a drawing nor purely text.

pages        — pdf mode only. 0-indexed page list. null = sequential.
ocr          — attempt OCR (pdf mode only).
max_chars    — total content budget.
archive_pick — ZIP only. Pick 1–2 files by exact name. null = manifest only.
note         — one sentence rationale.
"""


def _routing_agent(
    scout:       dict,
    client:      _ClientT,
    model:       str,
    api_timeout: int = 30,
) -> dict:
    """Single LLM call that returns extraction parameters for this file."""
    prompt = _ROUTING_PROMPT.format(scout_json=json.dumps(scout, ensure_ascii=False))
    try:
        kwargs: dict = dict(
            model       = model,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0,
        )
        try:
            kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(timeout=api_timeout, **kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(timeout=api_timeout, **kwargs)

        routing = json.loads(resp.choices[0].message.content or "{}")
        return {**_DEFAULT_ROUTING, **routing}
    except Exception as e:
        logger.debug("Routing agent failed for %s: %s", scout.get("filename"), e)
        return _DEFAULT_ROUTING.copy()


# ══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════

def _find_years_in_text(text: str, min_year: int = 2000, max_year: Optional[int] = None) -> list[str]:
    if max_year is None:
        max_year = datetime.now().year
    hits = re.findall(r"\b(20\d{2}|19\d{2})\b", text)
    return [h for h in hits if min_year <= int(h) <= max_year]


def _extract_year_with_confidence(
    filename:  str,
    content:   str = "",
    file_path: Optional[Path] = None,
    min_year:  int = 2000,
) -> tuple[Optional[str], str]:
    """Return (year_str, confidence).  Priority: filename > content frequency > mtime."""
    current_year = datetime.now().year

    fn_years = _find_years_in_text(filename, min_year=min_year, max_year=current_year)
    if fn_years:
        return max(fn_years, key=int), "high"

    if content:
        cleaned = re.sub(r"\b[A-Z]{1,5}[\s\-]\d{3,6}[-:]\d{2,4}\b", "", content)
        cleaned = re.sub(r"©\s*\d{4}", "", cleaned)
        cleaned = re.sub(r"v\d+\.\d+\s*\(\d{4}\)", "", cleaned)
        # Strip "AutoCAD YYYY" / "AutoCAD YYYY–YYYY" version strings so the
        # file-format specifier is never mistaken for the document creation year.
        # Works regardless of what other fields appear in the DWG bracket tag.
        cleaned = re.sub(r"AutoCAD\s+\d{4}(?:\s*[\-–]\s*\d{4})?",
                         "AutoCAD", cleaned)
        cy = _find_years_in_text(cleaned, min_year=min_year, max_year=current_year)
        if cy:
            freq     = Counter(cy)
            max_freq = max(freq.values())
            best     = max((y for y, c in freq.items() if c == max_freq), key=int)
            return best, ("high" if max_freq >= 3 else "medium")

    return None, "unknown"


def _extract_path_segments(file_path: Path) -> list[str]:
    _SKIP = {
        "users", "user", "home", "documents", "downloads", "desktop",
        "onedrive", "sharepoint", "sites", "shared documents",
        "c:", "d:", "volumes", "mnt", "/",
    }
    parts = []
    for part in file_path.parts[:-1]:
        clean = part.strip("/\\")
        if clean.lower() not in _SKIP and clean not in ("", "."):
            parts.append(clean)
    return parts[-8:]


def _extract_filename_signals(filename: str) -> dict:
    stem       = Path(filename).stem
    stem_upper = stem.upper()
    tokens     = [t for t in re.split(r"[_\-\.\s]+", stem_upper) if t]
    code_tokens = [t for t in tokens if re.fullmatch(r"[A-Z]{2,6}", t)]
    version_m   = re.search(
        r"[\-_](?:V|REV|R|P)[\-_]?(\d{1,3}|[A-F])(?=[\-_.]|$)|[\-_](\d{2})(?=[\-_.]|$)",
        stem_upper,
    )
    drawing_m = re.search(r"\b([A-Z]{1,4}[\-\.]\d{3,5})\b", stem_upper)
    has_date  = bool(
        re.search(r"20\d{2}[0-1]\d[0-3]\d", filename)
        or re.search(r"20\d{2}[-_][0-1]\d[-_][0-3]\d", filename)
    )
    return {
        "code_tokens":      code_tokens,
        "version":          version_m.group(0).strip("-_") if version_m else None,
        "drawing_number":   drawing_m.group(1) if drawing_m else None,
        "has_date_in_name": has_date,
        "numeric_tokens":   [t for t in tokens if re.fullmatch(r"\d{1,5}", t)],
        "token_count":      len(tokens),
        "raw_stem":         stem,
    }


def _size_category(size_kb: int) -> str:
    for lo, hi, label in _SIZE_CATEGORIES:
        if hi is None or size_kb < hi:
            if size_kb >= lo:
                return label
    return "xlarge"


def _is_single_page_drawing_pdf(
    file_path: Path,
    page_count: Optional[int],
    content_sample: str,
    filename_signals: dict,
) -> bool:
    """Detect single-page drawing-like PDFs where text extraction is usually weak."""
    if file_path.suffix.lower() != ".pdf" or page_count != 1:
        return False

    stem = file_path.stem.upper()
    name_hint = bool(re.search(
        r"\b(PLAN|SECTION|ELEV(?:ATION)?|DETAIL|LAYOUT|GA|GENERAL ARRANGEMENT|SITE)\b|\bA[0-4]\b",
        stem,
    ))
    structure_hint = bool(
        filename_signals.get("drawing_number")
        or filename_signals.get("version")
        or filename_signals.get("code_tokens")
    )
    text_density = len(re.sub(r"\s+", "", content_sample or ""))
    weak_text = text_density < 120

    return weak_text and (name_hint or structure_hint)


# ══════════════════════════════════════════════════════════════════════════
# OCR helpers
# ══════════════════════════════════════════════════════════════════════════

def _ocr_image(img) -> str:
    try:
        import pytesseract  # type: ignore
        max_side = max(img.width, img.height)
        if max_side > PDF_OCR_MAX_SIDE:
            ratio = PDF_OCR_MAX_SIDE / float(max_side)
            img   = img.resize((int(img.width * ratio), int(img.height * ratio)))
        return re.sub(r"\s+", " ", pytesseract.image_to_string(img) or "").strip()
    except Exception:
        return ""


def _render_page_to_pil(file_path: Path, page_index: int):
    try:
        from pdf2image import convert_from_path  # type: ignore
        images = convert_from_path(
            str(file_path), dpi=PDF_OCR_DPI,
            first_page=page_index + 1, last_page=page_index + 1,
        )
        return images[0] if images else None
    except Exception:
        pass
    try:
        import fitz  # type: ignore
        from PIL import Image
        doc = fitz.open(str(file_path))
        try:
            page = doc.load_page(page_index)
            zoom = max(1.0, PDF_OCR_DPI / 72.0)
            pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            mode = "RGB" if pix.n >= 3 else "L"
            return Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        finally:
            doc.close()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# Vision helpers
# ══════════════════════════════════════════════════════════════════════════

def _rasterize(file_path: Path, page_index: int = 0):
    """Rasterize a file to a PIL Image.  Returns None if not possible."""
    ext = file_path.suffix.lower()

    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        try:
            from PIL import Image  # type: ignore
            return Image.open(file_path).convert("RGB")
        except Exception:
            return None

    if ext == ".pdf":
        return _render_page_to_pil(file_path, page_index)

    # CAD / BIM / presentation / design apps: LibreOffice → PDF → render
    if ext in (".dwg", ".dxf", ".dwf", ".rvt", ".nwd", ".pptx",
               ".ai", ".dgn", ".indd"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _try_libreoffice_convert(file_path, Path(tmp_dir), DWG_CONVERT_TIMEOUT)
            if pdf_path:
                return _render_page_to_pil(pdf_path, 0)
        return None

    # PSD — Pillow native support
    if ext == ".psd":
        try:
            from PIL import Image  # type: ignore
            return Image.open(file_path).convert("RGB")
        except Exception:
            return None

    return None


_VISION_PROMPT = (
    "This is a page from a project document or drawing. "
    "Describe what you see concisely: document type, visible content, key labels, "
    "drawing numbers, discipline, phase, and any readable text. "
    "Be factual. Do not speculate beyond what is visible."
)


def _vision_extract(
    image,
    client:      _ClientT,
    model:       str,
    api_timeout: int = 30,
    max_chars:   int = 2000,
) -> tuple[str, str]:
    """Send a rasterized image to the VLM.  Returns (description, coverage)."""
    import base64, io as _io  # noqa: E401
    try:
        img = image
        if img.mode != "RGB":
            img = img.convert("RGB")
        max_side = 2000
        if max(img.width, img.height) > max_side:
            ratio = max_side / max(img.width, img.height)
            img   = img.resize((int(img.width * ratio), int(img.height * ratio)))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return f"[Vision error — image prep: {e}]", "extraction failed — image preparation"

    try:
        resp = client.chat.completions.create(
            model    = model,
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }],
            max_tokens  = 600,
            temperature = 0,
            timeout     = api_timeout,
        )
        description = (resp.choices[0].message.content or "").strip()
        return description[:max_chars], "vision — VLM description"
    except Exception as e:
        return f"[Vision error: {e}]", "extraction failed — VLM vision call"


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 — Extract: format-specific I/O executors (no decision logic)
# ══════════════════════════════════════════════════════════════════════════

def _extract_pdf(
    file_path: Path,
    pages:     list[int] | None,
    ocr:       bool,
    max_chars: int,
) -> tuple[str, int, str]:
    """Execute PDF extraction with explicit routing parameters."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return "[PDF error: pypdf not installed]", 0, "extraction failed — missing dependency: pypdf"

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        msg    = str(e).lower()
        reason = "file corrupted" if any(w in msg for w in ("corrupted", "bad", "invalid")) else type(e).__name__
        return f"[PDF error: {e}]", 0, f"extraction failed — {reason}"

    total = len(reader.pages)
    if total == 0:
        return "[PDF: empty document]", 0, "extraction failed — file empty"

    sample     = [i for i in pages if 0 <= i < total] if pages else list(range(total))
    parts      = []
    ocr_pages  = 0
    ocr_avail  = True
    chars_used = 0
    pages_read = 0

    for i in sample:
        if chars_used >= max_chars:
            break
        try:
            raw = (reader.pages[i].extract_text() or "").strip()
        except Exception:
            raw = ""

        if raw and len(raw) >= PDF_OCR_CHAR_THRESHOLD:
            snippet = raw[: max_chars - chars_used]
            parts.append(f"[p{i+1}] {snippet}")
            chars_used += len(snippet)
            pages_read += 1
        elif ocr and ocr_avail:
            img = _render_page_to_pil(file_path, i)
            if img is None:
                ocr_avail = False
            else:
                ocr_text = _ocr_image(img)
                if ocr_text:
                    snippet = ocr_text[: max_chars - chars_used]
                    parts.append(f"[p{i+1}][OCR] {snippet}")
                    chars_used += len(snippet)
                    ocr_pages  += 1
                    pages_read += 1

    content  = "\n".join(parts)
    coverage = f"{pages_read}/{total} pages · {chars_used} chars"
    if ocr_pages:
        coverage += f" — {ocr_pages} OCR"
    if not content.strip():
        coverage = (f"extraction failed — {total} pages blank after OCR" if ocr
                    else f"extraction failed — {total} pages, no text layer (OCR disabled)")
        content  = f"[PDF: no extractable content — {total} pages]"

    return content, total, coverage


def _dwg_filename_meta(file_path: Path) -> list[str]:
    parts = []
    stem  = file_path.stem
    drawing = re.search(r"[A-Z]{1,5}[\-\._ ]\d{3,5}", stem)
    if drawing:
        parts.append(f"Drawing: {drawing.group()}")
    disc = re.search(r"[\-\._ ]([ASMELCPFI])[\-\._ ]", stem)
    if disc:
        parts.append(f"Discipline: {_DWG_DISCIPLINES.get(disc.group(1), disc.group(1))}")
    phase = re.search(r"\b(SK|DD|CD|SD|CA|AB|IFC)\b", stem, re.IGNORECASE)
    if phase:
        parts.append(f"Phase: {_DWG_PHASES.get(phase.group(1).upper(), phase.group(1))}")
    return parts


def _dwg_binary_version(file_path: Path) -> Optional[str]:
    try:
        header = file_path.read_bytes()[:6].decode("ascii", errors="replace").strip()
        return _DWG_VERSION_MAP.get(header, header if re.match(r"AC\d{4}", header) else None)
    except Exception:
        return None


def _try_libreoffice_convert(file_path: Path, out_dir: Path, timeout: int) -> Optional[Path]:
    lo = shutil.which("soffice") or shutil.which("libreoffice")
    if not lo:
        return None
    try:
        result = subprocess.run(
            [lo, "--headless", "--norestore", "--nofirststartwizard", "--nologo",
             "--convert-to", "pdf", "--outdir", str(out_dir), str(file_path)],
            capture_output=True, timeout=timeout,
            env={**_os.environ, "SAL_USE_VCLPLUGIN": "svp"},
        )
        if result.returncode != 0:
            return None
        pdf_path = out_dir / (file_path.stem + ".pdf")
        return pdf_path if pdf_path.exists() else None
    except Exception:
        return None


def _try_ezdxf_metadata(file_path: Path) -> Optional[dict]:
    try:
        import ezdxf  # type: ignore
        try:
            doc = ezdxf.readfile(str(file_path))
        except Exception:
            return None
        layers: list[str] = [layer.dxf.name for layer in doc.layers][:30]
        counts: Counter   = Counter()
        try:
            for entity in doc.modelspace():
                counts[entity.dxftype()] += 1
        except Exception:
            pass
        header_info: list[str] = []
        try:
            extmin = doc.header.get("$EXTMIN")
            extmax = doc.header.get("$EXTMAX")
            if extmin and extmax:
                header_info.append(
                    f"Extents: {round(extmax[0]-extmin[0],1)} × {round(extmax[1]-extmin[1],1)} units"
                )
        except Exception:
            pass
        return {"layers": layers, "entities": counts.most_common(10), "header": header_info}
    except ImportError:
        return None


def _dwg_fallback(file_path: Path) -> tuple[str, Optional[int], str]:
    """Fallback for CAD/BIM when vision (rasterize) is unavailable."""
    ext      = file_path.suffix.lower()
    metadata = _dwg_filename_meta(file_path)
    ver      = _dwg_binary_version(file_path)
    if ver:
        metadata.append(f"Format: {ver}")
    tag = ext.lstrip(".").upper()

    # DXF: try ezdxf for layer/entity/text data
    if ext == ".dxf":
        ez = _try_ezdxf_metadata(file_path)
        if ez:
            parts = (["[DXF] " + " | ".join(metadata)] if metadata else []) + ez["header"]
            if ez["layers"]:
                parts.append(f"Layers ({len(ez['layers'])}): " + ", ".join(ez["layers"]))
            if ez["entities"]:
                parts.append("Entities: " + ", ".join(f"{t}×{c}" for t, c in ez["entities"]))
            try:
                import ezdxf  # type: ignore
                doc   = ezdxf.readfile(str(file_path))
                texts = [
                    (e.dxf.text or "").strip()
                    for e in doc.modelspace()
                    if e.dxftype() in ("TEXT", "MTEXT") and hasattr(e.dxf, "text")
                    and len((e.dxf.text or "").strip()) > 3
                ]
                if texts:
                    parts.append("Drawing text: " + " | ".join(dict.fromkeys(texts)[:20]))
            except Exception:
                pass
            return "\n".join(parts)[:DEFAULT_MAX_CHARS], None, f"ezdxf — {len(ez['layers'])} layers"

    if metadata:
        return f"[{tag}: {' | '.join(metadata)}]", None, "filename + binary header"
    return (f"[{tag} binary — analyze filename and folder path]", None,
            "extraction failed — LibreOffice not installed")


def _extract_tabular(file_path: Path, max_chars: int) -> tuple[str, Optional[int], str]:
    """Unified tabular extraction: CSV, Excel, JSON, spatial."""
    ext = file_path.suffix.lower()

    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd  # type: ignore
            engine   = "openpyxl" if ext == ".xlsx" else "xlrd"
            xl       = pd.ExcelFile(str(file_path), engine=engine)
            n_sheets = len(xl.sheet_names)
            # Sample up to 3 sheets so that data in sheet 2+ is not silently
            # ignored.  Budget max_chars evenly across sampled sheets.
            max_sample  = min(n_sheets, 3)
            per_budget  = max(400, (max_chars - 80) // max_sample)
            parts       = [f"Sheets: {xl.sheet_names}"]
            sampled     = []
            for sheet_name in xl.sheet_names[:max_sample]:
                try:
                    df    = pd.read_excel(file_path, sheet_name=sheet_name,
                                          nrows=5, engine=engine)
                    chunk = (f"\n[Sheet: {sheet_name}]\n"
                             f"Columns: {list(df.columns)}\n"
                             f"{df.head(5).to_string()}")
                    parts.append(chunk[:per_budget])
                    sampled.append(sheet_name)
                except Exception:
                    pass
            content  = "".join(parts)[:max_chars]
            coverage = (
                f"all {n_sheets} sheet(s) — first 5 rows each"
                if n_sheets <= max_sample
                else f"{len(sampled)} of {n_sheets} sheets sampled — first 5 rows each"
            )
            return content, None, coverage
        except Exception as e:
            return f"[Excel error: {e}]", None, f"extraction failed — {type(e).__name__}"

    if ext == ".csv":
        try:
            import pandas as pd  # type: ignore
            df      = pd.read_csv(file_path, nrows=5, encoding="utf-8", errors="replace")
            content = f"Columns: {list(df.columns)}\n{df.head(5).to_string()}"[:max_chars]
            return content, None, "first 5 rows + headers"
        except Exception as e:
            return f"[CSV error: {e}]", None, f"extraction failed — {type(e).__name__}"

    if ext == ".json":
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            return raw[:max_chars], None, (
                f"full content ({len(raw)} chars)" if len(raw) <= max_chars
                else f"truncated at {max_chars} chars"
            )
        except Exception as e:
            return f"[JSON error: {e}]", None, "extraction failed"

    if ext in (".shp", ".geojson", ".kml", ".gpkg"):
        try:
            import geopandas as gpd  # type: ignore
            gdf    = gpd.read_file(str(file_path), rows=5)
            bounds = gdf.geometry.total_bounds
            content = "\n".join([
                f"[Spatial — CRS: {gdf.crs}]",
                f"Bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}] to [{bounds[2]:.4f}, {bounds[3]:.4f}]",
                f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}",
                f"Columns: {list(gdf.columns)}",
                f"Features: {len(gdf)}",
            ])[:max_chars]
            return content, None, f"geopandas — {len(gdf)} features, CRS: {gdf.crs}"
        except ImportError:
            try:
                raw = file_path.read_text(encoding="utf-8", errors="replace")
                return raw[:400], None, f"{ext.upper()} — text fallback"
            except Exception:
                return f"[{ext.upper()} — binary]", None, "extraction failed — missing dependency: geopandas"
        except Exception as e:
            msg    = str(e).lower()
            reason = "file corrupted" if any(w in msg for w in ("corrupt", "invalid")) else type(e).__name__
            return f"[Spatial error: {e}]", None, f"extraction failed — {reason}"

    return "", None, "extraction failed — unsupported tabular format"


def _extract_text_doc(file_path: Path, max_chars: int) -> tuple[str, Optional[int], str]:
    """Unified text extraction: DOCX, TXT, EML, MSG, IFC."""
    ext = file_path.suffix.lower()

    if ext in (".docx", ".doc"):
        try:
            import docx  # type: ignore
            from docx.oxml.ns import qn  # type: ignore
            doc        = docx.Document(str(file_path))
            para_texts = [p.text for p in doc.paragraphs[:20] if p.text.strip()]
            # Also read text boxes (shapes) which doc.paragraphs misses
            txbx_texts: list[str] = []
            for el in doc.element.body.iter():
                if el.tag == qn("w:txbxContent"):
                    for p_el in el.iter(qn("w:p")):
                        t = "".join(r.text for r in p_el.iter(qn("w:t")) if r.text).strip()
                        if t and len(txbx_texts) < 10:
                            txbx_texts.append(t)
            table_texts: list[str] = []
            for idx, table in enumerate(doc.tables[:3]):
                table_texts.append(f"[Table {idx+1}]")
                for row in table.rows[:3]:
                    row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_text:
                        table_texts.append(row_text)
            parts   = para_texts + txbx_texts + (["[Tables:]"] + table_texts if table_texts else [])
            content = "\n".join(parts)[:max_chars]
            if not content.strip():
                return "[DOCX: empty]", None, "extraction failed — file empty"
            cov_parts = [f"{len(para_texts)} paragraphs"]
            if txbx_texts:
                cov_parts.append(f"{len(txbx_texts)} text-box(es)")
            if doc.tables:
                cov_parts.append(f"{len(doc.tables)} table(s)")
            coverage = " + ".join(cov_parts)
            return content, None, coverage
        except ImportError:
            return "[DOCX error: python-docx not installed]", None, "extraction failed — missing dependency: python-docx"
        except Exception as e:
            msg    = str(e).lower()
            reason = "file corrupted" if any(w in msg for w in ("corrupt", "invalid")) else type(e).__name__
            return f"[DOCX error: {e}]", None, f"extraction failed — {reason}"

    if ext == ".txt":
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            return raw[:max_chars], None, (
                f"full content ({len(raw)} chars)" if len(raw) <= max_chars
                else f"truncated at {max_chars} chars"
            )
        except Exception as e:
            return f"[Text error: {e}]", None, "extraction failed"

    if ext == ".eml":
        try:
            import email
            from email import policy as email_policy  # type: ignore
            with open(file_path, encoding="utf-8", errors="replace") as f:
                msg = email.message_from_file(f, policy=email_policy.default)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = (part.get_content() or "")[:400]
                        break
            else:
                body = (msg.get_content() or "")[:400]
            content = (f"From: {msg.get('From', '')}\n"
                       f"Subject: {msg.get('Subject', '')}\n"
                       f"Date: {msg.get('Date', '')}\n"
                       f"Body: {body.strip()}")
            return content, None, "headers + body excerpt"
        except Exception as e:
            return f"[EML error: {e}]", None, "extraction failed"

    if ext == ".msg":
        try:
            import extract_msg  # type: ignore
            msg     = extract_msg.Message(str(file_path))
            content = (f"From: {msg.sender}\n"
                       f"Subject: {msg.subject}\n"
                       f"Date: {msg.date}\n"
                       f"Body: {(msg.body or '')[:400]}")
            return content, None, "extract-msg — headers + body excerpt"
        except ImportError:
            return "[MSG binary — install extract-msg]", None, "extraction failed — missing dependency: extract-msg"
        except Exception as e:
            msg_str = str(e).lower()
            reason  = "file corrupted" if any(w in msg_str for w in ("corrupt", "invalid")) else type(e).__name__
            return f"[MSG error: {e}]", None, f"extraction failed — {reason}"

    if ext == ".ifc":
        try:
            raw    = file_path.read_text(encoding="utf-8", errors="replace")
            desc_m = re.search(r"FILE_DESCRIPTION\s*\(([^;]+)\)", raw)
            sch_m  = re.search(r"FILE_SCHEMA\s*\(\s*\('([^']+)'\)", raw)
            name_m = re.search(r"FILE_NAME\s*\('([^']*)'", raw)
            ents   = re.findall(r"^#\d+=\s*(IFC[A-Z0-9]+)\(", raw, re.MULTILINE)
            top_ent = ", ".join(f"{e}×{c}" for e, c in Counter(ents).most_common(8))
            header = "\n".join(filter(None, [
                f"[IFC schema: {sch_m.group(1)}]"                 if sch_m  else "",
                f"[Authored as: {name_m.group(1)}]"               if name_m else "",
                f"[Description: {desc_m.group(1)[:200].strip()}]" if desc_m else "",
                f"[Top entity types: {top_ent}]"                  if top_ent else "",
            ]))
            return (header + "\n" + raw[:800])[:max_chars], None, "IFC header + entity types"
        except Exception as e:
            msg    = str(e).lower()
            reason = ("file corrupted" if "invalid" in msg
                      else "encoding issue" if any(w in msg for w in ("encoding", "decode"))
                      else type(e).__name__)
            return f"[IFC error: {e}]", None, f"extraction failed — {reason}"

    return "", None, "extraction failed — unsupported text format"


def _extract_zip(
    file_path:    Path,
    archive_pick: list[str] | None,
    max_chars:    int,
) -> tuple[str, Optional[int], str]:
    try:
        import zipfile
        try:
            with zipfile.ZipFile(file_path) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            return (f"[ZIP error: corrupted — {file_path.stat().st_size // 1024} KB]",
                    None, "extraction failed — file corrupted")

        ext_sum = ", ".join(
            f"{e}×{c}" for e, c in Counter(
                Path(n).suffix.lower() for n in names if Path(n).suffix
            ).most_common(5)
        )
        header = f"[ZIP: {len(names)} files | types: {ext_sum} | samples: {', '.join(names[:8])}]"

        sampled: list[str] = []
        if archive_pick:
            with zipfile.ZipFile(file_path) as z:
                with tempfile.TemporaryDirectory() as tmp:
                    for name in archive_pick:
                        if name not in names:
                            continue
                        try:
                            z.extract(name, tmp)
                            sub_content, _, sub_cov = _extract_content(
                                Path(tmp) / name, routing=_DEFAULT_ROUTING
                            )
                            sampled.append(f"[sampled: {Path(name).name} | {sub_cov}]\n{sub_content}")
                        except Exception:
                            pass

        content  = (header + ("\n\n" + "\n\n".join(sampled) if sampled else ""))[:max_chars]
        coverage = (
            f"{len(names)} files in archive"
            + (f" — {len(sampled)} file(s) content-sampled" if sampled
               else " — manifest only (no files picked by routing)")
        )
        return content, None, coverage
    except Exception as e:
        return f"[ZIP error: {e}]", None, f"extraction failed — {type(e).__name__}"


# ══════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════

def _extract_content(
    file_path:   Path,
    routing:     dict,
    client:      _ClientT | None = None,
    model:       str | None      = None,
    api_timeout: int             = 30,
) -> tuple[str, Optional[int], str]:
    """Dispatch to extractor based on routing mode.  No decisions here."""
    ext          = file_path.suffix.lower()
    mode         = routing.get("mode") or _default_mode(ext)
    pages        = routing.get("pages")
    ocr          = bool(routing.get("ocr", True))
    max_chars    = int(routing.get("max_chars", 16000))
    archive_pick = routing.get("archive_pick")

    # ── Vision path ───────────────────────────────────────────────────────
    if mode == "vision":
        img = _rasterize(file_path)
        if img is not None and client and model:
            desc, cov = _vision_extract(img, client, model, api_timeout, max_chars)
            # If VLM produced a weak or uncertain description for a raster
            # image, try OCR as a supplementary signal.
            if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
                _WEAK = {
                    "abstract", "distorted", "no identifiable", "cannot determine",
                    "unclear", "no readable text", "chaotic", "vision error",
                    "unable to determine", "not possible to identify",
                }
                desc_lower = desc.lower()
                if any(ph in desc_lower for ph in _WEAK) and len(desc.strip()) < 350:
                    ocr_text = _ocr_image(img)
                    if ocr_text and len(ocr_text.strip()) > 40:
                        desc = f"{desc}\n[OCR fallback] {ocr_text[:500]}"
                        cov  = f"{cov} + OCR (weak VLM)"
            # If VLM failed for a PDF, fall back to text extraction so we don't
            # discard a working text layer just because vision errored.
            if ext == ".pdf" and "extraction failed" in cov.lower():
                return _extract_pdf(file_path, pages=pages, ocr=ocr, max_chars=max_chars)
            return desc, None, cov
        # Fallback: CAD metadata
        if ext in (".dwg", ".dxf", ".dwf", ".rvt", ".nwd"):
            return _dwg_fallback(file_path)
        # Fallback: images → OCR
        if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
            try:
                from PIL import Image  # type: ignore
                img2     = Image.open(file_path).convert("RGB")
                ocr_text = _ocr_image(img2)
                size_kb  = file_path.stat().st_size // 1024
                content  = (
                    f"[Image {img2.width}×{img2.height}px size={size_kb}KB]\n"
                    + (f"[OCR] {ocr_text[:700]}" if ocr_text else "[OCR] no readable text extracted")
                )[:DEFAULT_MAX_CHARS]
                return content, None, f"OCR — {'extracted' if ocr_text else 'no text'}"
            except Exception as e:
                return f"[Image error: {e}]", None, "extraction failed"
        # Fallback: PPTX → text extraction
        if ext == ".pptx":
            try:
                from pptx import Presentation  # type: ignore
                prs   = Presentation(str(file_path))
                parts = []
                for i, slide in enumerate(list(prs.slides)[:8]):
                    texts = [
                        shape.text.strip()
                        for shape in slide.shapes
                        if shape.has_text_frame and shape.text.strip()
                    ]
                    if texts:
                        parts.append(f"[Slide {i+1}] " + " | ".join(texts[:3]))
                if not parts:
                    return "[PPTX: image-only slides]", None, "extraction failed — no text in slides"
                return "\n".join(parts)[:max_chars], None, f"{len(parts)} slides sampled"
            except Exception as e:
                return f"[PPTX error: {e}]", None, "extraction failed"
        # Design app / 3D files → metadata signals when rasterize failed
        if ext in (".ai", ".indd", ".psd", ".dgn", ".3dm"):
            _FORMAT_LABELS = {
                ".ai":   "Adobe Illustrator vector",
                ".indd": "InDesign layout",
                ".psd":  "Photoshop raster",
                ".dgn":  "MicroStation CAD",
                ".3dm":  "Rhino 3D model",
            }
            label = _FORMAT_LABELS.get(ext, ext.lstrip(".").upper())
            fn_meta = _dwg_filename_meta(file_path)
            if fn_meta:
                return f"[{label}: {' | '.join(fn_meta)}]", None, "filename signals only"
            return (f"[{label} — analyze filename and folder path]",
                    None, "extraction failed — binary format, no conversion available")

        return (f"[{ext.upper()} — vision unavailable, no client]",
                None, "extraction failed — no vision client")

    # ── PDF path (text layer + OCR fallback) ──────────────────────────────
    if mode == "pdf" or ext == ".pdf":
        return _extract_pdf(file_path, pages=pages, ocr=ocr, max_chars=max_chars)

    # ── Tabular path ──────────────────────────────────────────────────────
    if mode == "tabular":
        return _extract_tabular(file_path, max_chars)

    # ── Text path ─────────────────────────────────────────────────────────
    if mode == "text":
        return _extract_text_doc(file_path, max_chars)

    # ── Archive ───────────────────────────────────────────────────────────
    if ext == ".zip":
        return _extract_zip(file_path, archive_pick=archive_pick, max_chars=max_chars)

    if ext in (".7z", ".rar"):
        fmt = ext.lstrip(".").upper()
        return (f"[{fmt} archive — {file_path.stat().st_size // 1024} KB — not supported]",
                None, f"extraction failed — {fmt} format not supported")

    if ext in (".mp4", ".mov"):
        fmt = ext.lstrip(".").upper()
        return (f"[{fmt} video — {file_path.stat().st_size // 1024} KB — no text content]",
                None, "extraction failed — video format, no text content")

    return "", None, "extraction failed — unsupported format"


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════

def layer1_technical(
    file_path:   Path,
    client:      _ClientT | None = None,
    model:       str | None      = None,
    api_timeout: int             = 30,
) -> dict:
    """Extract technical metadata and content sample for a single file.

    When client + model are provided, a routing agent (one LLM call) decides
    the extraction strategy.  Otherwise safe defaults are used (sequential
    read, OCR enabled, 16000-char budget).
    """
    ext = file_path.suffix.lower()

    # Phase 1: Scout
    scout = _scout(file_path)

    # Phase 2: Route
    routing = (
        _routing_agent(scout, client, model, api_timeout)
        if client and model
        else _DEFAULT_ROUTING.copy()
    )

    # Phase 3: Extract
    content, pages, coverage = _extract_content(
        file_path, routing,
        client=client, model=model, api_timeout=api_timeout,
    )

    year, year_conf   = _extract_year_with_confidence(file_path.name, content, file_path)
    size_kb           = file_path.stat().st_size // 1024
    cov_lower         = coverage.lower()
    filename_signals  = _extract_filename_signals(file_path.name)
    drawing_mode_flag = (
        "single_page_drawing"
        if _is_single_page_drawing_pdf(file_path, pages, content, filename_signals)
        else ""
    )

    mode = routing.get("mode") or _default_mode(ext)
    if mode == "vision" and "extraction failed" not in cov_lower:
        # DWG/BIM fallback produces "filename + binary header" coverage —
        # no visual analysis occurred, so label it "metadata" not "vision".
        if "binary header" in cov_lower or "filename +" in cov_lower:
            extraction_method = "metadata"
        else:
            extraction_method = "vision"
    elif "ocr" in cov_lower and ("pages" in cov_lower or "text" in cov_lower):
        extraction_method = "text+ocr"
    elif "ocr" in cov_lower:
        extraction_method = "ocr"
    elif "libreoffice" in cov_lower or "converted" in cov_lower:
        extraction_method = "converted"
    elif is_extraction_failed(coverage) or "signals only" in cov_lower:
        extraction_method = "metadata"
    else:
        extraction_method = "text"

    return {
        "filename":            file_path.name,
        "format":              ext.lstrip(".").upper(),
        "format_type":         INFO_TYPE_MAP.get(ext, "Unknown"),
        "year":                year,
        "page_count":          pages,
        "size_kb":             size_kb,
        "folder":              file_path.parent.name,
        "file_path":           str(file_path),
        "content_sample":      content,
        "extraction_coverage": coverage,
        "size_category":       _size_category(size_kb),
        "path_segments":       _extract_path_segments(file_path),
        "filename_signals":    filename_signals,
        "year_confidence":     year_conf,
        "extraction_method":   extraction_method,
        "drawing_mode":        drawing_mode_flag,
        "routing_note":        routing.get("note", ""),
    }
