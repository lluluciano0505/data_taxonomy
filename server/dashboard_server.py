"""
dashboard_server.py — Flask backend for dashboard.html.

Serves the static HTML and exposes the endpoints the dashboard needs:

  GET  /                     — serve dashboard.html
  GET  /api/csv-list         — list available output CSVs
  GET  /api/data?csv=<path>  — full dataset as JSON (normalised vs. taxonomy)
  GET  /api/taxonomy         — taxonomy.yaml as JSON (for editable dropdowns)
  GET  /api/reviewed         — persisted review state (reviewed filenames + edits)
  POST /api/review           — save review edits & mark file reviewed
  POST /api/unreview         — un-mark a file
  GET  /api/preview?path=... — best-effort first-page preview (PDF/image/text)
  POST /api/open-file        — open file / folder / reveal in Finder
  GET  /api/download-csv     — stream a CSV file back to the browser

Run:  python dashboard_server.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, redirect

from core.config_loader import (
    get_paths_config,
    get_project_config,
    get_processing_config,
    load_config as load_app_config,
    load_taxonomy,
    PathsConfig, ProcessingConfig,
)
from core.pipeline import run as pipeline_run, SUPPORTED_FORMATS, build_config

load_dotenv(_PROJECT_ROOT / ".env")

ROOT                = _PROJECT_ROOT
CONFIG_PATH         = ROOT / "config.yaml"
TAXONOMY_PATH       = ROOT / "taxonomy.yaml"
SAVED_CONFIGS_PATH  = ROOT / "saved_configs.json"
REVIEWED_STATE      = ROOT / ".reviewed_files.json"   # sidecar persistence

app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────
def _resolve_default_csv() -> Path:
    """Mirror dashboard.py default: use paths.output_csv from config.yaml."""
    try:
        cfg = load_app_config(str(CONFIG_PATH))
    except Exception:
        cfg = {}
    paths: PathsConfig | None = get_paths_config(cfg) if cfg else None
    out = paths.output_csv.expanduser() if paths else Path("test_output.csv")
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    return out


def _csv_candidates() -> list[Path]:
    default = _resolve_default_csv()
    d = default.parent if str(default.parent) not in {"", "."} else ROOT
    cands: list[Path] = []
    if d.exists():
        cands = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if default.exists() and default not in cands:
        cands.insert(0, default)
    if not cands:
        cands = [default]
    return cands


_review_lock = threading.RLock()


def _review_path(csv_path=None):
    csv_path = Path(csv_path or _resolve_default_csv()).resolve()
    return Path(str(csv_path) + ".reviews.json")


def _load_reviewed_state(csv_path=None) -> dict:
    path = _review_path(csv_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"reviewed": [], "edits": {}}


def _save_reviewed_state(state, csv_path=None):
    path = _review_path(csv_path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _result_config(csv_path):
    snapshot = Path(str(csv_path) + ".config.yaml")
    if not snapshot.exists():
        raise ValueError("This older result has no saved run configuration. Run it again from its project before using AI analysis or incremental processing.")
    return load_app_config(str(snapshot))


def _apply_reviews(df, csv_path):
    edits = _load_reviewed_state(csv_path).get("edits", {})
    for idx, row in df.iterrows():
        for field, value in edits.get(str(row.get("file_path", row.get("filename", ""))), {}).items():
            if field in df.columns and field not in {"file_path", "filename"}:
                df.at[idx, field] = value
    return df


def _cleanup_orphan_sidecars() -> int:
    """Delete .taxonomy.yaml files whose companion CSV no longer exists."""
    removed = 0
    try:
        outputs_dir = _resolve_default_csv().parent
        if not outputs_dir.is_dir():
            return 0
        for sidecar in outputs_dir.glob("*.csv.taxonomy.yaml"):
            companion = Path(str(sidecar)[: -len(".taxonomy.yaml")])
            if not companion.exists():
                sidecar.unlink(missing_ok=True)
                removed += 1
    except Exception:
        pass
    return removed


# ── Static ─────────────────────────────────────────────────────────────────
@app.route("/project-ui")
def project_ui():
    cfg = load_app_config(str(CONFIG_PATH))
    port = int(cfg.get("configuration", {}).get("port", 5173))
    destinations = {"home": "/", "settings": "/wizard.html", "taxonomy": "/taxonomy-editor"}
    path = destinations.get(request.args.get("page", "home"), "/")
    return redirect(f"http://localhost:{port}{path}")


@app.route("/")
def index():
    return send_file(ROOT / "static" / "dashboard.html")


# ── API: CSV list ──────────────────────────────────────────────────────────
@app.route("/api/csv-list")
def csv_list():
    default = _resolve_default_csv()
    return jsonify({
        "csvs": [str(p) for p in _csv_candidates()],
        "default": str(default),
    })


# ── API: Full dataset ──────────────────────────────────────────────────────
# L3 dimensions are rubric levels (strings) as of Jul 2026 — no longer numeric.
_NUMERIC_COLS = [
    "certainty_score", "size_kb", "year",
]


@app.route("/api/data")
def data():
    csv_arg = (request.args.get("csv") or "").strip()
    csv_path = Path(csv_arg).expanduser() if csv_arg else _resolve_default_csv()
    if not csv_path.exists():
        return jsonify({"error": f"CSV not found: {csv_path}"}), 404

    # Return cached response if the file hasn't changed on disk
    try:
        mtime = csv_path.stat().st_mtime
    except Exception:
        mtime = 0
    cached = _data_cache.get(str(csv_path))
    if cached and cached[0] == mtime:
        return jsonify(cached[1])

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return jsonify({"error": f"Could not load CSV: {e}"}), 500

    # Fix domain/scale column swap (early pipeline bug) — do NOT replace unknown values.
    # Rows classified with an older taxonomy keep their original domain names; the
    # dashboard filter pills are built from actual data so they remain accurate.
    sidecar_tx = Path(str(csv_path) + ".taxonomy.yaml")
    tx_path = str(sidecar_tx) if sidecar_tx.exists() else str(TAXONOMY_PATH)
    try:
        tx = load_taxonomy(tx_path)
    except Exception:
        tx = {}
    allowed_d = {str(x.get("name", "")).strip()
                 for x in (tx.get("domains", []) if isinstance(tx, dict) else [])
                 if str(x.get("name", "")).strip()}
    allowed_s = {str(x.get("name", "")).strip()
                 for x in (tx.get("scales", []) if isinstance(tx, dict) else [])
                 if str(x.get("name", "")).strip()}

    if "domain" in df.columns and "scale" in df.columns and allowed_d and allowed_s:
        df["domain"] = df["domain"].fillna("Unknown").astype(str).str.strip()
        df["scale"]  = df["scale"].fillna("Unknown").astype(str).str.strip()
        swap = df["domain"].isin(allowed_s) & df["scale"].isin(allowed_d)
        if swap.any():
            _d = df.loc[swap, "domain"].copy()
            df.loc[swap, "domain"] = df.loc[swap, "scale"]
            df.loc[swap, "scale"]  = _d

    if "file_path" in df.columns:
        df = df.drop_duplicates(subset=["file_path"], keep="first").reset_index(drop=True)

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _apply_reviews(df, csv_path)
    rows = json.loads(df.to_json(orient="records", date_format="iso"))
    provenance = {}
    snapshot = Path(str(csv_path) + ".config.yaml")
    if snapshot.exists():
        saved = load_app_config(str(snapshot))
        processing = get_processing_config(saved)
        provenance = {"project": saved.get("project", {}).get("name", ""),
                      "provider": processing.provider, "model": processing.model,
                      "input_dir": saved.get("paths", {}).get("input_dir", "")}
    result = {"csv": str(csv_path), "count": len(rows), "rows": rows, "run": provenance}
    _data_cache[str(csv_path)] = (mtime, result)
    return jsonify(result)


# ── API: Taxonomy ──────────────────────────────────────────────────────────
_data_cache: dict[str, tuple[float, dict]] = {}      # csv_path → (mtime, response)


@app.route("/api/taxonomy/presets")
def list_taxonomy_presets():
    """List available taxonomy preset files in taxonomies/ folder."""
    presets_dir = ROOT / "taxonomies"
    if not presets_dir.exists():
        return jsonify([])
    presets = []
    for f in sorted(presets_dir.glob("*.yaml")):
        try:
            with open(f, encoding="utf-8") as fp:
                tx = yaml.safe_load(fp) or {}
            domain_count = len(tx.get("domains", []))
            presets.append({"name": f.stem, "file": f.name, "domains": domain_count})
        except Exception:
            continue
    return jsonify(presets)


@app.route("/api/taxonomy/preset/<name>")
def get_taxonomy_preset(name):
    """Return a specific taxonomy preset by filename stem."""
    preset_file = ROOT / "taxonomies" / f"{name}.yaml"
    if not preset_file.exists():
        return jsonify({"error": "not found"}), 404
    with open(preset_file, encoding="utf-8") as f:
        return jsonify(yaml.safe_load(f) or {})


@app.route("/api/taxonomy")
def get_taxonomy():
    """Return the taxonomy that was active when this CSV was produced (sidecar).
    Falls back to the root taxonomy.yaml if no sidecar exists."""
    csv_arg = (request.args.get("csv") or "").strip()

    # Sidecar written by pipeline at run time — only source of truth
    if csv_arg:
        sidecar = Path(csv_arg + ".taxonomy.yaml")
        if not sidecar.exists():
            sidecar = Path(csv_arg).expanduser().parent / (Path(csv_arg).name + ".taxonomy.yaml")
        if sidecar.exists():
            try:
                with open(sidecar, encoding="utf-8") as f:
                    tx = yaml.safe_load(f)
                if tx:
                    return jsonify(tx)
            except Exception:
                pass

    # Fallback: root taxonomy.yaml (for CSV files produced before sidecar was introduced)
    if TAXONOMY_PATH.exists():
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            return jsonify(yaml.safe_load(f) or {})
    return jsonify({})


# ── API: Review state ──────────────────────────────────────────────────────
@app.route("/api/reviewed")
def list_reviewed():
    return jsonify(_load_reviewed_state(request.args.get("csv")))


@app.route("/api/review", methods=["POST"])
@app.route("/api/unreview", methods=["POST"])
def save_review():
    body = request.get_json(force=True) or {}
    csv_path = Path(body.get("csv") or _resolve_default_csv()).resolve()
    key = str(body.get("file_path") or "").strip()
    if not csv_path.exists():
        return jsonify({"error": "Result not found"}), 404
    df = pd.read_csv(csv_path)
    if not key or "file_path" not in df or key not in set(df["file_path"].astype(str)):
        return jsonify({"error": "File does not belong to this result"}), 400
    edits = {k: v for k, v in (body.get("edits") or {}).items() if k in df.columns and k not in {"file_path", "filename"}}
    with _review_lock:
        state = _load_reviewed_state(csv_path)
        if request.path.endswith("/unreview"):
            state["reviewed"] = [x for x in state["reviewed"] if x != key]
            state["edits"].pop(key, None)
        else:
            if key not in state["reviewed"]:
                state["reviewed"].append(key)
            state["edits"][key] = edits
        _save_reviewed_state(state, csv_path)
    _data_cache.pop(str(csv_path), None)
    return jsonify({"ok": True})


# ── API: Preview ───────────────────────────────────────────────────────────
_IMG_EXTS  = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff", ".tif"}
_TEXT_EXTS = {".txt", ".md", ".json", ".yaml", ".yml", ".eml", ".csv"}


def _as_data_url(raw: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


@app.route("/api/preview")
def preview():
    path = (request.args.get("path") or "").strip()
    if not path:
        return jsonify({"kind": "none"})
    p = Path(path)
    if not p.exists() or not p.is_file():
        return jsonify({"kind": "none"})
    ext = p.suffix.lower()

    # Images
    if ext in _IMG_EXTS:
        try:
            from PIL import Image  # type: ignore
            img = Image.open(str(p)).convert("RGB")
            max_side = 900
            if max(img.width, img.height) > max_side:
                r = max_side / float(max(img.width, img.height))
                img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=72, optimize=True)
            return jsonify({"kind": "image", "data": _as_data_url(buf.getvalue())})
        except Exception:
            pass

    # PDFs — try rendering first page, fall back to text extraction
    if ext == ".pdf":
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(p))
            try:
                if len(doc) > 0:
                    page = doc.load_page(0)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
                    try:
                        from PIL import Image  # type: ignore
                        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        max_side = 900
                        if max(img.width, img.height) > max_side:
                            r = max_side / float(max(img.width, img.height))
                            img = img.resize(
                                (int(img.width * r), int(img.height * r)), Image.LANCZOS
                            )
                        buf = io.BytesIO()
                        img.save(buf, "JPEG", quality=65, optimize=True)
                        return jsonify({"kind": "image", "data": _as_data_url(buf.getvalue())})
                    except Exception:
                        return jsonify({
                            "kind": "image",
                            "data": _as_data_url(pix.tobytes("jpg", jpg_quality=60)),
                        })
            finally:
                doc.close()
        except Exception:
            pass
        try:
            from pypdf import PdfReader  # type: ignore
            r = PdfReader(str(p))
            if len(r.pages) > 0:
                txt = (r.pages[0].extract_text() or "").strip()
                if txt:
                    return jsonify({"kind": "text", "data": txt[:1800]})
        except Exception:
            pass

    # Excel / CSV — return a compact textual preview (HTML renders it as a table)
    if ext in {".xlsx", ".xls", ".csv"}:
        try:
            if ext == ".csv":
                head = pd.read_csv(str(p), nrows=12, encoding="utf-8", on_bad_lines="skip")
            else:
                engine = "openpyxl" if ext == ".xlsx" else "xlrd"
                head = pd.read_excel(str(p), nrows=12, engine=engine)
            cols = [str(c) for c in head.columns][:8]
            rows = head.fillna("").astype(str).values.tolist()
            rows = [r[:8] for r in rows]
            return jsonify({"kind": "table", "columns": cols, "rows": rows})
        except Exception:
            pass

    # DOCX — just pull first ~20 paragraphs as text
    if ext in {".docx", ".doc"}:
        try:
            import docx  # type: ignore
            d = docx.Document(str(p))
            paras = [x.text.strip() for x in d.paragraphs[:25] if x.text and x.text.strip()]
            if paras:
                return jsonify({"kind": "text", "data": "\n\n".join(paras)[:1800]})
        except Exception:
            pass

    # Plain text formats
    if ext in _TEXT_EXTS:
        try:
            return jsonify({
                "kind": "text",
                "data": p.read_text(encoding="utf-8", errors="replace")[:1800],
            })
        except Exception:
            pass

    return jsonify({"kind": "none"})


# ── API: Open file in OS ───────────────────────────────────────────────────
@app.route("/api/open-file", methods=["POST"])
def open_file():
    body = request.get_json(force=True) or {}
    path = (body.get("path") or "").strip()
    mode = (body.get("mode") or "file").strip()   # "file" | "folder" | "reveal"
    if not path:
        return jsonify({"error": "path required"}), 400

    p = Path(path)
    try:
        if mode == "reveal":
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(p)])
            elif os.name == "nt":
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p.parent)])
        elif mode == "folder":
            target = str(p.parent)
            if sys.platform == "darwin":
                subprocess.Popen(["open", target])
            elif os.name == "nt":
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", target])
        else:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            elif os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(p)])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Deep analysis (Layer 3 agent) ────────────────────────────────────
@app.route("/api/deep-analysis", methods=["POST"])
def deep_analysis():
    """
    Run the Layer 3 agent on a single file row from the Review Queue.

    POST body:
      csv       — path to the project CSV (for tool search context)
      row       — the full CSV row dict for the target file

    Returns the same fields as layer3_priority() plus agent_evidence.
    """
    from openai import OpenAI
    from core.layer3 import layer3_priority

    body     = request.get_json(force=True) or {}
    csv_arg  = (body.get("csv") or "").strip()
    row      = body.get("row") or {}

    if not row:
        return jsonify({"error": "No row data provided"}), 400

    csv_path = Path(csv_arg).expanduser() if csv_arg else _resolve_default_csv()

    try:
        app_cfg = _result_config(csv_path)
        project = get_project_config(app_cfg)
        proc    = get_processing_config(app_cfg)
        from core.api_connection import require_api_key
        load_dotenv(ROOT / ".env", override=True)
        api_key = require_api_key(proc.provider)
    except Exception as e:
        return jsonify({"error": f"Config error: {e}"}), 500

    client = OpenAI(
        api_key  = api_key,
        base_url = proc.base_url,
        timeout  = proc.api_timeout,
    )

    # Reconstruct layer1 / layer2 dicts from the CSV row
    layer1 = {
        "filename":           row.get("filename", ""),
        "format":             row.get("format", ""),
        "file_path":          row.get("file_path", ""),
        "folder":             row.get("folder", ""),
        "size_kb":            row.get("size_kb", ""),
        "page_count":         row.get("page_count", ""),
        "year_confidence":    row.get("year_confidence", ""),
        "extraction_method":  row.get("extraction_method", ""),
    }
    layer2 = {
        "domain":            row.get("domain", ""),
        "lifecycle":         row.get("lifecycle", ""),
        "asset_type":        row.get("asset_type", ""),
        "information_type":  row.get("information_type", ""),
        "governance":        row.get("governance", ""),
        "short_summary":     row.get("short_summary", ""),
        "keywords":          row.get("keywords", ""),
        "year":              row.get("year", ""),
    }
    content_sample = str(row.get("content_sample", ""))

    try:
        result = layer3_priority(
            layer1         = layer1,
            layer2         = layer2,
            project        = project,
            client         = client,
            model          = proc.model,
            api_timeout    = proc.api_timeout,
            content_sample = content_sample,
            use_agent      = True,
            csv_path       = csv_path,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Download CSV ──────────────────────────────────────────────────────
@app.route("/api/download-csv")
def download_csv():
    csv_arg = (request.args.get("csv") or "").strip()
    p = Path(csv_arg).expanduser() if csv_arg else _resolve_default_csv()
    if not p.exists():
        return jsonify({"error": f"{p.name} not found"}), 404
    df = _apply_reviews(pd.read_csv(p), p)
    return Response(df.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={p.name}"})


# ── API: Download filtered subset ──────────────────────────────────────────
@app.route("/api/download-filtered", methods=["POST"])
def download_filtered():
    """Accept a list of filenames to include, return a filtered CSV."""
    body = request.get_json(force=True) or {}
    csv_arg   = (body.get("csv") or "").strip()
    filenames = body.get("filenames") or []
    csv_path  = Path(csv_arg).expanduser() if csv_arg else _resolve_default_csv()
    if not csv_path.exists():
        return jsonify({"error": "CSV not found"}), 404
    df = _apply_reviews(pd.read_csv(csv_path), csv_path)
    if filenames and "filename" in df.columns:
        df = df[df["filename"].isin(filenames)]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=filtered_{csv_path.name}"},
    )


# ── API: Incremental diff ──────────────────────────────────────────────────
@app.route("/api/incremental-diff")
def incremental_diff():
    csv_arg  = (request.args.get("csv") or "").strip()
    csv_path = Path(csv_arg).expanduser() if csv_arg else _resolve_default_csv()
    if not csv_path.exists():
        return jsonify({"error": f"CSV not found: {csv_path}"}), 404

    try:
        df = pd.read_csv(csv_path, usecols=["file_path"])
    except Exception as e:
        return jsonify({"error": f"Cannot read CSV: {e}"}), 500

    paths = df["file_path"].dropna().astype(str).tolist()
    if not paths:
        return jsonify({"error": "No file_path entries in CSV"}), 400

    try:
        source_root = Path(_result_config(csv_path)["paths"]["input_dir"]).expanduser()
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400

    if not source_root.is_dir():
        return jsonify({"error": f"Source root not found: {source_root}"}), 404

    processed = set(paths)
    all_files = [
        str(p) for p in source_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_FORMATS
        and not p.name.startswith("~$")
        and not p.name.startswith(".")
    ]
    new_files = [f for f in all_files if f not in processed]

    return jsonify({
        "source_root": str(source_root),
        "processed":   len(processed),
        "new_count":   len(new_files),
        "new_files":   new_files[:20],
    })


# ── API: Incremental run (SSE) ─────────────────────────────────────────────
@app.route("/api/run-incremental", methods=["POST"])
def run_incremental():
    body     = request.get_json(force=True) or {}
    csv_arg  = (body.get("csv") or "").strip()
    csv_path = Path(csv_arg).expanduser() if csv_arg else _resolve_default_csv()

    if not csv_path.exists():
        return jsonify({"error": "CSV not found"}), 404

    try:
        app_cfg = _result_config(csv_path)
        project = get_project_config(app_cfg)
        proc    = get_processing_config(app_cfg)
        from core.api_connection import require_api_key
        load_dotenv(ROOT / ".env", override=True)
        api_key = require_api_key(proc.provider)
        config  = build_config(
            project     = project,
            model       = proc.model,
            api_key     = api_key,
            api_timeout = proc.api_timeout,
            base_url    = proc.base_url,
            delay       = proc.delay,
        )
    except Exception as e:
        return jsonify({"error": f"Config error: {e}"}), 500

    try:
        df = pd.read_csv(csv_path, usecols=["file_path"])
        paths       = df["file_path"].dropna().astype(str).tolist()
        source_root = Path(os.path.commonpath(paths))
        if source_root.is_file():
            source_root = source_root.parent
    except Exception as e:
        return jsonify({"error": f"Cannot infer source root: {e}"}), 400

    source_root = Path(app_cfg["paths"]["input_dir"]).expanduser()
    config["taxonomy"] = load_taxonomy(str(csv_path) + ".taxonomy.yaml")
    workers = int(body.get("workers", 3))
    q: queue.Queue = queue.Queue()

    def on_progress(i: int, total: int, row: dict) -> None:
        q.put({"type": "progress", "i": i, "total": total, "file": row.get("filename", "")})

    def _run() -> None:
        try:
            summary = pipeline_run(
                input_path  = source_root,
                output_csv  = csv_path,
                config      = config,
                on_progress = on_progress,
                parallel    = workers,
                incremental = True,
            )
            # Write taxonomy sidecar so dashboard always loads the exact
            # taxonomy that was active when this CSV was generated.
            try:
                sidecar = Path(str(csv_path) + ".taxonomy.yaml")
                import shutil
                sidecar.write_text(yaml.safe_dump(config["taxonomy"], allow_unicode=True), encoding="utf-8")
                _taxonomy_cache.pop(str(csv_path), None)   # invalidate cache
            except Exception:
                pass
            q.put({"type": "done", "added": summary.get("total", 0)})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})

    threading.Thread(target=_run, daemon=True).start()

    def generate():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        while True:
            try:
                event = q.get(timeout=180)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import socket
    import webbrowser

    def _free_port(start: int) -> int:
        for p in range(start, start + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    return p
        raise RuntimeError("No free port found in range 5174–5224")

    preferred = int(os.environ.get("DASHBOARD_PORT", 5174))
    port = _free_port(preferred)
    url  = f"http://localhost:{port}"
    print(f"DataTaxonomy Dashboard -> {url}")
    cleaned = _cleanup_orphan_sidecars()
    if cleaned:
        print(f"[startup] Removed {cleaned} orphaned taxonomy sidecar(s)")
    if not os.environ.get("NO_AUTO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
