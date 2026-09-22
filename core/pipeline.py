import csv
import os
import time
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import pandas as pd

from .layer1 import layer1_technical, INFO_TYPE_MAP
from .layer2 import layer2_domain
from .layer3 import layer3_priority
from .layer4 import layer4_trust
from .project_intel import extract_project_intelligence
from .vector_store import VectorStore
from .output import FIELDNAMES, sanitize_row, write_rows, read_rows, read_processed_paths

# ── Supported file formats ────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    ".pdf", ".docx", ".doc", ".txt", ".pptx",
    ".xlsx", ".xls", ".csv", ".json",
    ".dwg", ".dxf", ".dwf", ".ifc", ".rvt", ".nwd",
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".mp4", ".mov",
    ".shp", ".geojson", ".kml", ".gpkg",
    ".eml", ".msg", ".zip", ".7z", ".rar",
    # Adobe design apps
    ".ai", ".psd", ".indd",
    # Other CAD / 3D
    ".dgn", ".3dm",
    # Parametric script / plain list
    ".gh", ".lst",
}

# ── Phase-2 agent thresholds ─────────────────────────────────────────────
_AGENT_AUTHORITY_LEVELS = ("Statutory", "Hard Constraint")  # authority levels that trigger agent re-assessment
_AGENT_MIN_ROWS         = 50   # CSV must have at least this many rows before agent runs



# ── Config loader — reads from env or explicit dict ───────────────────────
def build_config(
    project: dict,
    model: str,
    api_key: str,
    api_timeout: int = 30,
    base_url: str = "https://openrouter.ai/api/v1",
    delay: float = 0.3,
    layer2_settings: dict | None = None,
) -> dict:
    """
    Packages everything the pipeline needs into a single config dict.
    Use this instead of relying on notebook globals.

    project   — same dict as PROJECT in the notebook
    model     — OpenRouter model string
    api_key   — OpenRouter API key
    """
    p           = project
    consultants = ", ".join(p.get("consultants", []))
    authorities = ", ".join(p.get("authorities", []))
    yr          = p.get("year_range", [])
    year_str    = f"{yr[0]}–{yr[1]}" if len(yr) == 2 else "unknown"

    project_context = (
        f"Project: {p.get('name', 'Unknown')} | "
        f"Location: {p.get('location', 'Unknown')} | "
        f"Years: {year_str} | "
        f"Lead firm: {p.get('lead_firm', 'Unknown')} | "
        f"Consultants: {consultants} | "
        f"Authorities: {authorities} | "
        f"Drawing code format: {p.get('drawing_code', 'Unknown')} | "
        f"Notes: {p.get('notes', '')}"
    )

    return {
        "project":         project,
        "project_context": project_context,
        "model":           model,
        "api_key":         api_key,
        "base_url":        base_url,
        "api_timeout":     int(api_timeout),
        "temperature":     0,
        "delay":           float(delay),
        "layer2_settings": layer2_settings or {},
    }


_THREAD_LOCAL = threading.local()


def _get_thread_client(config: dict) -> OpenAI:
    """Create/reuse one OpenAI client per worker thread (safer than sharing one client across threads)."""
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=int(config.get("api_timeout", 30)),
        )
        _THREAD_LOCAL.client = client
    return client


# ── Vector store helpers ──────────────────────────────────────────────────

def _make_embed_text(l2: dict) -> str:
    """Build a short text string from L2 outputs to embed into the vector store."""
    return " ".join(filter(None, [
        l2.get("domain", ""),
        l2.get("lifecycle", ""),
        l2.get("asset_type", ""),
        l2.get("information_type", ""),
        l2.get("keywords", "") if isinstance(l2.get("keywords"), str)
            else ", ".join(l2.get("keywords", [])),
        (l2.get("short_summary") or "")[:300],
    ]))


def _make_vs_meta(l1: dict, l2: dict) -> dict:
    """Metadata stored per file in the vector store (what the LLM sees as search results)."""
    keywords = l2.get("keywords", "")
    if isinstance(keywords, list):
        keywords = ", ".join(keywords)
    return {
        "filename":      l1.get("filename", ""),
        "domain":        l2.get("domain", ""),
        "lifecycle":     l2.get("lifecycle", ""),
        "asset_type":    l2.get("asset_type", ""),
        "short_summary": (l2.get("short_summary") or "")[:200],
        "authority":     "",   # updated after L3 runs
    }


# ── L1+L2 sub-processor ───────────────────────────────────────────────────

def _run_l1l2(
    file_path: Path,
    client: OpenAI,
    config: dict,
    input_path: Optional[Path] = None,
) -> tuple[dict, dict]:
    """Run Layer 1 extraction + Layer 2 classification. Returns (l1, l2)."""
    resolved_input_path = input_path or config.get("input_path") or file_path.parent
    l1 = layer1_technical(
        file_path,
        client      = client,
        model       = config["model"],
        api_timeout = int(config.get("api_timeout", 30)),
    )

    folder_path    = str(file_path.parent)
    folder_results = config.get("_folder_results")
    folder_lock    = config.get("_folder_lock")

    if folder_lock and folder_results is not None:
        with folder_lock:
            siblings = list(folder_results.get(folder_path, []))
    else:
        siblings = []

    l2 = layer2_domain(
        meta            = l1,
        client          = client,
        model           = config["model"],
        input_path      = resolved_input_path,
        project_context = config["project_context"],
        temperature     = config["temperature"],
        taxonomy        = config.get("taxonomy"),
        settings        = config.get("layer2_settings", {}),
        api_timeout     = int(config.get("api_timeout", 30)),
        folder_context  = siblings or None,
    )

    if folder_lock and folder_results is not None:
        with folder_lock:
            if folder_path not in folder_results:
                folder_results[folder_path] = []
            folder_results[folder_path].append({
                "filename":  l1["filename"],
                "domain":    l2.get("domain",    "Unknown"),
                "lifecycle": l2.get("lifecycle", "Unknown"),
                "asset_type":l2.get("asset_type","Unknown"),
                "certainty": l2.get("certainty", "Low"),
            })

    return l1, l2


# ── L3+L4 sub-processor ───────────────────────────────────────────────────

def _run_l3l4(
    file_path: Path,
    l1: dict,
    l2: dict,
    client: OpenAI,
    config: dict,
    vector_store: Optional["VectorStore"] = None,
) -> dict:
    """Run Layer 3 priority + Layer 4 trust. Returns sanitized row dict."""
    # L3 gate (data-focused). Per the project's "Data Taxonomy" scope, decision
    # scoring applies to data that enters the decision discussion: structured data
    # (Data / Calculation / Statutory) AND unstructured data (Document — reports,
    # minutes, narrative findings). Drawings, models, and media are visual/spatial
    # artifacts, not data, and are excluded. The pre-screen content signal gates
    # quality consistently — replacing the former candidate-match logic that scored
    # narrative documents only when "data" happened to land in the candidate list,
    # leaving strategic Brief/Concept docs (visioning, workshop notes) unscored at
    # random. (`content_signal` is read from the pre-screen's info_value_signal field.)
    _DATA_ASSETS    = ("data", "calculation", "statutory", "document")
    _asset          = str(l2.get("asset_type", "")).strip().lower()
    _content_signal = str((l2.get("_prescreen") or {}).get("info_value_signal", "Low")).strip().lower()
    _run_l3         = (_content_signal in ("high", "medium")) and (_asset in _DATA_ASSETS)
    if _run_l3:
        l3 = layer3_priority(
            layer1         = l1,
            layer2         = l2,
            project        = config["project"],
            client         = client,
            model          = config["model"],
            api_timeout    = int(config.get("api_timeout", 30)),
            content_sample = str(l1.get("content_sample", "")),
            project_intel  = config.get("project_intel"),
            vector_store   = vector_store,
        )
    else:
        l3 = {}

    l4 = layer4_trust(
        file_path = file_path,
        meta      = l1,
        layer2    = l2,
        project   = config["project"],
        settings  = config.get("layer2_settings"),
        layer3    = l3,
    )

    llm_raw    = str(l2.get("llm", ""))
    llm_status = llm_raw if llm_raw.startswith("error") else ""
    prescreen  = l2.get("_prescreen") or {}

    row = {
        "filename":            l1["filename"],
        "format":              l1["format"],
        "file_path":           l1["file_path"],
        "size_kb":             l1["size_kb"],
        "page_count":          l1.get("page_count", ""),
        "folder":              l1.get("folder", ""),
        "extraction_coverage": l1.get("extraction_coverage", ""),
        "extraction_method":   l1.get("extraction_method", ""),
        "year_confidence":     l2.get("year_confidence") or l1.get("year_confidence", "unknown"),
        "drawing_mode":        l1.get("drawing_mode", ""),
        "is_data_asset":       l4.get("is_data_asset", "No"),
        "info_value_hint":     str(prescreen.get("info_value_signal", "")).strip() or "Low",
        "information_type":    l2.get("information_type", l1.get("format_type", "Unknown")),
        "year":                l2.get("year") or l1.get("year"),
        "domain":              l2.get("domain",    "Unknown"),
        "domain_candidates":   ", ".join(l2.get("domain_candidates", [])) if isinstance(l2.get("domain_candidates", []), list) else str(l2.get("domain_candidates", "")),
        "scale":               l2.get("scale",     "Unknown"),
        "lifecycle":           l2.get("lifecycle", "Unknown"),
        "asset_type":            l4.get("asset_type", l2.get("asset_type", "Unknown")),
        "asset_type_candidates": ", ".join(l2.get("asset_type_candidates", [])) if isinstance(l2.get("asset_type_candidates", []), list) else str(l2.get("asset_type_candidates", "")),
        "short_summary":       l2.get("short_summary", ""),
        "keywords":            ", ".join(l2.get("keywords", [])) if isinstance(l2.get("keywords", []), list) else str(l2.get("keywords", "")),
        "governance":          l4["governance"],
        "confidentiality":        l4["confidentiality"],
        "confidentiality_reason": l4.get("confidentiality_reason", ""),
        "certainty":           l2.get("certainty", l2.get("confidence", "Low")),
        "certainty_score":     l2.get("certainty_score", l2.get("confidence_score", 0)),
        "certainty_reason":    l2.get("certainty_reason", l2.get("confidence_reason", "")),
        "age_warning":         l4["age_warning"],
        "review_priority":     l4["review_priority"],
        "action":              l4["action"],
        "review_reasons":      l4["review_reasons"],
        "authority":                l3.get("authority", ""),
        "authority_reason":         l3.get("authority_reason", ""),
        "scope":                    l3.get("scope", ""),
        "scope_reason":             l3.get("scope_reason", ""),
        "urgency":                  l3.get("urgency", ""),
        "urgency_reason":           l3.get("urgency_reason", ""),
        "coverage":                 l3.get("coverage", ""),
        "coverage_reason":          l3.get("coverage_reason", ""),
        "accessibility":            l3.get("accessibility", ""),
        "accessibility_reason":     l3.get("accessibility_reason", ""),
        "decision_priority_reason": l3.get("decision_priority_reason", ""),
        "_reasoning":          l2.get("_reasoning", ""),
        "llm_status":          llm_status,
        "processed_at":        datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    return sanitize_row(row)


# ── Single file processor (backward-compat: no vector store) ─────────────
def process_file(file_path: Path, client: OpenAI, config: dict, input_path: Optional[Path] = None) -> dict:
    """
    Run L1+L2+L3+L4 on a single file. Returns a flat CSV-ready dict.
    Used by selective_rerun() and direct callers that don't have a vector store.
    The two-phase run() below calls _run_l1l2 / _run_l3l4 directly.
    """
    l1, l2 = _run_l1l2(file_path, client, config, input_path)
    return _run_l3l4(file_path, l1, l2, client, config, vector_store=None)


# ── Parallel file processor wrapper ───────────────────────────────────────
def process_file_safe(args: tuple) -> tuple:
    """
    Wrapper to process a file safely within a thread pool.
    Returns (file_path, result_row, error, is_success)
    """
    fp, config = args
    try:
        delay = float(config.get("delay", 0) or 0)
        if delay > 0:
            time.sleep(random.uniform(0, delay))
        client = _get_thread_client(config)
        row = process_file(fp, client, config)
        return (fp, row, None, True)
    except Exception as e:
        return (fp, None, str(e), False)


# ── Selective re-run helpers ──────────────────────────────────────────────

def _parse_filter(filter_spec: str):
    """
    Parse a filter spec string into a predicate function (row: dict) -> bool.

    Supported formats:
      "failed"              — rows where llm_status is not empty
      "certainty=Low"       — exact column match
      "domain=Unknown"      — exact column match
      "folder=<substring>"  — file_path or folder contains substring
    """
    spec = filter_spec.strip()

    if spec == "failed":
        return lambda row: bool(str(row.get("llm_status", "")).strip())

    if "=" in spec:
        col, val = spec.split("=", 1)
        col = col.strip()
        val = val.strip()
        if col == "folder":
            return lambda row, v=val: (
                v in str(row.get("file_path", ""))
                or v in str(row.get("folder", ""))
            )
        return lambda row, c=col, v=val: str(row.get(c, "")).strip() == v

    raise ValueError(
        f"Unknown filter spec: {spec!r}\n"
        "Valid formats: 'failed' | 'certainty=Low' | 'domain=Unknown' | 'folder=<path>'"
    )


def _preload_folder_results(rows: list[dict]) -> dict:
    """
    Build folder_results from existing CSV rows so that selective re-runs
    benefit from folder-sibling RAG context from previously classified files.
    Called before selective_rerun() starts processing.
    """
    folder_results: dict[str, list[dict]] = {}
    for row in rows:
        fp = str(row.get("file_path") or "").strip()
        if not fp:
            continue
        folder = str(Path(fp).parent)
        folder_results.setdefault(folder, []).append({
            "filename":   row.get("filename",  ""),
            "domain":     row.get("domain",    "Unknown"),
            "lifecycle":  row.get("lifecycle", "Unknown"),
            "asset_type": row.get("asset_type","Unknown"),
            "certainty":  row.get("certainty", "Low"),
        })
    return folder_results


def selective_rerun(
    output_csv: Path,
    filter_spec: str,
    input_path: Path,
    config: dict,
    parallel: int = 1,
    on_progress: Optional[callable] = None,
) -> dict:
    """
    Re-process files matching filter_spec from an existing output CSV.

    Steps:
      1. Read existing CSV → split into keep_rows / rerun_rows
      2. Pre-load keep_rows into folder_results (RAG context for re-run)
      3. Run pipeline on rerun files only
      4. Write keep_rows + new_rows back to output_csv (merged, header preserved)

    filter_spec: "failed" | "certainty=Low" | "domain=Unknown" | "folder=<substr>"
    """
    if not output_csv.exists():
        raise FileNotFoundError(
            f"No existing CSV at {output_csv}. Run a full pass first."
        )

    predicate = _parse_filter(filter_spec)

    keep_rows:   list[dict] = []
    rerun_paths: list[Path] = []
    missing      = 0

    for row in read_rows(output_csv):
        if predicate(row):
            fp = str(row.get("file_path", "")).strip()
            p  = Path(fp) if fp else None
            if p and p.exists():
                rerun_paths.append(p)
            else:
                missing += 1   # file moved/deleted — drop silently
        else:
            keep_rows.append(row)

    print(f"\n>> Selective re-run  filter='{filter_spec}'")
    print(f"   Keep  : {len(keep_rows):>5} rows  (unchanged)")
    print(f"   Re-run: {len(rerun_paths):>5} files")
    if missing:
        print(f"   Skipped {missing} matching rows (file no longer on disk)")

    if not rerun_paths:
        print("   [OK] Nothing to re-run.")
        return {"rerun": 0, "kept": len(keep_rows), "errors": 0, "duration_s": 0,
                "output_csv": str(output_csv)}

    config = {**config, "input_path": input_path}

    # Pre-load previous results so folder-sibling RAG has full context
    _folder_results = _preload_folder_results(keep_rows)
    _folder_lock    = threading.Lock()

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=int(config.get("api_timeout", 30)),
    )

    print(">> Scanning for project meeting / decision documents...")
    project_intel = extract_project_intelligence(
        input_path  = input_path,
        client      = client,
        model       = config["model"],
        api_timeout = int(config.get("api_timeout", 30)),
    )
    _src = project_intel.get("sources", [])
    if _src:
        print(f"   Found {len(_src)} file(s): {', '.join(_src[:3])}{'…' if len(_src) > 3 else ''}")
    else:
        print("   None found.")

    config = {
        **config,
        "project_intel":   project_intel,
        "_folder_results": _folder_results,
        "_folder_lock":    _folder_lock,
    }

    new_rows:    list[dict] = []
    error_count  = 0
    start_time   = time.time()

    print(f"\n{'='*65}")
    print(f"  Re-running {len(rerun_paths)} files  "
          f"({'PARALLEL ' + str(parallel) + ' workers' if parallel > 1 else 'SERIAL'})")
    print(f"{'='*65}\n")

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(process_file_safe, (fp, config)) for fp in rerun_paths]
            for i, future in enumerate(as_completed(futures), 1):
                elapsed  = time.time() - start_time
                eta_secs = int((elapsed / i) * (len(rerun_paths) - i)) if i > 1 else 0
                eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s" if i > 1 else "--"
                fp, row, error, ok = future.result()
                print(f"[{i:04d}/{len(rerun_paths)}] {fp.name:<50} ETA {eta_str} ", end="", flush=True)
                if ok:
                    new_rows.append(sanitize_row(row))
                    print(f"✓  {row.get('domain','')[:20]:<20} | {row.get('review_priority','')}")
                    if on_progress:
                        on_progress(i, len(rerun_paths), row)
                else:
                    print(f"✗  {error}")
                    error_count += 1
    else:
        for i, fp in enumerate(rerun_paths, 1):
            elapsed  = time.time() - start_time
            eta_secs = int((elapsed / i) * (len(rerun_paths) - i)) if i > 1 else 0
            eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s" if i > 1 else "--"
            print(f"[{i:04d}/{len(rerun_paths)}] {fp.name:<50} ETA {eta_str} ", end="", flush=True)
            try:
                row = process_file(fp, client, config)
                new_rows.append(sanitize_row(row))
                print(f"✓  {row.get('domain','')[:20]:<20} | {row.get('review_priority','')}")
                if on_progress:
                    on_progress(i, len(rerun_paths), row)
            except Exception as e:
                print(f"✗  {e}")
                error_count += 1
            time.sleep(float(config.get("delay", 0) or 0))

    total_time = int(time.time() - start_time)

    # Merge and write
    write_rows(keep_rows + new_rows, output_csv)

    print(f"\n[OK] Done -- {len(new_rows)} re-run, {error_count} errors, "
          f"{len(keep_rows)} kept unchanged")
    print(f"   Total rows: {len(keep_rows) + len(new_rows)}  |  "
          f"Time: {total_time//60}m{total_time%60:02d}s")
    print(f"   Output: {output_csv}")

    return {
        "rerun":      len(new_rows),
        "kept":       len(keep_rows),
        "errors":     error_count,
        "duration_s": total_time,
        "output_csv": str(output_csv),
    }


# ── Agent deep-assessment (high-impact file re-grading) ──────────────────
def _agent_phase(output_csv: Path, config: dict, client, vector_store: Optional["VectorStore"] = None) -> int:
    """Re-assess high-impact files with the L3 agent. Returns count of upgraded files."""
    try:
        df = pd.read_csv(output_csv, low_memory=False)
    except Exception:
        return 0

    if len(df) < _AGENT_MIN_ROWS:
        return 0

    auth     = df.get("authority", pd.Series(dtype=str)).astype(str)
    eligible = df[auth.isin(_AGENT_AUTHORITY_LEVELS)].index.tolist()
    if not eligible:
        return 0

    print(f"\n>> Agent deep-assessment: {len(eligible)} high-impact files "
          f"(authority in {'/'.join(_AGENT_AUTHORITY_LEVELS)})")

    upgraded = 0
    for idx in eligible:
        row      = df.loc[idx]
        filename = str(row.get("filename", ""))
        l1 = {
            "filename":   filename,
            "folder":     str(row.get("folder", "")),
            "format":     str(row.get("format", "")),
            "size_kb":    row.get("size_kb", ""),
            "page_count": row.get("page_count", ""),
        }
        l2 = {
            "domain":           str(row.get("domain", "")),
            "lifecycle":        str(row.get("lifecycle", "")),
            "asset_type":       str(row.get("asset_type", "")),
            "information_type": str(row.get("information_type", "")),
            "governance":       str(row.get("governance", "")),
            "short_summary":    str(row.get("short_summary", "")),
            "keywords":         str(row.get("keywords", "")),
        }
        try:
            result = layer3_priority(
                layer1       = l1,
                layer2       = l2,
                project      = config["project"],
                client       = client,
                model        = config["model"],
                api_timeout  = int(config.get("api_timeout", 60)),
                use_agent    = True,
                csv_path     = output_csv,
                vector_store = vector_store,
            )
        except Exception as e:
            print(f"   x {filename}: {e}")
            continue

        old_auth = str(row.get("authority", "?"))
        new_auth = str(result.get("authority", old_auth))
        for field in [
            "authority", "authority_reason",
            "scope", "scope_reason",
            "urgency", "urgency_reason",
            "coverage", "coverage_reason",
            "accessibility", "accessibility_reason",
            "decision_priority_reason",
        ]:
            if field in result:
                df.at[idx, field] = result[field]

        # Keep vector store metadata current with latest grade
        if vector_store is not None:
            vector_store.update(filename, {"authority": new_auth})

        upgraded += 1
        marker = "=" if new_auth == old_auth else "*"
        print(f"   [{upgraded:03d}/{len(eligible)}] {filename[:52]:<52} {old_auth} -> {new_auth} {marker}")

    if upgraded:
        df.to_csv(output_csv, index=False)
        print(f"   Agent phase complete -- {upgraded} files re-graded.")

    return upgraded


def _l1l2_safe(args: tuple) -> tuple:
    """Thread-pool wrapper for _run_l1l2. Returns (fp, l1, l2, error)."""
    fp, config = args
    try:
        delay = float(config.get("delay", 0) or 0)
        if delay > 0:
            time.sleep(random.uniform(0, delay))
        client = _get_thread_client(config)
        l1, l2 = _run_l1l2(fp, client, config)
        return (fp, l1, l2, None)
    except Exception as e:
        return (fp, None, None, str(e))


def _l3l4_safe(args: tuple) -> tuple:
    """Thread-pool wrapper for _run_l3l4. Returns (fp, row, error)."""
    fp, l1, l2, config, vector_store = args
    try:
        client = _get_thread_client(config)
        row = _run_l3l4(fp, l1, l2, client, config, vector_store=vector_store)
        return (fp, row, None)
    except Exception as e:
        return (fp, None, str(e))


# ── Batch runner ──────────────────────────────────────────────────────────
def run(
    input_path:  Path,
    output_csv:  Path,
    config:      dict,
    sample_n:    Optional[int] = None,
    on_progress: Optional[callable] = None,
    parallel:    int = 1,
    incremental: bool = False,
) -> dict:
    """
    Two-phase batch processor:

    Phase 1 — L1+L2 for all files (parallel).
               Builds a similarity index from L2 outputs.

    Phase 2 — L3+L4 for all files (serial when parallel=1, else parallel
               with the same worker count as Phase 1 — see the parallel>1
               branch below; production runs use parallel=10).
               Each L3 call searches the full vector store for similar
               already-classified files; its result is written to CSV and
               its score is stored back into the vector store for later files.

    on_progress(i, total, row) — optional callback for progress updates.
    parallel  — number of parallel workers for both Phase 1 and Phase 2.
    incremental — if True, skip already-processed files and append new rows.
    """
    config = {**config, "input_path": input_path}

    # ── Collect files ─────────────────────────────────────────────────────
    all_files = [
        p for p in input_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_FORMATS
        and not p.name.startswith("~$")
        and not p.name.startswith(".")
    ]

    if incremental and output_csv.exists():
        processed_paths = read_processed_paths(output_csv)
        files = [f for f in all_files if str(f) not in processed_paths]
        if sample_n and sample_n < len(files):
            files = random.sample(files, sample_n)
            scope_label = f"INCREMENTAL SAMPLE {sample_n} new (of {len(all_files)} total)"
        else:
            scope_label = f"INCREMENTAL {len(files)} new (of {len(all_files)} total)"
    elif sample_n and sample_n < len(all_files):
        files       = random.sample(all_files, sample_n)
        scope_label = f"SAMPLE {sample_n} of {len(all_files)}"
    else:
        files       = all_files
        scope_label = f"ALL {len(all_files)}"

    if incremental and not files:
        print("\n[OK] All files already processed -- nothing to add.\n")
        return {"scope": scope_label, "total": 0, "errors": 0, "llm_failures": 0,
                "risk_high": 0, "risk_medium": 0, "risk_low": 0,
                "duration_s": 0, "output_csv": str(output_csv)}

    # ── LLM client ────────────────────────────────────────────────────────
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=int(config.get("api_timeout", 30)),
    )

    # ── Project Intelligence (L3 urgency context) ─────────────────────────
    print(">> Scanning for project meeting / decision documents...")
    project_intel = extract_project_intelligence(
        input_path  = input_path,
        client      = client,
        model       = config["model"],
        api_timeout = int(config.get("api_timeout", 30)),
    )
    _src = project_intel.get("sources", [])
    if _src:
        print(f"   Found {len(_src)} meeting/decision file(s): "
              f"{', '.join(_src[:3])}{'...' if len(_src) > 3 else ''}")
    else:
        print("   None found -- urgency scored from classification signals only.")

    # ── Shared state ──────────────────────────────────────────────────────
    _folder_results: dict = {}
    _folder_lock    = threading.Lock()

    config = {
        **config,
        "project_intel":   project_intel,
        "_folder_results": _folder_results,
        "_folder_lock":    _folder_lock,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"""
=================================================================
  Urban Asset Classifier
  Input:  {input_path}
  Scope:  {scope_label}
  Output: {output_csv}
  Phase 1 workers: {parallel}
=================================================================
""")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 1 — L1 + L2 for all files (parallel)
    # ═════════════════════════════════════════════════════════════════════
    print(f"-- Phase 1/2: L1+L2 extraction & classification ({len(files)} files) --\n")

    l1l2_results: dict[Path, tuple[dict, dict]] = {}
    p1_errors    = 0
    p1_error_log: list[dict] = []
    p1_start     = time.time()

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_l1l2_safe, (fp, config)): fp
                for fp in files
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                fp, l1, l2, err = future.result()
                elapsed  = time.time() - p1_start
                eta_secs = int((elapsed / done) * (len(files) - done)) if done > 1 else 0
                eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s" if done > 1 else "--"
                print(f"  L2 [{done:04d}/{len(files)}] {fp.name:<50} ETA {eta_str} ", end="", flush=True)
                if err:
                    print(f"x  {err}")
                    p1_errors += 1
                    p1_error_log.append({"phase": 1, "filename": fp.name, "file_path": str(fp), "error": err})
                else:
                    l1l2_results[fp] = (l1, l2)
                    print(f"ok  {str(l2.get('domain',''))[:20]:<20} | {l2.get('lifecycle','')}")
    else:
        for i, fp in enumerate(files, 1):
            elapsed  = time.time() - p1_start
            eta_secs = int((elapsed / i) * (len(files) - i)) if i > 1 else 0
            eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s" if i > 1 else "--"
            print(f"  L2 [{i:04d}/{len(files)}] {fp.name:<50} ETA {eta_str} ", end="", flush=True)
            try:
                l1, l2 = _run_l1l2(fp, client, config)
                l1l2_results[fp] = (l1, l2)
                print(f"ok  {str(l2.get('domain',''))[:20]:<20} | {l2.get('lifecycle','')}")
            except Exception as e:
                print(f"x  {e}")
                p1_errors += 1
                p1_error_log.append({"phase": 1, "filename": fp.name, "file_path": str(fp), "error": str(e)})
            time.sleep(float(config.get("delay", 0) or 0))

    p1_time = int(time.time() - p1_start)
    print(f"\n   Phase 1 done: {len(l1l2_results)} ok, {p1_errors} errors -- {p1_time//60}m{p1_time%60:02d}s")

    # ── Build vector store from all L2 results ────────────────────────────
    print("\n>> Building similarity index from L2 outputs...")
    vector_store = VectorStore()
    for fp, (l1, l2) in l1l2_results.items():
        vector_store.add(
            text = _make_embed_text(l2),
            meta = _make_vs_meta(l1, l2),
            key  = l1.get("filename", ""),
        )
    print(f"   {vector_store.size} files indexed (token overlap)")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 2 — L3 + L4 for all files (parallel if parallel>1, else serial).
    #           The vector store still grows in place either way; CSV writes
    #           are serialized via write_lock below.
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n-- Phase 2/2: L3+L4 priority & trust assessment ({len(l1l2_results)} files) --\n")

    csv_mode = "a" if (incremental and output_csv.exists()) else "w"
    ok_count = error_count = llm_fail_count = 0
    risk_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    p2_error_log: list[dict] = []
    p2_start = time.time()

    ordered_files = [fp for fp in files if fp in l1l2_results]
    write_lock = threading.Lock()

    with open(output_csv, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if csv_mode == "w":
            writer.writeheader()

        if parallel > 1:
            # ── PARALLEL Phase 2 ──────────────────────────────────────────
            p2_args = [
                (fp, l1l2_results[fp][0], l1l2_results[fp][1], config, vector_store)
                for fp in ordered_files
            ]
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(_l3l4_safe, a): a[0] for a in p2_args}
                done = 0
                for future in as_completed(futures):
                    done += 1
                    fp, row, err = future.result()
                    elapsed  = time.time() - p2_start
                    eta_secs = int((elapsed / done) * (len(ordered_files) - done)) if done > 1 else 0
                    eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s" if done > 1 else "--"
                    print(f"  L3 [{done:04d}/{len(ordered_files)}] {fp.name:<50} ETA {eta_str} ", end="", flush=True)
                    if err:
                        print(f"x  {err}")
                        error_count += 1
                        p2_error_log.append({"phase": 2, "filename": fp.name, "file_path": str(fp), "error": err})
                    else:
                        auth = row.get("authority", "")
                        if auth:
                            l1 = l1l2_results[fp][0]
                            vector_store.update(l1.get("filename", ""), {"authority": auth})
                        priority = row["review_priority"]
                        print(f"ok  {str(row.get('domain',''))[:20]:<20} | {priority}")
                        with write_lock:
                            writer.writerow(row)
                            f.flush()
                        ok_count += 1
                        risk_counts[priority] = risk_counts.get(priority, 0) + 1
                        if row.get("llm_status"):
                            llm_fail_count += 1
                        if on_progress:
                            on_progress(done, len(ordered_files), row)
        else:
            # ── SERIAL Phase 2 ────────────────────────────────────────────
            for i, fp in enumerate(ordered_files, 1):
                elapsed  = time.time() - p2_start
                eta_secs = int((elapsed / i) * (len(ordered_files) - i)) if i > 1 else 0
                eta_str  = f"{eta_secs//60}m{eta_secs%60:02d}s" if i > 1 else "--"
                print(f"  L3 [{i:04d}/{len(ordered_files)}] {fp.name:<50} ETA {eta_str} ", end="", flush=True)
                l1, l2 = l1l2_results[fp]
                try:
                    row = _run_l3l4(fp, l1, l2, client, config, vector_store=vector_store)
                    auth = row.get("authority", "")
                    if auth:
                        vector_store.update(l1.get("filename", ""), {"authority": auth})
                    writer.writerow(row)
                    f.flush()
                    priority = row["review_priority"]
                    print(f"ok  {str(row.get('domain',''))[:20]:<20} | {priority}")
                    ok_count += 1
                    risk_counts[priority] = risk_counts.get(priority, 0) + 1
                    if row.get("llm_status"):
                        llm_fail_count += 1
                    if on_progress:
                        on_progress(i, len(ordered_files), row)
                except Exception as e:
                    print(f"x  ERROR: {e}")
                    error_count += 1
                    p2_error_log.append({"phase": 2, "filename": fp.name, "file_path": str(fp), "error": str(e)})

    p2_time    = int(time.time() - p2_start)
    total_time = int(time.time() - start_time)

    summary = {
        "scope":          scope_label,
        "total":          ok_count,
        "errors":         error_count + p1_errors,
        "llm_failures":   llm_fail_count,
        "risk_critical":  risk_counts.get("Critical", 0),
        "risk_high":      risk_counts.get("High",     0),
        "risk_medium":    risk_counts.get("Medium",   0),
        "risk_low":       risk_counts.get("Low",      0),
        "phase1_time_s":  p1_time,
        "phase2_time_s":  p2_time,
        "duration_s":     total_time,
        "output_csv":     str(output_csv),
        "error_details":  p1_error_log + p2_error_log,
    }

    # ── Agent deep-assessment on high-impact files ───────────────────────
    agent_upgraded = 0
    if ok_count > 0:
        agent_upgraded = _agent_phase(output_csv, config, client, vector_store=vector_store)
    summary["agent_upgraded"] = agent_upgraded

    print(f"""
[OK] {ok_count} assets -> {output_csv}

=================================================================
  Scope           : {scope_label}
  Total processed : {ok_count}
  Errors          : {error_count + p1_errors}
  LLM failures    : {llm_fail_count}
  Agent upgraded  : {agent_upgraded}
  Risk -- Critical: {risk_counts.get("Critical", 0)}
  Risk -- High    : {risk_counts.get("High", 0)}
  Risk -- Medium  : {risk_counts.get("Medium", 0)}
  Risk -- Low     : {risk_counts.get("Low", 0)}
  Phase 1 time    : {p1_time//60}m{p1_time%60:02d}s
  Phase 2 time    : {p2_time//60}m{p2_time%60:02d}s
  Total time      : {total_time//60}m{total_time%60:02d}s
=================================================================
""")

    return summary
