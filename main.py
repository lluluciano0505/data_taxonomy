"""
main.py — All-in-one: Process Data → Dashboard
Configuration is read from config.yaml (no code edits needed!)

Run with:
    python main.py [--config config.yaml]
    python main.py --sample 20                    # process 20 random files (overrides config)
    python main.py --sample 0                     # process all files
    python main.py --incremental                  # skip already-processed files
    python main.py --rerun certainty=Low          # re-process low-certainty rows
    python main.py --rerun domain=Unknown         # re-process Unknown domain rows
    python main.py --rerun folder=Foundations     # re-process a specific folder
    python main.py --rerun failed                 # re-process LLM-failed rows
"""

import os
import sys
import json
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path

# Ensure the app root is on sys.path — required for embedded Python (installer)
# which does not automatically include the script's directory.
_APP_ROOT = Path(__file__).parent.resolve()
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
from dotenv import load_dotenv
import subprocess

# Windows cmd defaults to cp1252 which can't encode emoji — force UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.config_loader import (
    load_config, get_project_config, get_paths_config,
    get_processing_config, get_dashboard_config,
    validate_input_path, load_taxonomy,
    PathsConfig, ProcessingConfig, DashboardConfig,
)

# ── Load environment ────────────────────────────────────────────────────
load_dotenv(_APP_ROOT / ".env")

from core.api_connection import require_api_key


def _cleanup_old_logs(logs_dir: Path, keep_days: int = 30) -> None:
    """Delete log files older than keep_days days."""
    cutoff = datetime.now().timestamp() - keep_days * 86400
    for f in logs_dir.glob("run_*.log"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except OSError:
                pass


def _setup_logging(app_root: Path) -> Path:
    """Configure file + console logging. Returns the log file path."""
    logs_dir = app_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    _cleanup_old_logs(logs_dir)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{run_id}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — INFO+: captures all module-level logger calls
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler — WARNING+ only (print() handles user-facing output)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(ch)

    return log_path


def main():
    """Main entry point."""
    log_path = _setup_logging(_APP_ROOT)

    parser = argparse.ArgumentParser(description="DataTaxonomy Pipeline")
    parser.add_argument("--config", default=str(_APP_ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Number of parallel workers (1=serial, >1=parallel). 0 = use config value")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Run processing only, do not launch dashboard")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Launch dashboard and config UI without running the pipeline")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip already-processed files; append new results to existing CSV")
    parser.add_argument("--rerun", metavar="FILTER",
                        help=(
                            "Re-process a subset of an existing CSV matching FILTER. "
                            "Examples: 'certainty=Low'  'domain=Unknown'  "
                            "'folder=Foundations'  'failed'"
                        ))
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Override config sample_n: process N random files. 0 = all files.")
    args = parser.parse_args()

    # ── Dashboard-only shortcut ───────────────────────────────────────────
    if args.dashboard_only:
        config = load_config(args.config)
        dashboard_cfg = get_dashboard_config(config)
        try:
            launch_dashboard(dashboard_cfg)
        except KeyboardInterrupt:
            print("\n\nDashboard closed.")
        return

    # ── Load configuration ────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Configuration error: {e}")
        sys.exit(1)

    project       = get_project_config(config)
    paths         = get_paths_config(config)
    processing    = get_processing_config(config)
    API_KEY = require_api_key(processing.provider)
    if not processing.model:
        raise ValueError("Set a model ID for the selected API provider")
    dashboard_cfg = get_dashboard_config(config)
    age_analysis  = config.get("age_analysis", {})
    # Persist the exact non-secret configuration with this output for future reruns.
    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(json.dumps(config))
    snapshot["paths"]["input_dir"] = str(paths.input_dir)
    snapshot["paths"]["output_csv"] = str(paths.output_csv)
    Path(str(paths.output_csv) + ".config.yaml").write_text(
        __import__("yaml").safe_dump(snapshot, allow_unicode=True), encoding="utf-8")


    effective_parallel = (
        args.parallel if args.parallel and args.parallel > 0
        else processing.parallel_workers
    )

    # CLI --incremental overrides config; config incremental key also supported
    incremental = args.incremental or processing.incremental

    # CLI --sample overrides config sample_n (0 means process all)
    if args.sample is not None:
        processing.sample_n = args.sample if args.sample > 0 else None

    if os.getenv("MODEL", "").strip():
        processing.model = os.getenv("MODEL")

    # ── Validate input ────────────────────────────────────────────────────
    if not validate_input_path(paths.input_dir):
        print("[ERROR] Cannot proceed. Check config.yaml paths.input_dir")
        sys.exit(1)

    # ── Header ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  DataTaxonomy Pipeline")
    print("="*70)
    print(f"  Project : {project['name']} ({project['location']})")
    print(f"  Input   : {paths.input_dir.name}")

    # ── Mode: Selective re-run ────────────────────────────────────────────
    if args.rerun:
        print(f"  Mode: Selective re-run  (filter: {args.rerun})")
        print("="*70 + "\n")
        if not paths.output_csv.exists():
            print(f"[ERROR] No existing CSV at {paths.output_csv}. Run a full pass first.")
            sys.exit(1)
        ok = rerun_data(paths, project, processing, API_KEY, effective_parallel,
                        age_analysis, args.rerun)
        if not ok:
            sys.exit(1)

    # ── Mode: Full / Incremental run ──────────────────────────────────────
    else:
        mode_label = "INCREMENTAL (skip processed)" if incremental else (
            f"SAMPLE {processing.sample_n}" if processing.sample_n else "ALL FILES"
        )
        print(f"  Files to process : {mode_label}")
        print(f"  Parallel workers : {effective_parallel}")
        print("="*70 + "\n")

        if not process_data(paths, project, processing, API_KEY,
                            effective_parallel, age_analysis, incremental,
                            log_path=log_path):
            print("[WARN] Data processing failed. Skipping dashboard.")
            sys.exit(1)

    if args.no_dashboard:
        print("[OK] Processing finished (--no-dashboard).")
        return

    # ── Step 2: Launch Dashboard ──────────────────────────────────────────
    try:
        launch_dashboard(dashboard_cfg)
    except KeyboardInterrupt:
        print("\n\nDashboard closed.")


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_pipeline_config(project: dict, processing: ProcessingConfig, api_key: str,
                           age_analysis: dict | None, paths: PathsConfig) -> dict:
    """Build the config dict used by both full-run and selective re-run."""
    from core.pipeline import build_config
    project_for_pipeline = {**project, "age_analysis": age_analysis or {}}
    cfg = build_config(
        project         = project_for_pipeline,
        model           = processing.model,
        api_key         = api_key,
        api_timeout     = processing.api_timeout,
        base_url        = processing.base_url,
        delay           = processing.delay,
        layer2_settings = processing.layer2_settings,
    )
    cfg["taxonomy"] = load_taxonomy(paths.taxonomy_path)
    return cfg


def _write_sidecar(paths: PathsConfig) -> None:
    """Copy taxonomy file next to the output CSV so dashboard uses the right taxonomy."""
    taxonomy_src = Path(paths.taxonomy_path)
    sidecar      = Path(str(paths.output_csv) + ".taxonomy.yaml")
    if taxonomy_src.exists():
        shutil.copy2(taxonomy_src, sidecar)


def process_data(
    paths: PathsConfig,
    project: dict,
    processing: ProcessingConfig,
    api_key: str,
    parallel: int = 1,
    age_analysis: dict | None = None,
    incremental: bool = False,
    log_path: Path | None = None,
) -> bool:
    """Run the full (or incremental) pipeline and generate CSV."""
    from core.pipeline import run

    print("STEP 1: Processing Files...")
    print("-" * 70)

    cfg = _build_pipeline_config(project, processing, api_key, age_analysis, paths)
    started_at = datetime.now()

    try:
        summary = run(
            input_path  = paths.input_dir,
            output_csv  = paths.output_csv,
            config      = cfg,
            sample_n    = processing.sample_n,
            parallel    = parallel,
            incremental = incremental,
        )
        finished_at = datetime.now()
        print(f"\n[OK] Success! Results saved to: {paths.output_csv}")
        _write_sidecar(paths)
        _write_run_json(paths, project, processing, summary,
                        started_at, finished_at, log_path)
        return True

    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")
        logging.getLogger(__name__).exception("Pipeline run() raised an exception")
        return False


def _write_run_json(
    paths: PathsConfig,
    project: dict,
    processing: ProcessingConfig,
    summary: dict,
    started_at: datetime,
    finished_at: datetime,
    log_path: Path | None,
) -> None:
    """Write a structured run summary JSON alongside the CSV."""
    run_doc = {
        "run_id":       started_at.strftime("%Y%m%d_%H%M%S"),
        "started_at":   started_at.isoformat(timespec="seconds"),
        "finished_at":  finished_at.isoformat(timespec="seconds"),
        "project":      project.get("name", ""),
        "input_dir":    str(paths.input_dir),
        "output_csv":   str(paths.output_csv),
        "log_file":     str(log_path) if log_path else None,
        "model":        processing.model,
        "workers":      processing.parallel_workers,
        "stats": {
            "scope":         summary.get("scope", ""),
            "total":         summary.get("total", 0),
            "errors":        summary.get("errors", 0),
            "llm_failures":  summary.get("llm_failures", 0),
            "agent_upgraded":summary.get("agent_upgraded", 0),
            "risk_critical": summary.get("risk_critical", 0),
            "risk_high":     summary.get("risk_high", 0),
            "risk_medium":   summary.get("risk_medium", 0),
            "risk_low":      summary.get("risk_low", 0),
            "phase1_time_s": summary.get("phase1_time_s", 0),
            "phase2_time_s": summary.get("phase2_time_s", 0),
            "duration_s":    summary.get("duration_s", 0),
        },
        "errors": summary.get("error_details", []),
    }
    json_path = Path(str(paths.output_csv) + ".run.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_doc, f, indent=2, ensure_ascii=False)
    print(f"     Run log  : {log_path or 'n/a'}")
    print(f"     Run JSON : {json_path}\n")


def rerun_data(
    paths: PathsConfig,
    project: dict,
    processing: ProcessingConfig,
    api_key: str,
    parallel: int = 1,
    age_analysis: dict | None = None,
    filter_spec: str = "failed",
) -> bool:
    """Re-process a filtered subset of an existing CSV."""
    from core.pipeline import selective_rerun

    cfg = _build_pipeline_config(project, processing, api_key, age_analysis, paths)

    try:
        summary = selective_rerun(
            output_csv  = paths.output_csv,
            filter_spec = filter_spec,
            input_path  = paths.input_dir,
            config      = cfg,
            parallel    = parallel,
        )
        print(f"\n[OK] Re-run complete: {summary}")
        _write_sidecar(paths)
        return True

    except Exception as e:
        print(f"[ERROR] Error during re-run: {e}")
        return False


def launch_dashboard(dashboard_cfg: DashboardConfig) -> None:
    """Launch the dashboard server and config server."""
    port = dashboard_cfg.port
    print("=" * 70)
    print("STEP 2: Launching Dashboard...")
    print("=" * 70)
    print(f"  Dashboard  → http://localhost:{port}")
    print(f"  Config UI  → http://localhost:5173\n")
    env = {**os.environ, "DASHBOARD_PORT": str(port)}
    subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "server" / "dashboard_server.py")],
        env=env,
    )
    subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "server" / "config_server.py")],
    )


if __name__ == "__main__":
    main()
