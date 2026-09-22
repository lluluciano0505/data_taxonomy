"""
Backend for config_wizard_prototype.html.
Serves the wizard and exposes API endpoints:
  GET  /                        — serve HTML
  GET  /load-config             — read config.yaml
  POST /save-config             — write config.yaml
  GET  /file-count?path=<dir>   — count files recursively
  POST /autofill-from-url       — scrape URL → LLM → project metadata
  GET  /taxonomy                — read taxonomy.yaml
  POST /save-taxonomy           — write taxonomy.yaml
  POST /run-pipeline            — start pipeline (SSE stream)
  GET  /pipeline-status         — is pipeline running?
  POST /stop-pipeline           — kill pipeline
  GET  /dashboard-status        — is dashboard running on port?
  POST /launch-dashboard        — start dashboard subprocess
  GET  /download-csv            — serve output CSV file
  GET  /api/outputs             — list CSV + sidecar pairs in outputs dir
  DELETE /api/outputs?name=X   — delete CSV and its sidecar together

Run:  python config_server.py
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests as http_requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv, set_key
from core.api_connection import PROVIDERS, connection_config, get_api_key, provider_for
from flask import Flask, Response, jsonify, request, send_file, stream_with_context, redirect

load_dotenv(_PROJECT_ROOT / ".env")

ROOT              = _PROJECT_ROOT
CONFIG_PATH       = ROOT / "config.yaml"
TAX_PATH          = ROOT / "taxonomy.yaml"
TAXONOMIES_DIR    = ROOT / "taxonomies"
SAVED_CONFIGS_PATH = ROOT / "saved_configs.json"

TAXONOMIES_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


def _connection(processing=None):
    if processing is None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
        processing = (cfg or {}).get("processing", {})
    return connection_config(processing)


@app.errorhandler(ValueError)
def invalid_config(error):
    return jsonify({"error": str(error)}), 400


# ── Pipeline process state ────────────────────────────────────────────────
_pipeline_proc: subprocess.Popen | None = None
_pipeline_lock = threading.Lock()


# ── Static ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(ROOT / "static" / "config_index.html")


@app.route("/results-ui")
def results_ui():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    port = int((cfg or {}).get("dashboard", {}).get("port", 5174))
    return redirect(f"http://localhost:{port}/")


@app.route("/wizard.html")
def wizard():
    return send_file(ROOT / "static" / "wizard.html")


# ── Config ────────────────────────────────────────────────────────────────
@app.route("/load-config")
def load_config():
    if not CONFIG_PATH.exists():
        return jsonify({}), 200
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return jsonify(yaml.safe_load(f) or {})


@app.route("/save-config", methods=["POST"])
def save_config():
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "empty body"}), 400

    existing = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    merged = deep_merge(existing, body)
    connection = _connection(merged.get("processing", {}))
    merged.setdefault("processing", {}).update(connection)
    with _project_lock:
        store = _load_saved_configs()
        current_id = merged.get("project", {}).get("id")
        for record in store.values():
            if current_id and record.get("config", {}).get("project", {}).get("id") == current_id:
                record["config"] = merged
        _commit_files({CONFIG_PATH: yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
                       SAVED_CONFIGS_PATH: json.dumps(store, ensure_ascii=False, indent=2)})
    return jsonify({"ok": True, "path": str(CONFIG_PATH)})


# Project saves include a taxonomy snapshot and replace, rather than merge, projects.
_project_lock = threading.RLock()


def _commit_files(files):
    previous = {p: p.read_bytes() if p.exists() else None for p in files}
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(content, encoding="utf-8")
            temp.replace(path)
    except Exception:
        for path, old in previous.items():
            if old is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(old)
        raise


def _activate_record(record, store):
    cfg = copy.deepcopy(record["config"])
    cfg.setdefault("processing", {}).update(_connection(cfg.get("processing", {})))
    project_id = cfg.setdefault("project", {}).setdefault("id", uuid.uuid4().hex)
    tax = record.get("taxonomy")
    if not isinstance(tax, dict) or not tax:
        raise ValueError("This project has no saved taxonomy. Open Settings and save it first.")
    tax_path = ROOT / "taxonomies" / (project_id + ".yaml")
    cfg.setdefault("paths", {})["taxonomy_path"] = str(tax_path)
    record["config"] = cfg
    _commit_files({CONFIG_PATH: yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                   tax_path: yaml.safe_dump(tax, allow_unicode=True, sort_keys=False),
                   TAX_PATH: yaml.safe_dump(tax, allow_unicode=True, sort_keys=False),
                   SAVED_CONFIGS_PATH: json.dumps(store, ensure_ascii=False, indent=2)})
    return cfg


@app.route("/save-project", methods=["POST"])
def save_project():
    body = request.get_json(force=True) or {}
    cfg = copy.deepcopy(body.get("config") or {})
    active = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    for section in ("configuration", "dashboard"):
        if section not in cfg and (active or {}).get(section):
            cfg[section] = copy.deepcopy(active[section])
    tax = body.get("taxonomy")
    project = cfg.setdefault("project", {})
    name = str(project.get("name", "")).strip()
    if not name or not project.get("location"):
        raise ValueError("Project name and location are required")
    if not isinstance(tax, dict) or not tax.get("domains"):
        raise ValueError("At least one taxonomy domain is required")
    paths = cfg.setdefault("paths", {})
    if not Path(paths.get("input_dir", "")).expanduser().is_dir():
        raise ValueError("Choose an existing input folder")
    if not paths.get("output_csv"):
        raise ValueError("Output filename is required")
    cfg.setdefault("processing", {}).update(_connection(cfg.get("processing", {})))
    if not cfg["processing"]["model"]:
        raise ValueError("Choose a model before saving")
    with _project_lock:
        store = _load_saved_configs()
        old_name = body.get("original_name")
        if name in store and name != old_name:
            return jsonify({"error": "A project with this name already exists. Choose another name."}), 409
        project["id"] = (store.get(old_name, {}).get("config", {}).get("project", {}).get("id") or uuid.uuid4().hex)
        # Prevent one project's run overwriting another project's output.
        out = (ROOT / paths["output_csv"]).resolve()
        for saved_name, record in store.items():
            other = record.get("config", {}).get("paths", {}).get("output_csv")
            if saved_name != old_name and other and (ROOT / other).resolve() == out:
                raise ValueError("This output filename belongs to another project. Choose a different name.")
        if old_name and old_name != name:
            store.pop(old_name, None)
        record = {"config": cfg, "taxonomy": tax, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        store[name] = record
        cfg = _activate_record(record, store)
    return jsonify({"ok": True, "name": name, "config": cfg})


@app.route("/activate-project", methods=["POST"])
def activate_project():
    name = (request.get_json(force=True) or {}).get("name")
    with _project_lock:
        store = _load_saved_configs()
        if name not in store:
            raise ValueError("Project not found")
        cfg = _activate_record(store[name], store)
    return jsonify({"ok": True, "config": cfg})


@app.route("/default-taxonomy")
def default_taxonomy():
    from core.config_loader import _TAXONOMY_DEFAULTS
    return jsonify(_TAXONOMY_DEFAULTS)


# ── Native folder picker ─────────────────────────────────────────────────
@app.route("/pick-folder")
def pick_folder():
    if sys.platform == "darwin":
        command = ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select project folder")']
    elif sys.platform == "win32":
        command = ["powershell", "-NoProfile", "-Command", "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}"]
    else:
        command = ["zenity", "--file-selection", "--directory"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return jsonify({"error": "Folder picker unavailable. Paste the folder path below."}), 400
    path = result.stdout.strip()
    return jsonify({"path": path}) if path else jsonify({"cancelled": True})


# ── File count ────────────────────────────────────────────────────────────
@app.route("/file-count")
def file_count():
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    p = Path(path).expanduser()
    if not p.is_dir():
        return jsonify({"error": "not a directory"}), 404
    count = sum(1 for _ in p.rglob("*") if _.is_file())
    return jsonify({"count": count, "path": str(p)})


# ── URL Autofill ──────────────────────────────────────────────────────────
_AUTOFILL_PROMPT = """\
You are a project metadata extractor for architectural / urban design projects.
I will give you the raw text scraped from a project webpage.
Extract the following fields and return ONLY valid JSON — no markdown, no explanation.

Fields (use null if not found):
{{
  "name":        string,
  "location":    string,
  "year_start":  integer or null,
  "year_end":    integer or null,
  "lead_firm":   string or null,
  "consultants": [string, ...],
  "authorities": [string, ...],
  "notes":       string or null
}}

--- WEBPAGE TEXT START ---
{text}
--- WEBPAGE TEXT END ---
"""


@app.route("/autofill-from-url", methods=["POST"])
def autofill_from_url():
    body = request.get_json(force=True) or {}
    url  = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url required"}), 400

    try:
        resp = http_requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Could not fetch URL: {e}"}), 400

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = re.sub(r"\s{3,}", "\n\n", soup.get_text(separator="\n"))[:6000]

    connection = _connection(body.get("processing"))
    api_key, source = get_api_key(connection["provider"])
    if not api_key:
        return jsonify({"error": f"{source} not set"}), 500

    try:
        llm_resp = http_requests.post(
            connection["base_url"] + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": connection["model"],
                "messages": [{"role": "user", "content": _AUTOFILL_PROMPT.format(text=text)}],
                "temperature": 0,
            },
            timeout=30,
        )
        llm_resp.raise_for_status()
    except http_requests.exceptions.HTTPError as e:
        return jsonify({"error": f"LLM API error ({e.response.status_code}): {e.response.text[:200]}"}), 500
    except Exception as e:
        return jsonify({"error": f"LLM request failed: {e}"}), 500

    raw = llm_resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return jsonify(json.loads(raw))
    except json.JSONDecodeError:
        return jsonify({"error": "LLM returned invalid JSON", "raw": raw}), 500


# ── Taxonomy ──────────────────────────────────────────────────────────────
@app.route("/taxonomy")
def get_taxonomy():
    if not TAX_PATH.exists():
        return jsonify({}), 200
    with open(TAX_PATH, "r", encoding="utf-8") as f:
        return jsonify(yaml.safe_load(f) or {})


@app.route("/save-taxonomy", methods=["POST"])
def save_taxonomy():
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "empty body"}), 400
    with open(TAX_PATH, "w", encoding="utf-8") as f:
        yaml.dump(body, f, allow_unicode=True, sort_keys=False)
    return jsonify({"ok": True})


# ── Taxonomy presets library (taxonomies/ folder) ─────────────────────────
@app.route("/list-taxonomies")
def list_taxonomies():
    files = sorted(p.stem for p in TAXONOMIES_DIR.glob("*.yaml") if not re.fullmatch(r"[0-9a-f]{32}", p.stem))
    return jsonify(files)


@app.route("/load-taxonomy/<name>")
def load_taxonomy_preset(name):
    path = TAXONOMIES_DIR / f"{name}.yaml"
    if not path.exists():
        return jsonify({"error": f"'{name}' not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Apply as active taxonomy
    with open(TAX_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    return jsonify(data)


@app.route("/save-taxonomy-as", methods=["POST"])
def save_taxonomy_as():
    body = request.get_json(force=True) or {}
    name = re.sub(r"[^\w\-]", "_", str(body.get("name", "")).strip())
    data = body.get("taxonomy")
    if not name or not data:
        return jsonify({"error": "name and taxonomy required"}), 400
    path = TAXONOMIES_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    return jsonify({"ok": True, "name": name})


@app.route("/delete-taxonomy/<name>", methods=["DELETE"])
def delete_taxonomy(name):
    path = TAXONOMIES_DIR / f"{name}.yaml"
    if not path.exists():
        return jsonify({"error": f"'{name}' not found"}), 404
    path.unlink()
    return jsonify({"ok": True})


@app.route("/read-taxonomy/<name>")
def read_taxonomy(name):
    """Return taxonomy data without setting it as the active taxonomy."""
    path = TAXONOMIES_DIR / f"{name}.yaml"
    if not path.exists():
        return jsonify({"error": f"'{name}' not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return jsonify(data)


@app.route("/taxonomy-editor")
def taxonomy_editor():
    return send_file(ROOT / "static" / "taxonomy_editor.html")


# ── Pipeline ──────────────────────────────────────────────────────────────
_run_state = {"running": False, "lines": [], "code": None, "output_csv": "", "project": "", "run_id": "", "stopped": False}


@app.route("/pipeline-status")
def pipeline_status():
    with _pipeline_lock:
        state = copy.deepcopy(_run_state)
    if not state["run_id"]:
        state_path = ROOT / "logs" / "last_pipeline_state.json"
        if state_path.exists():
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if not saved.get("running"):
                state = saved
    return jsonify(state)


@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    global _pipeline_proc
    from core.api_connection import require_api_key
    from core.config_loader import get_paths_config
    with _pipeline_lock:
        if _run_state["running"]:
            return jsonify({"error": "Pipeline already running"}), 409
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        connection = _connection(cfg.get("processing", {}))
        require_api_key(connection["provider"])
        if not connection["model"]:
            raise ValueError("Choose a model in connection settings")
        paths = get_paths_config(cfg)
        if not paths.input_dir.is_dir():
            raise ValueError("Input folder does not exist")
        run_id = uuid.uuid4().hex
        run_dir = ROOT / "logs" / "runs" / run_id
        run_dir.mkdir(parents=True)
        tax = yaml.safe_load(Path(paths.taxonomy_path).read_text(encoding="utf-8"))
        tax_path = run_dir / "taxonomy.yaml"
        tax_path.write_text(yaml.safe_dump(tax, allow_unicode=True), encoding="utf-8")
        cfg["paths"]["taxonomy_path"] = str(tax_path)
        cfg["paths"]["input_dir"] = str(paths.input_dir)
        cfg["paths"]["output_csv"] = str(paths.output_csv)
        run_path = run_dir / "config.yaml"
        run_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        env = os.environ.copy()
        env.pop("MODEL", None)  # UI model selection is authoritative for UI runs.
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        _pipeline_proc = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "main.py"), "--config", str(run_path), "--no-dashboard"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env,
            encoding="utf-8", errors="replace", bufsize=1)
        proc = _pipeline_proc
        _run_state.update(running=True, lines=[], code=None, output_csv=str(paths.output_csv),
                          project=cfg.get("project", {}).get("name", ""), run_id=run_id, stopped=False)

    def collect():
        for line in proc.stdout:
            with _pipeline_lock:
                _run_state["lines"].append(line.rstrip())
                _run_state["lines"] = _run_state["lines"][-500:]
        code = proc.wait()
        with _pipeline_lock:
            _run_state.update(running=False, code=code)
            state_path = ROOT / "logs" / "last_pipeline_state.json"
            state_path.write_text(json.dumps(_run_state, ensure_ascii=False), encoding="utf-8")
    threading.Thread(target=collect, daemon=True).start()
    return jsonify({"ok": True, "run_id": run_id})


@app.route("/stop-pipeline", methods=["POST"])
def stop_pipeline():
    with _pipeline_lock:
        if _pipeline_proc and _pipeline_proc.poll() is None:
            _run_state["stopped"] = True
            _pipeline_proc.terminate()
            return jsonify({"ok": True})
    return jsonify({"error": "No running pipeline"}), 409


# ── Dashboard ─────────────────────────────────────────────────────────────
def _is_port_open(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    result = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    return result


@app.route("/dashboard-status")
def dashboard_status():
    port = int(request.args.get("port", 5174))
    return jsonify({"running": _is_port_open(port), "port": port})


@app.route("/launch-dashboard", methods=["POST"])
def launch_dashboard():
    body = request.get_json(force=True) or {}
    port = int(body.get("port", 5174))
    if _is_port_open(port):
        return jsonify({"ok": True, "already": True, "url": f"http://localhost:{port}"})

    subprocess.Popen(
        [sys.executable, str(ROOT / "server" / "dashboard_server.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
        env={**os.environ, "DASHBOARD_PORT": str(port)},
    )
    return jsonify({"ok": True, "already": False, "url": f"http://localhost:{port}"})


# ── Download CSV ──────────────────────────────────────────────────────────
@app.route("/download-csv")
def download_csv():
    name = request.args.get("name", "")
    if not name:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            name = cfg.get("paths", {}).get("output_csv", "")
    if not name:
        return jsonify({"error": "no csv name"}), 400
    p = (ROOT / name).resolve()
    if not p.exists():
        return jsonify({"error": f"{name} not found"}), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="text/csv")


# ── Output files (CSV + sidecar management) ───────────────────────────────

def _outputs_dir() -> Path:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        rel = cfg.get("paths", {}).get("output_csv", "outputs/Master.csv")
        p = Path(rel)
        return (ROOT / p).resolve().parent if not p.is_absolute() else p.resolve().parent
    return (ROOT / "outputs").resolve()


@app.route("/api/outputs")
def list_outputs():
    d = _outputs_dir()
    if not d.exists():
        return jsonify([])
    results = []
    for csv in sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        sidecar = Path(str(csv) + ".taxonomy.yaml")
        results.append({
            "name":         csv.name,
            "path":         str(csv),
            "size_kb":      round(csv.stat().st_size / 1024, 1),
            "mtime":        csv.stat().st_mtime,
            "has_taxonomy": sidecar.exists(),
        })
    return jsonify(results)


@app.route("/api/outputs", methods=["DELETE"])
def delete_output():
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    d = _outputs_dir()
    csv_path = (d / name).resolve()
    # Prevent path traversal
    if not str(csv_path).startswith(str(d.resolve())):
        return jsonify({"error": "invalid path"}), 400
    if not csv_path.exists():
        return jsonify({"error": "not found"}), 404
    csv_path.unlink()
    sidecar = Path(str(csv_path) + ".taxonomy.yaml")
    sidecar_deleted = False
    if sidecar.exists():
        sidecar.unlink()
        sidecar_deleted = True
    return jsonify({"ok": True, "deleted": name, "sidecar_deleted": sidecar_deleted})


# ── Saved Configs ────────────────────────────────────────────────────────
def _load_saved_configs() -> dict:
    if not SAVED_CONFIGS_PATH.exists():
        return {}
    with open(SAVED_CONFIGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_saved_configs(data: dict):
    with open(SAVED_CONFIGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/import-config", methods=["POST"])
def import_config():
    body = request.get_json(force=True) or {}
    text = body.get("text", "")
    if not text:
        return jsonify({"error": "empty"}), 400
    try:
        cfg = yaml.safe_load(text)
        return jsonify(cfg or {})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/saved-configs")
def list_saved_configs():
    store = _load_saved_configs()
    items = [
        {"name": k, "timestamp": v.get("timestamp", ""), "project_name": v.get("config", {}).get("project", {}).get("name", "")}
        for k, v in store.items()
    ]
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(items)


@app.route("/saved-configs", methods=["POST"])
def save_config_as():
    body = request.get_json(force=True) or {}
    name   = (body.get("name") or "").strip()
    config = body.get("config")
    if not name:
        return jsonify({"error": "name required"}), 400
    if config is None:
        return jsonify({"error": "config required"}), 400
    # Snapshot the active taxonomy so the dashboard can restore it per-project
    taxonomy_snapshot = None
    if TAX_PATH.exists():
        with open(TAX_PATH, encoding="utf-8") as f:
            taxonomy_snapshot = yaml.safe_load(f)

    store = _load_saved_configs()
    from datetime import datetime, timezone
    store[name] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "config":   config,
        "taxonomy": taxonomy_snapshot,
    }
    _write_saved_configs(store)
    return jsonify({"ok": True})


@app.route("/saved-configs/<name>")
def get_saved_config(name):
    store = _load_saved_configs()
    if name not in store:
        return jsonify({"error": "not found"}), 404
    return jsonify(store[name])


@app.route("/saved-configs/<name>", methods=["DELETE"])
def delete_saved_config(name):
    store = _load_saved_configs()
    if name not in store:
        return jsonify({"error": "not found"}), 404
    del store[name]
    _write_saved_configs(store)
    return jsonify({"ok": True})


@app.route("/api/providers")
def api_providers():
    defaults = {'openrouter': 'amazon/nova-lite-v1', 'vectorengine': 'gpt-4o-mini'}
    return jsonify({name: {**PROVIDERS[name], 'model': model} for name, model in defaults.items()})


@app.route('/api/models', methods=['POST'])
def api_models():
    body = request.get_json(force=True) or {}
    provider = body.get('provider')
    if provider not in ('openrouter', 'vectorengine'):
        raise ValueError('Choose OpenRouter or Vector Engine')
    key = str(body.get('key') or '').strip() or get_api_key(provider)[0]
    if any(c.isspace() for c in key):
        raise ValueError('API key must not contain whitespace')
    if not key and provider == 'vectorengine':
        raise ValueError('Paste your Vector Engine API key to load models, then click Refresh models.')
    try:
        response = http_requests.get(
            PROVIDERS[provider]['base_url'] + '/models',
            headers={'Authorization': f'Bearer {key}'} if key else {},
            timeout=20, allow_redirects=False,
        )
        if response.status_code != 200:
            return jsonify({'error': f'Model list returned HTTP {response.status_code}. Check your API key and retry.'}), 400
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
            raise ValueError('Invalid model list')
        models = {}
        for item in payload['data']:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'].strip():
                continue
            architecture = item.get('architecture') or {}
            outputs = architecture.get('output_modalities') if isinstance(architecture, dict) else None
            if outputs and 'text' not in outputs:
                continue
            models[item['id']] = {'id': item['id'], 'name': item.get('name') or item['id']}
        if not models:
            raise ValueError('Empty model list')
    except (http_requests.RequestException, ValueError):
        return jsonify({'error': 'Could not load the model list. Check your connection and retry.'}), 400
    return jsonify({'models': sorted(models.values(), key=lambda model: model['id'].lower())})


@app.route('/api/recommended-models')
def recommended_models():
    provider = request.args.get('provider')
    if provider not in ('openrouter', 'vectorengine'):
        raise ValueError('Choose OpenRouter or Vector Engine')
    catalog = json.loads((_PROJECT_ROOT / 'core' / 'recommended_models.json').read_text(encoding='utf-8'))
    return jsonify(catalog[provider])


@app.route("/apikey-status")
def apikey_status():
    provider = request.args.get("provider") or _connection()["provider"]
    if provider not in PROVIDERS:
        raise ValueError("Unknown API provider")
    key, source = get_api_key(provider)
    return jsonify({"set": bool(key), "masked": "••••" + key[-4:] if key else "", "source": source})


@app.route("/save-apikey", methods=["POST"])
def save_apikey():
    body = request.get_json(force=True) or {}
    provider = body.get("provider") or _connection()["provider"]
    if provider not in PROVIDERS:
        raise ValueError("Unknown API provider")
    key = str(body.get("key") or "").strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("A non-empty API key without whitespace is required")
    source = PROVIDERS[provider]["key_env"]
    env_path = ROOT / ".env"
    env_path.touch(mode=0o600, exist_ok=True)
    set_key(str(env_path), source, key)
    os.environ[source] = key
    return jsonify({"ok": True})


@app.route("/test-connection", methods=["POST"])
def test_connection():
    body = request.get_json(force=True) or {}
    connection = _connection(body.get("processing"))
    entered_key = str(body.get("key") or "").strip()
    key, source = get_api_key(connection["provider"])
    key = entered_key or key
    if entered_key and any(c.isspace() for c in entered_key):
        raise ValueError("API key must not contain whitespace")
    if not key:
        raise ValueError(f"Save {source} first")
    if not connection["model"]:
        raise ValueError("Enter the model ID from your provider console")
    try:
        response = http_requests.post(
            connection["base_url"] + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": connection["model"], "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 16},
            timeout=30, allow_redirects=False,
        )
        if response.status_code != 200:
            return jsonify({"error": f"Provider returned HTTP {response.status_code}. Check endpoint, key permissions, model and balance."}), 400
        payload = response.json()
        if not payload.get("choices") or not isinstance(payload["choices"][0].get("message"), dict):
            raise ValueError("Provider did not return a chat completion")
    except http_requests.RequestException:
        return jsonify({"error": "Could not reach provider within 30 seconds. Check the endpoint and network."}), 400
    if entered_key:
        env_path = ROOT / ".env"
        env_path.touch(mode=0o600, exist_ok=True)
        set_key(str(env_path), PROVIDERS[connection["provider"]]["key_env"], entered_key)
        os.environ[PROVIDERS[connection["provider"]]["key_env"]] = entered_key
    return jsonify({"ok": True, "provider": connection["provider"], "model": connection["model"]})


if __name__ == "__main__":
    import webbrowser, threading
    startup_cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    port = int((startup_cfg or {}).get("configuration", {}).get("port", 5173))
    url = f"http://localhost:{port}"
    print(f"Config Wizard -> {url}")
    if not os.environ.get("NO_AUTO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False, use_reloader=False, threaded=True)
