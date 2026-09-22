# DataTaxonomy — Windows Deployment Package

DataTaxonomy is a local desktop application for classifying architectural and urban-design project files. It scans a folder containing PDFs, drawings, spreadsheets, models, images, and documents, then produces a structured CSV catalogue and a local review dashboard.

This package is intentionally distributed without project data, previous outputs, previous logs, saved profiles, or development history.

## What is included

- Local four-layer processing pipeline
- Configuration wizard
- Local dashboard and review queue
- Generic taxonomy template
- Windows installer script

## What is not included

- Any sample project files
- Previous CSV results or run reports
- Previous processing logs
- Saved project profiles
- API keys
- Git history or development settings

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- Internet access for package installation and the configured LLM provider
- An OpenRouter API key for the default configuration

## Installation

1. Extract this package to a local folder. Do not extract it into a client project folder.
2. Double-click `install.bat`.
3. Allow the installer to create a local Python virtual environment.
4. Enter the user's own OpenRouter API key when prompted.
5. After installation, start the application with:

```text
.venv\Scripts\python.exe main.py --dashboard-only
```

The installer creates a local `.env` and `config.yaml`. Both files are user-specific and must never be copied back into a release package.

## First-time setup

1. Open the configuration page at `http://localhost:5173`.
2. Enter the project name and project context.
3. Select the user's input folder.
4. Choose an output CSV name under `outputs/`.
5. Select or edit the taxonomy.
6. Start the pipeline.
7. Open the dashboard at `http://localhost:5174` when processing finishes.

For a small first run, use the sample option in the wizard or run:

```text
.venv\Scripts\python.exe main.py --sample 20
```

## Command-line usage

```text
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe main.py --sample 20
.venv\Scripts\python.exe main.py --incremental
.venv\Scripts\python.exe main.py --rerun certainty=Low
.venv\Scripts\python.exe main.py --no-dashboard
```

## Local files created after installation

These files are generated on the user's computer and are not part of this package:

```text
.env
config.yaml
outputs/
logs/
saved_configs.json
.reviewed_files.json
```

They may contain project names, file names, local paths, classification results, review edits, and processing diagnostics. Treat them as project data. Do not send them to the package maintainer unless the user has reviewed and approved their contents.

## Data and privacy boundary

The application servers bind to `127.0.0.1` and are intended for one local computer. The dashboard is not an authenticated multi-user service and should not be exposed to a LAN or the public internet.

File discovery and output writing happen locally. The configured LLM provider may receive information needed for classification, including:

- file names and folder context;
- extracted text and OCR text;
- table or metadata previews;
- project context entered in the wizard;
- visual previews of drawings, PDFs, and images when visual analysis is required.

Therefore this is a local application, but it is not an offline processing system. Before using it with confidential project material, confirm that the selected provider and account satisfy the user's data-processing requirements. For fully offline processing, a local text/vision model integration is required; this package does not provide that mode by default.

## Releasing a clean copy

Before sharing a new package, build it from an allowlist. Include only:

```text
core/
server/
static/
taxonomies/default.yaml
taxonomy.yaml
main.py
requirements.txt
install.bat
config.example.yaml
.env.example
README.md
```

Do not include:

```text
.git/
.claude/
.venv/
__pycache__/
.DS_Store
config.yaml
.env
outputs/
logs/
saved_configs.json
.reviewed_files.json
*.csv
*.bak
research notes, thesis files, or private project configurations
```

If an API key has ever appeared in a repository, settings file, command, or log, revoke it and create a new key before distributing any package. Removing the visible file does not remove the key from Git history or backups.

## Troubleshooting

- If the input path is invalid, reopen the configuration page and select the folder again.
- If scanned PDFs have no readable text, install Tesseract OCR and run the pipeline again.
- If the dashboard is unavailable, close stale Python processes and start `main.py --dashboard-only` again.
- If a provider rejects the selected model, choose a model supported by the configured base URL in the wizard or `config.yaml`.

## License and support

This deployment package is intended for authorized project use. Review the provider's terms and the project's information-governance requirements before processing project files.

## Vector Engine connection

Open **Configuration → AI Connection**, choose **Vector Engine**, enter the Base URL
from your provider console (preset: `https://api.vectorengine.ai/v1`), and copy the exact
model ID available to your token group. Click **Save & Verify Connection**, then save the project. The test sends a short chat request and may incur a small provider charge.
Use a chat-completions model; image analysis also requires vision support.

Keys are stored separately in `.env`: `VECTOR_ENGINE_API_KEY`, `OPENROUTER_API_KEY`,
`DEEPSEEK_API_KEY`, or `CUSTOM_API_KEY`. Switching providers does not reuse another
provider's key. Pipeline runs, dashboard reruns and URL autofill share this selection.
The model field in the project run panel accepts provider-specific model IDs.

```yaml
processing:
  provider: vectorengine
  base_url: https://api.vectorengine.ai/v1
  model: YOUR_MODEL_ID_FROM_CONSOLE
  api_timeout: 60
```

Base URLs can include `/v1` or `/v1/chat/completions`; the latter is normalized.
Existing configurations infer their provider from the endpoint until saved. A non-empty
`MODEL` environment variable overrides the CLI model; leave it blank for UI selection.
Restart the configuration and dashboard servers after upgrading.
`configuration.port` selects the project manager port; `dashboard.port` selects the results port.
Navigation between them uses these settings, so multiple checkouts can use different ports.

Provider integration reference: https://github.com/Archer-ai-hub/vectorengine-api-gateway

## Project workflow

- New projects start with blank project information and reuse the current AI connection.
- Save writes project settings and taxonomy together; Save & Run returns to the manager and starts processing.
- Existing projects allow direct section navigation and Save & Return.
- Running jobs use a configuration snapshot and remain visible after refreshing the page.
- Each result stores its run configuration, taxonomy, and file-path-based review records separately.
- Dashboard exports include saved review edits. Historical results without a configuration snapshot must be rerun from their project before AI reruns are available.
- Connection-loading failures disable setup instead of leaving editable default settings.
