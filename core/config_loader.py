"""config.py — Load configuration from YAML or environment variables."""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml
from .api_connection import connection_config


logger = logging.getLogger(__name__)


# ── Typed config objects ──────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    input_dir: Path
    output_csv: Path
    taxonomy_path: str = "taxonomy.yaml"


@dataclass
class ProcessingConfig:
    provider: str = "openrouter"
    model: str = "google/gemini-2.0-flash-001"
    sample_n: Optional[int] = None
    parallel_workers: int = 1
    api_timeout: int = 30
    base_url: str = "https://openrouter.ai/api/v1"
    delay: float = 0.3
    incremental: bool = False
    model_options: list = field(default_factory=list)
    data_formats: list = field(default_factory=list)
    layer2_settings: dict = field(default_factory=dict)


@dataclass
class DashboardConfig:
    port: int = 5174
    auto_launch: bool = True


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    cfg_file = Path(config_path)
    
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create {config_path} in the project root.\n"
            f"See config.yaml.example for template."
        )
    
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing {config_path}: {e}")
    
    return config or {}


def expand_path(path_str: str, base_dir: Path | None = None) -> Path:
    """Expand a user path, resolving relative paths from the app directory."""
    path = Path(str(path_str)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.absolute()


def get_project_config(config: dict) -> dict:
    """Extract and validate project configuration."""
    project = config.get("project", {})
    
    required = ["name", "location", "year_range"]
    for key in required:
        if key not in project:
            raise ValueError(f"Missing required project field: {key}")
    
    return project


def get_paths_config(config: dict) -> PathsConfig:
    """Extract and validate paths configuration."""
    paths = config.get("paths", {})
    _root = Path(__file__).parent.parent.resolve()
    input_raw = paths.get("input_dir", "~/Desktop")
    output_raw = paths.get("output_csv", "output.csv")
    taxonomy_raw = paths.get("taxonomy_path", "taxonomy.yaml")
    return PathsConfig(
        input_dir     = expand_path(input_raw, _root),
        output_csv   = expand_path(output_raw, _root),
        taxonomy_path = str(expand_path(taxonomy_raw, _root)),
    )


def get_processing_config(config: dict) -> ProcessingConfig:
    """Extract processing configuration."""
    proc = config.get("processing", {})
    connection = connection_config(proc)
    return ProcessingConfig(
        provider          = connection["provider"],
        model             = connection["model"],
        sample_n          = proc.get("sample_n", None),
        parallel_workers  = int(proc.get("parallel_workers", 1)),
        api_timeout       = int(proc.get("api_timeout", 30)),
        base_url          = connection["base_url"],
        delay             = float(proc.get("delay", 0.3)),
        incremental       = bool(proc.get("incremental", False)),
        model_options     = proc.get("model_options", [
            "google/gemini-2.0-flash-001",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
        ]),
        data_formats      = proc.get("data_formats", [
            ".csv", ".xlsx", ".xls", ".json", ".shp", ".geojson", ".kml", ".gpkg",
        ]),
        layer2_settings   = proc.get("layer2_settings", {}),
    )


def get_dashboard_config(config: dict) -> DashboardConfig:
    """Extract dashboard configuration."""
    dashboard = config.get("dashboard", {})
    return DashboardConfig(
        port        = dashboard.get("port", 5174),
        auto_launch = dashboard.get("auto_launch", True),
    )


# ── Taxonomy defaults (fallback when taxonomy.yaml not present) ───────────
_TAXONOMY_DEFAULTS: dict = {
    "domains": [
        {"name": "Architecture & Buildings", "description": "building design, plans, sections, elevations, facades, interior layouts"},
        {"name": "Landscape & Public Realm", "description": "landscape design, planting, hardscape, parks, streetscape, public realm"},
        {"name": "Structural Engineering",   "description": "foundations, load-bearing systems, steel/concrete/timber structural design"},
        {"name": "MEP - HVAC",               "description": "heating, ventilation, air conditioning, thermal systems and ducting"},
        {"name": "MEP - Plumbing",           "description": "water supply, drainage, sewage, stormwater and plumbing systems"},
        {"name": "MEP - Electrical",         "description": "power distribution, lighting, low-voltage systems and electrical infrastructure"},
        {"name": "Civil & Infrastructure",   "description": "utilities, roads, bridges, tunnels, grading, external infrastructure works"},
        {"name": "Mobility & Transport",     "description": "pedestrian networks, transit, traffic analysis, cycling routes, parking"},
        {"name": "Environment & Climate",    "description": "ecology, biodiversity, hydrology, sustainability, wind/noise studies"},
        {"name": "QS & Commercial",          "description": "BOQ, estimates, valuations, payments, quantity surveying and commercial controls"},
        {"name": "Administrative & Legal",   "description": "contracts, permits, approvals, legal correspondence and compliance docs"},
        {"name": "Project Management",       "description": "schedules, RFIs, transmittals, meeting records and internal coordination"},
        {"name": "Reference & Research",     "description": "standards, precedents, benchmark studies and background research"},
        {"name": "Unknown",                  "description": "cannot determine from available information"},
    ],
    "scales": [
        {"name": "Component / Room",        "description": "interior details, room-level layouts, component-level drawings"},
        {"name": "Floor / Level",           "description": "single floor plans, level coordination and floor-specific documentation"},
        {"name": "Plot / Block",            "description": "parcel/red-line scope, block-level planning and design packages"},
        {"name": "Neighborhood / District", "description": "urban block, district, zone or precinct scale"},
        {"name": "City / Municipal",        "description": "city-wide or municipal-scale planning and strategy"},
        {"name": "Metropolitan",            "description": "cross-city or metro-region planning and infrastructure scope"},
        {"name": "Regional / National",     "description": "regional, national or cross-boundary policy/infrastructure scale"},
        {"name": "Non-spatial",             "description": "no meaningful geographic scope"},
    ],
    "lifecycle_stages": [
        {"name": "Competition",                 "description": "design competition entries, competition boards, RFP responses and bid-stage submissions"},
        {"name": "Concept / Schematic",         "description": "early ideas, vision docs, feasibility studies, concept and SD-phase drawings"},
        {"name": "Design Development",          "description": "DD-phase developed layouts, coordinated discipline inputs and technical refinement"},
        {"name": "Construction Documents",      "description": "permit sets, issue-for-construction packages and final specifications"},
        {"name": "Construction Administration", "description": "construction-phase support — site instructions, RFIs, submittals and change orders"},
        {"name": "As-Built / Handover",         "description": "final built condition, as-built records and handover submissions"},
        {"name": "Unknown",                     "description": "cannot determine from available information"},
    ],
    "confidentiality_levels": [
        {"name": "Confidential", "description": "contracts, fee proposals, budgets, cost plans, legal agreements, NDAs, invoices, financial models, HR files"},
        {"name": "Not Confidential", "description": "all files without explicit legal/commercial/financial/HR sensitivity, including technical drawings and ordinary internal documents"},
    ],
    "asset_types": [
        {"name": "Data",      "description": "structured or semi-structured datasets (tables, spreadsheets, GIS, surveys, inventories)"},
        {"name": "Document",  "description": "textual deliverables and reports (PDF/Word/PPT, memos, specs, meeting notes)"},
        {"name": "Drawing",   "description": "CAD/BIM design sheets and plans (DWG/DXF/IFC/RVT)"},
        {"name": "Model",     "description": "BIM/federated models, 3D model exchanges, point clouds and model coordination assets"},
        {"name": "Calculation", "description": "engineering calculations, simulation sheets, thermal/structural computation artifacts"},
        {"name": "Statutory", "description": "regulatory submissions, permits, authority approvals and compliance certificates"},
        {"name": "Media",     "description": "imagery and audiovisual files (photos, renderings, videos)"},
        {"name": "Archive",   "description": "compressed containers and bundled packages (ZIP/7Z/RAR)"},
        {"name": "Unknown",   "description": "insufficient evidence to determine asset type"},
    ],
    # Fixed built-in vocabulary used by Layer 2 classification (the form of the content);
    # mirrored here for the dashboard review-queue dropdown, not read by the classifier.
    "information_types": [
        {"name": "Schematic / Technical", "description": "drawings, diagrams, CAD/BIM and technical content"},
        {"name": "Quantitative / Tabular", "description": "tables, spreadsheets, calculations, structured data"},
        {"name": "Narrative / Textual", "description": "reports, notes, specifications, correspondence"},
        {"name": "Spatial / Cartographic", "description": "GIS, maps, geographic data"},
        {"name": "Visual / Media", "description": "photos, renders, images, video"},
        {"name": "Archive", "description": "compressed or bundled packages"},
        {"name": "Unknown", "description": "cannot determine"},
    ],
    "governance_sources": [
        {"name": "Internal", "description": "Generated internally by lead firm or core team"},
        {"name": "External", "description": "Sourced from external consultants or third parties"},
        {"name": "Consultant", "description": "Deliverables from design/engineering consultants"},
        {"name": "Partner", "description": "Work from development partners or joint venture partners"},
        {"name": "Authority", "description": "Submissions from regulatory/authority sources"},
        {"name": "Vendor", "description": "Supplier/contractor documentation"},
        {"name": "Unknown", "description": "Source cannot be determined from available evidence"},
    ],
}


def load_taxonomy(taxonomy_path: str = "taxonomy.yaml") -> dict:
    """Load classification taxonomy from YAML. Falls back to built-in defaults if file not found."""
    tax_file = Path(taxonomy_path)
    if not tax_file.exists():
        return _TAXONOMY_DEFAULTS.copy()
    try:
        with open(tax_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if data else _TAXONOMY_DEFAULTS.copy()
    except yaml.YAMLError as e:
        logger.error("Invalid taxonomy YAML at %s: %s", tax_file, e)
        raise ValueError(f"Error parsing taxonomy YAML ({taxonomy_path}): {e}") from e



def validate_input_path(input_dir: Path) -> bool:
    """Check if input directory exists and has files."""
    if not input_dir.exists():
        logger.warning("Input directory not found: %s", input_dir)
        return False

    files = list(input_dir.glob("*"))
    if not files:
        logger.warning("Input directory is empty: %s", input_dir)
        return False

    return True
