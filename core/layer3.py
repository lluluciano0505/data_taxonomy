"""Layer 3 — Design Decision Priority Assessment.

Measures how much a file would influence active design decisions if read now.
Not data quality — decision impact.

Five dimensions, each graded on a NAMED RUBRIC LEVEL (no numeric scores).
LLMs assign categorical levels with definitions far more reliably than they
produce calibrated numbers, and a level is an auditable claim ("this file is a
statutory constraint") where a number is not. The former 0–100 geometric-mean
composite was removed Jul 2026; its replicate noise (±2.4 mean, up to ±28 per
file) exceeded the differences it claimed to express.

  Ranked dimensions (drive the lexicographic ordering, strongest first):
    authority — Statutory > Hard Constraint > Client Brief > Advisory > Background
    scope     — Whole Project > Multi-Discipline > Single Discipline > Subsystem > Single Detail
    coverage  — Complete > Mostly Complete > Partial > Fragmentary > Stub

  Diagnostic only (recorded, NOT in the ranking):
    urgency       — Pivotal > Relevant > Marginal > Off-Phase (phase-fit heuristic;
                    cannot judge the project's live decision state, see paper §3.4/§6)
    accessibility — Direct Use > Stated Conclusions > Expert Extractable >
                    Specialist Only > Raw Data (threshold-of-use, feeds gap analysis)

Ranking: files sort lexicographically by (authority, scope, coverage) level —
see priority_sort_key(). There is no composite score.

Agent mode (use_agent=True):
  The LLM can call two tools before grading —
    search_similar_files  — find project files with similar keywords/domain
    get_folder_siblings   — list files in the same folder (version detection)
  Returns the same dict plus:
    agent_evidence  — list of {tool, args, result_summary} from tool calls
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .vector_store import VectorStore

logger = logging.getLogger(__name__)

_ClientT = Any
MAX_ROUNDS = 4
_TOP_K = 5


# ── Shared utilities ──────────────────────────────────────────────────────────

def _llm_call(
    client: _ClientT,
    model: str,
    system: str,
    user: str,
    temperature: float = 0,
    api_timeout: int = 30,
) -> Any:
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        timeout=api_timeout,
    )


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ── Rubric levels (each list ordered weakest → strongest) ────────────────────

LEVELS: dict[str, list[str]] = {
    "authority":     ["Background", "Advisory", "Client Brief", "Hard Constraint", "Statutory"],
    "scope":         ["Single Detail", "Subsystem", "Single Discipline", "Multi-Discipline", "Whole Project"],
    "coverage":      ["Stub", "Fragmentary", "Partial", "Mostly Complete", "Complete"],
    "urgency":       ["Off-Phase", "Marginal", "Relevant", "Pivotal"],
    "accessibility": ["Raw Data", "Specialist Only", "Expert Extractable", "Stated Conclusions", "Direct Use"],
}

RANKED_DIMS     = ("authority", "scope", "coverage")   # drive the lexicographic ordering
DIAGNOSTIC_DIMS = ("urgency", "accessibility")
_ALL_DIMS       = ("authority", "scope", "urgency", "coverage", "accessibility")

# Loose synonym needles for mapping off-vocabulary LLM output back to a level.
_LEVEL_SYNONYMS: dict[str, dict[str, tuple[str, ...]]] = {
    "authority": {
        "Statutory":       ("statut", "regulat", "mandate", "legal", "code requirement", "permit"),
        "Hard Constraint": ("hard", "technical constraint", "engineering", "physical"),
        "Client Brief":    ("client", "brief", "programme", "program"),
        "Advisory":        ("advis", "best practice", "guideline", "recommend"),
        "Background":      ("background", "reference", "informational"),
    },
    "scope": {
        "Whole Project":     ("whole", "entire", "master", "project-wide", "all disciplines"),
        "Multi-Discipline":  ("multi", "several disciplines", "cross-disc"),
        "Single Discipline": ("single disc", "one disc"),
        "Subsystem":         ("subsystem", "sub-system", "component family"),
        "Single Detail":     ("detail", "single element", "one element", "room", "product"),
    },
    "coverage": {
        "Complete":        ("comprehensive", "complete set", "full"),
        "Mostly Complete": ("mostly", "minor gap", "largely"),
        "Partial":         ("partial", "notable gap"),
        "Fragmentary":     ("fragment", "extract", "portion", "cover page"),
        "Stub":            ("stub", "empty", "placeholder", "template", "index sheet"),
    },
    "urgency": {
        "Pivotal":   ("pivotal", "critical", "window open", "irreplaceable", "stall"),
        "Relevant":  ("relevant", "directly useful", "useful now"),
        "Marginal":  ("marginal", "context"),
        "Off-Phase": ("off", "wrong phase", "closed", "not open", "decorative", "duplicate"),
    },
    "accessibility": {
        "Direct Use":         ("direct", "plain language", "universal", "anyone"),
        "Stated Conclusions": ("stated", "conclusion", "explicit", "skippable"),
        "Expert Extractable": ("extract", "embedded", "expertise needed", "buried"),
        "Specialist Only":    ("specialist", "domain expert"),
        "Raw Data":           ("raw", "no narrative", "unprocessed", "sensor"),
    },
}


def level_index(dim: str, value: str) -> int:
    """1-based ordinal of a level within its dimension (higher = stronger). 0 = unknown."""
    try:
        return LEVELS[dim].index(str(value)) + 1
    except (ValueError, KeyError):
        return 0


def normalize_level(dim: str, value: Any) -> str:
    """Map raw LLM output to a canonical level name. Falls back to the middle level."""
    levels = LEVELS[dim]
    v = str(value or "").strip()
    for name in levels:
        if v.lower() == name.lower():
            return name
    try:  # legacy numeric output (1–10) → proportional level
        n = float(v)
        if 1 <= n <= 10:
            return levels[round((n - 1) / 9 * (len(levels) - 1))]
    except ValueError:
        pass
    vl = v.lower()
    for name, needles in _LEVEL_SYNONYMS[dim].items():
        if any(nd in vl for nd in needles):
            return name
    return levels[len(levels) // 2]


def priority_sort_key(result: dict) -> tuple[int, int, int]:
    """Lexicographic ranking key: authority, then scope, then coverage.

    Higher tuples sort first. This replaces the former 0–100 composite: the
    ordering is the output, and no arithmetic is performed on the levels.
    """
    return tuple(level_index(d, result.get(d, "")) for d in RANKED_DIMS)


def _make_fallback(agent_mode: bool = False) -> dict:
    msg  = ("Agent failed" if agent_mode else "LLM failed") + " — mid-level grades assigned."
    base: dict = {}
    for dim in _ALL_DIMS:
        base[dim] = LEVELS[dim][len(LEVELS[dim]) // 2]
        base[f"{dim}_reason"] = "Assessment unavailable — defaulting to mid-level."
    base["decision_priority_reason"] = msg
    if agent_mode:
        base["agent_evidence"] = []
    return base


# ── Standard mode — single LLM call with full prompt ─────────────────────────

_IMPACT_PROMPT = """\
You are an AEC (Architecture, Engineering, Construction) senior data strategist.
Assess how much this file would influence active design decisions if a designer
read it right now — not data quality, but decision impact.

FILE:
  Filename        : {filename}
  Folder path     : {folder_path}
  Format          : {format}
  Year            : {year} ({year_confidence} confidence)
  Size            : {size_kb} KB
  Pages           : {page_count}

CLASSIFICATION:
  Domain          : {domain}
  Lifecycle Stage : {lifecycle}
  Asset Type      : {asset_type}
  Information Type: {information_type}
  Governance      : {governance}
  Summary         : {short_summary}
  Keywords        : {keywords}
{l2_reasoning_block}
FILE CONTENT EXCERPT:
{content_sample_block}
PROJECT CONTEXT:
  Project name    : {project_name}
  Project years   : {project_years}
  Current year    : {current_year}
  Lead firm       : {lead_firm}

────────────────────────────────────────────────────────────────────────────────

Grade the file on FIVE dimensions. For each dimension choose EXACTLY ONE named
level from its list (levels are ordered strongest → weakest), and justify the
choice in one sentence citing specific evidence. There is NO numeric score:
your level choices ARE the ranking.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. AUTHORITY  (ranked — sorts first)
   "Can this file force a redesign if its findings conflict with current design?"

   Statutory        — statutory / regulatory mandate: non-compliance = illegal
                      or unbuildable.
   Hard Constraint  — physical/engineering reality that cannot be negotiated
                      (bearing capacity, utility positions, structural load paths).
   Client Brief     — client / brief constraint: formally stated, changeable
                      via scope process.
   Advisory         — best-practice recommendation or guideline, overrideable
                      with justification.
   Background       — pure reference: ignoring it carries no design consequence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. SCOPE  (ranked — sorts second)
   "How much of the project does this file's content affect?"

   Whole Project     — entire building, all disciplines, or master plan.
   Multi-Discipline  — several disciplines or a major system strategy.
   Single Discipline — one full discipline, not the others.
   Subsystem         — a defined subsystem or component family.
   Single Detail     — one element, room, or product selection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT INTELLIGENCE  (extracted from recent meeting / decision documents)
{urgency_context_block}

  ↳ Use this to calibrate URGENCY: if this file directly addresses an open
    decision listed above, that is strong evidence for Pivotal or Relevant.
    If unrelated to all listed items, grade Marginal or Off-Phase.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. URGENCY  (diagnostic — recorded, NOT used for ranking)
   "How well does this file match what the team needs to decide RIGHT NOW
    in the '{lifecycle}' phase?"

   Open decision windows by phase:
   — Competition: competition strategy, parti, key concept moves, programme framing
   — Concept/Schematic: programme, massing, site strategy, structural strategy, system selection
   — Design Development: system coordination, key dimensions, material families
   — Construction Documents: full technical packages, specs, procurement
   — Construction Administration: site instructions, RFIs, change orders
   — As-Built/Handover: as-built records, handover packages, operational data

   Pivotal   — decision window is open NOW; missing this file would stall work.
   Relevant  — directly useful at this phase; other sources could partially substitute.
   Marginal  — useful context, but the phase's key decisions don't hinge on it.
   Off-Phase — window not yet open or already closed; no actionable output from
               reading now.

   ARCHIVE MODE: If project years {project_years} ended before {current_year},
   no live design windows exist. Reinterpret urgency as ARCHIVAL CENTRALITY:
     Pivotal   — irreplaceable record of a pivotal, site-specific decision or
                 constraint (authority approval, critical survey, defining brief);
                 losing it would leave the project record incomplete.
     Relevant  — standard project record; useful context.
     Marginal  — useful but easily reconstructed from other sources.
     Off-Phase — decorative or duplicate: stock imagery, sketch explorations,
                 generic reference, XREF backgrounds, texture files.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. COVERAGE  (ranked — sorts third)
   "How completely does this file document its own subject matter —
    independent of phase?"

   Complete        — nothing essential is missing (e.g. complete borehole log
                     set, full authority approval with all attachments,
                     complete as-built drawing package).
   Mostly Complete — minor gaps or references to external appendices, but the
                     core content is self-contained.
   Partial         — addresses the topic but with notable gaps; reader must
                     consult other documents to form a complete picture.
   Fragmentary     — only a portion of the subject is covered: a summary,
                     extract, or cover page pointing elsewhere.
   Stub            — almost no substantive content (blank template, index
                     sheet, placeholder file).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. ACCESSIBILITY  (diagnostic — recorded, NOT used for ranking)
   Answer TWO sub-questions together:
   (a) READABILITY — Are conclusions/findings stated explicitly? Clear legends, plain language?
   (b) AUDIENCE    — Can any project team member act on this, or must a domain expert interpret it?

   Direct Use         — plain language throughout; conclusions front-and-centre;
                        no specialist knowledge needed.
                        ✓ Meeting minutes with action items · Client approval letter
                        ✓ Executive summary · Public guidance document
   Stated Conclusions — conclusions stated clearly; supporting technical detail
                        present but skippable.
                        ✓ Strategy memo with explicit recommendations
                        ✓ Design report with a clear "we recommend…" section
   Expert Extractable — findings present but embedded in technical content;
                        expertise needed to extract them.
                        ✓ Engineering assessment without executive summary
                        ✓ Technical specification
   Specialist Only    — specialist-accessible only; non-specialists cannot act
                        on it without expert help.
                        ✓ Structural calculation package · Geotechnical test report
   Raw Data           — no conclusions, no narrative; only a domain expert can
                        interpret.
                        ✓ Sensor logs · Borehole core data · Native BIM/IFC geometry
                        ✗ Do NOT grade a written report Raw Data just because it
                          is technical.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RANKING
  Files are ordered lexicographically: AUTHORITY level first, then SCOPE, then
  COVERAGE. Urgency and accessibility are recorded as diagnostics only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CALIBRATION — READ BEFORE GRADING

  The middle level is the default for an ordinary file. Assign the top level
  only when specific evidence supports it, and use the bottom levels freely
  for files that genuinely fall short — they are expected and correct.

  Expected spread in a real 100-file AEC archive:
    Authority Statutory or Hard Constraint : ~10–15 files
    Authority Background                   : ~20–30 files
    Urgency Pivotal                        : ~10 files
    Urgency Off-Phase                      : ~30 files

  File-type anchors — override only with specific evidence:
    Texture / background / stock image  → Authority Background · Scope Single Detail · Urgency Off-Phase
    XREF or bind/reference CAD file     → Authority Advisory · Urgency Marginal
    Concept sketch / process diagram    → Authority Advisory · Scope Subsystem
    Regulatory approval / permit        → Authority Statutory
    Geotechnical / structural report    → Authority Hard Constraint
    Client brief / programme document   → Authority Client Brief · Scope Multi-Discipline or Whole Project
    Complete as-built drawing package   → Coverage Complete · Authority Hard Constraint or Client Brief

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

────────────────────────────────────────────────────────────────────────────────

Return ONLY valid JSON — no markdown, no commentary:

{{
  "authority":          "<Statutory | Hard Constraint | Client Brief | Advisory | Background>",
  "authority_reason":   "<one sentence: what makes it binding or advisory>",
  "scope":              "<Whole Project | Multi-Discipline | Single Discipline | Subsystem | Single Detail>",
  "scope_reason":       "<one sentence: how much of the project it affects>",
  "urgency":            "<Pivotal | Relevant | Marginal | Off-Phase>",
  "urgency_reason":     "<one sentence: why this is/isn't the right moment>",
  "coverage":             "<Complete | Mostly Complete | Partial | Fragmentary | Stub>",
  "coverage_reason":      "<one sentence: how complete the file's own content is>",
  "accessibility":        "<Direct Use | Stated Conclusions | Expert Extractable | Specialist Only | Raw Data>",
  "accessibility_reason": "<one sentence: readability + who can act on this without expert help>",
  "decision_priority_reason": "<2–3 sentences — BE SPECIFIC: (1) What specific design decision does this file directly answer — name the decision. (2) What unique information does it provide that no generic source substitutes. (3) Why NOW at the '{lifecycle}' phase is the right moment. Project-specific language, no generic filler.>"
}}
"""


def _similar_files_block(vector_store: "VectorStore", query: str, exclude_key: str) -> str:
    """Return a formatted prompt block listing similar project files."""
    similar = vector_store.search(query, k=5, exclude_key=exclude_key)
    if not similar:
        return ""
    lines = [
        "\n────────────────────────────────────────────────────────────────────────────────",
        "SIMILAR FILES ALREADY IN THIS PROJECT (similarity search — calibrate uniqueness):",
    ]
    for s in similar:
        auth_str = f" | Authority: {s['authority']}" if s.get("authority") else ""
        lines.append(
            f"  • {s.get('filename', '')} | {s.get('domain', '')} | "
            f"{s.get('lifecycle', '')}{auth_str}"
        )
        if s.get("short_summary"):
            lines.append(f"    ↳ {str(s['short_summary'])[:120]}")
    lines.append(
        "\nIf close duplicates exist, consider whether this file adds unique value "
        "or overlaps with already-available project knowledge."
    )
    return "\n".join(lines)


def _run_standard(
    layer1: dict,
    layer2: dict,
    project: dict,
    client: _ClientT,
    model: str,
    api_timeout: int,
    content_sample: str,
    project_intel: dict | None,
    vector_store: "VectorStore | None" = None,
) -> dict:
    filename = layer1.get("filename", "")
    fallback = _make_fallback(agent_mode=False)

    try:
        file_format     = layer1.get("format", "Unknown")
        year            = layer2.get("year") or layer1.get("year") or "Unknown"
        year_confidence = layer1.get("year_confidence", "unknown")
        size_kb         = layer1.get("size_kb", "?")
        page_count      = layer1.get("page_count") or "unknown"

        domain           = layer2.get("domain", "Unknown")
        lifecycle        = layer2.get("lifecycle", "Unknown")
        asset_type       = layer2.get("asset_type", "Unknown")
        information_type = layer2.get("information_type", "Unknown")
        governance       = layer2.get("governance", "Unknown")
        short_summary    = (layer2.get("short_summary") or "")[:500]
        keywords         = layer2.get("keywords", "")
        _l2_raw_reasoning = (layer2.get("_reasoning") or "").strip()
        l2_reasoning_block = (
            f"\nL2 CLASSIFICATION REASONING:\n  {_l2_raw_reasoning[:400]}\n"
            if _l2_raw_reasoning else ""
        )

        year_range    = project.get("year_range", [])
        project_years = (
            f"{year_range[0]}–{year_range[1]}"
            if isinstance(year_range, list) and len(year_range) == 2
            else "unknown"
        )
        current_year = datetime.now().year

        file_path_str = str(layer1.get("file_path", ""))
        folder_path = "/".join(file_path_str.replace("\\", "/").split("/")[-4:-1]) if file_path_str else "Unknown"

        _raw_sample = (content_sample or "").strip()
        # Page-marker prefixes like [p1] or [p12][OCR] are real extracted content.
        # All other [... patterns are layer1 error/fallback markers → metadata-only.
        _is_page_content = bool(re.match(r'^\[p\d+\]', _raw_sample))
        _metadata_only = not _raw_sample or (_raw_sample.startswith("[") and not _is_page_content)
        # Content-rich = large extracted text OR large file with many pages.
        _content_rich = not _metadata_only and (
            len(_raw_sample) > 3000 or float(size_kb or 0) > 500
        )
        if not _metadata_only:
            _sample_limit = 12000 if _content_rich else 3000
            _truncated = _raw_sample[:_sample_limit]
            if len(_raw_sample) > _sample_limit:
                _truncated += "\n  … [truncated]"
            _label = f"first ~{_sample_limit} chars"
            content_sample_block = f"  ({_label} of extracted text)\n  ───\n{_truncated}\n  ───\n\n"
        else:
            content_sample_block = "  [No text extracted — scoring based on metadata only]\n\n"

        from .project_intel import format_for_prompt as _fmt_intel

        # If no explicit open decisions but early phase, assume all decisions are open
        if project_intel and not project_intel.get('open_decisions') and lifecycle in ("Competition", "Concept / Schematic"):
            urgency_context_block = (
                "⚠️ No specific meeting/decision documents available. "
                "However, this early phase by definition has ALL design decisions open. "
                "Urgency should reflect: does this file directly address any major choices "
                "in programme, massing, site strategy, site organization, or key constraints?"
            )
        else:
            urgency_context_block = _fmt_intel(project_intel)

        prompt = _IMPACT_PROMPT.format(
            filename              = filename,
            folder_path           = folder_path,
            format                = file_format,
            year                  = year,
            year_confidence       = year_confidence,
            size_kb               = size_kb,
            page_count            = page_count,
            domain                = domain,
            lifecycle             = lifecycle,
            asset_type            = asset_type,
            information_type      = information_type,
            governance            = governance,
            short_summary         = short_summary,
            keywords              = keywords,
            l2_reasoning_block    = l2_reasoning_block,
            content_sample_block  = content_sample_block,
            project_name          = project.get("name", "Unknown"),
            lead_firm             = project.get("lead_firm", "Unknown"),
            project_years         = project_years,
            current_year          = current_year,
            urgency_context_block = urgency_context_block,
        )

        if _metadata_only:
            prompt += (
                "\n⚠ METADATA-ONLY MODE: No file content was extractable. "
                "Base your scores ONLY on filename, folder path, and the Layer 2 classification above. "
                "In `decision_priority_reason`, begin with: "
                "\"Assessment based on filename and classification only — no content extracted.\" "
                "Do NOT invent specific design decisions, reference document contents, or use "
                "phrases like \"avoid project delays\" or \"not available in generic sources\".\n"
            )
        elif _content_rich:
            prompt += (
                "\n📄 CONTENT-RICH FILE: Substantial text is available above. "
                "For EACH dimension reason, write 2–3 sentences that cite specific elements "
                "from the content — named entities, quantities, clause references, section "
                "titles, stated constraints, or commitments. Generic observations like "
                "'this document is relevant to the project' are not acceptable. "
                "In `decision_priority_reason`, write 4–5 sentences: name the specific design "
                "decision(s) this file answers, quote or paraphrase a key fact or constraint "
                "directly from the content, explain what substitute documents could not provide, "
                "and state why the current lifecycle phase makes this file particularly critical.\n"
            )

        if vector_store is not None and vector_store.size > 0:
            vs_query = f"{domain} {lifecycle} {asset_type} {keywords}"
            prompt += _similar_files_block(vector_store, vs_query, exclude_key=filename)

        resp   = _llm_call(
            client, model,
            system="You are an AEC design decision analyst. Respond ONLY with valid JSON.",
            user=prompt,
            temperature=0,
            api_timeout=api_timeout,
        )
        result = _parse_json(resp.choices[0].message.content or "")

        for dim in _ALL_DIMS:
            result[dim] = normalize_level(dim, result.get(dim, fallback[dim]))
            result.setdefault(f"{dim}_reason", fallback[f"{dim}_reason"])
        result.setdefault("decision_priority_reason", fallback["decision_priority_reason"])
        return result

    except Exception as exc:
        logger.warning("layer3_priority failed for %s [%s: %s]", filename, type(exc).__name__, exc)
        return dict(fallback)


# ── Agent mode — tool-call loop ───────────────────────────────────────────────

_AGENT_SYSTEM = """\
You are an AEC (Architecture, Engineering, Construction) senior data strategist.
Your job is to assess how much a specific file would influence active design
decisions — not data quality, but DECISION IMPACT.

You have two tools:
  search_similar_files  — find project files covering similar topics
  get_folder_siblings   — list files in the same folder (version detection)

STRATEGY:
1. Call search_similar_files with the file's keywords to check for duplicates.
2. Call get_folder_siblings to check if a newer version exists.
3. Use what you learned to grade the five dimensions accurately.

Grade each dimension by choosing EXACTLY ONE named level (no numbers).
Files are ranked lexicographically: authority level first, then scope, then
coverage; urgency and accessibility are recorded as diagnostics only.

CALIBRATION: the middle level is the default. Assign top levels only when
actively justified; use bottom levels freely for decorative, duplicate, or
off-phase files.
File anchors: texture/stock image → Authority Background, Urgency Off-Phase;
regulatory permit → Authority Statutory; geotech/structural report →
Authority Hard Constraint; XREF/bind CAD → Authority Advisory.

Return ONLY valid JSON (no markdown):

{
  "authority":          "<Statutory | Hard Constraint | Client Brief | Advisory | Background>",
  "authority_reason":   "<one sentence>",
  "scope":              "<Whole Project | Multi-Discipline | Single Discipline | Subsystem | Single Detail>",
  "scope_reason":       "<one sentence>",
  "urgency":            "<Pivotal | Relevant | Marginal | Off-Phase>",
  "urgency_reason":     "<one sentence>",
  "coverage":             "<Complete | Mostly Complete | Partial | Fragmentary | Stub>",
  "coverage_reason":      "<one sentence>",
  "accessibility":        "<Direct Use | Stated Conclusions | Expert Extractable | Specialist Only | Raw Data>",
  "accessibility_reason": "<one sentence>",
  "decision_priority_reason": "<2-3 sentences with project-specific evidence from tool results>"
}

LEVEL DEFINITIONS:
authority     — Statutory=regulatory mandate; Hard Constraint=non-negotiable engineering
                reality; Client Brief=formally stated, changeable via scope; Advisory=
                best practice; Background=pure reference
scope         — Whole Project; Multi-Discipline; Single Discipline; Subsystem; Single Detail
urgency       — Pivotal=window open NOW, missing this stalls work; Relevant=directly useful
                now; Marginal=useful context; Off-Phase=window closed or not open
coverage      — Complete; Mostly Complete; Partial; Fragmentary; Stub=near-empty
accessibility — Direct Use=plain language, anyone can act; Stated Conclusions=explicit,
                detail skippable; Expert Extractable=findings buried in technical content;
                Specialist Only=domain expert needed; Raw Data=no narrative, no conclusions
"""

_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_files",
            "description": (
                "Search the project file database for files that cover similar topics, "
                "domains, or keywords. Use this to assess whether the target file's "
                "information already exists elsewhere in the project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Comma-separated keywords or phrases to search for."},
                    "domain":   {"type": "string", "description": "Optional: filter results to this domain only."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_folder_siblings",
            "description": (
                "List all other files in the same folder as the target file. "
                "Use this to detect version history (v1/v2/draft/final patterns)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "The folder path to search (partial match is fine)."},
                },
                "required": ["folder"],
            },
        },
    },
]


def _tool_search_similar(df: Any, keywords: str, domain: str | None) -> list[dict]:
    if df.empty:
        return []
    query_tokens = set(re.split(r"[\s,;]+", keywords.lower()))
    query_tokens.discard("")

    def _score_row(row) -> float:
        text = " ".join([str(row.get("keywords", "")), str(row.get("short_summary", "")), str(row.get("domain", ""))]).lower()
        hits = sum(1 for t in query_tokens if t in text)
        return hits / max(len(query_tokens), 1)

    candidates = df.copy()
    if domain:
        mask = candidates["domain"].str.lower().str.contains(domain.lower(), na=False)
        if mask.sum() > 0:
            candidates = candidates[mask]

    candidates = candidates.copy()
    candidates["_sim"] = candidates.apply(_score_row, axis=1)
    top = candidates.nlargest(_TOP_K, "_sim")

    return [
        {
            "filename":      str(r.get("filename", "")),
            "domain":        str(r.get("domain", "")),
            "lifecycle":     str(r.get("lifecycle", "")),
            "asset_type":    str(r.get("asset_type", "")),
            "authority":     str(r.get("authority", "")),
            "short_summary": str(r.get("short_summary", ""))[:200],
        }
        for _, r in top.iterrows()
    ]


def _tool_folder_siblings(df: Any, folder: str) -> list[dict]:
    if df.empty:
        return []
    mask = df["folder"].str.lower().str.contains(folder.lower().strip("/"), na=False)
    return [
        {
            "filename":   str(r.get("filename", "")),
            "year":       str(r.get("year", "")),
            "asset_type": str(r.get("asset_type", "")),
            "lifecycle":  str(r.get("lifecycle", "")),
            "authority":  str(r.get("authority", "")),
        }
        for _, r in df[mask].head(20).iterrows()
    ]


def _execute_tool(name: str, args: dict, df: Any, vector_store: "VectorStore | None" = None) -> dict:
    if name == "search_similar_files":
        if vector_store is not None and vector_store.size > 0:
            query = f"{args.get('keywords', '')} {args.get('domain', '')}".strip()
            rows  = vector_store.search(query, k=_TOP_K)
        else:
            rows  = _tool_search_similar(df, args.get("keywords", ""), args.get("domain"))
        return {"results": rows, "count": len(rows)}
    if name == "get_folder_siblings":
        rows = _tool_folder_siblings(df, args.get("folder", ""))
        return {"results": rows, "count": len(rows)}
    return {"error": f"Unknown tool: {name}"}


def _build_agent_user_msg(layer1: dict, layer2: dict, project: dict, content_sample: str) -> str:
    yr = project.get("year_range", [])
    project_years = f"{yr[0]}–{yr[1]}" if isinstance(yr, list) and len(yr) == 2 else "unknown"
    _raw = (content_sample or "").strip()
    _is_page_content = bool(re.match(r'^\[p\d+\]', _raw))
    if _raw and (_is_page_content or not _raw.startswith("[")):
        excerpt = _raw[:3000]
        if len(_raw) > 3000:
            excerpt += "\n  … [truncated]"
        content_block = f"  ───\n{excerpt}\n  ───"
    else:
        content_block = "  [No text extracted — base assessment on metadata only]"

    return f"""\
FILE TO ASSESS:
  Filename        : {layer1.get('filename', '')}
  Folder          : {layer1.get('folder', '')}
  Format          : {layer1.get('format', '')}
  Year            : {layer2.get('year') or layer1.get('year', 'Unknown')}
  Size            : {layer1.get('size_kb', '?')} KB
  Pages           : {layer1.get('page_count', 'unknown')}

CLASSIFICATION:
  Domain          : {layer2.get('domain', 'Unknown')}
  Lifecycle Stage : {layer2.get('lifecycle', 'Unknown')}
  Asset Type      : {layer2.get('asset_type', 'Unknown')}
  Information Type: {layer2.get('information_type', 'Unknown')}
  Governance      : {layer2.get('governance', 'Unknown')}
  Summary         : {(layer2.get('short_summary') or '')[:400]}
  Keywords        : {layer2.get('keywords', '')}

FILE CONTENT EXCERPT:
{content_block}

PROJECT CONTEXT:
  Project         : {project.get('name', 'Unknown')}
  Years           : {project_years}
  Lead firm       : {project.get('lead_firm', 'Unknown')}

Use your tools first, then return the JSON score.
"""


def _run_agent(
    layer1: dict,
    layer2: dict,
    project: dict,
    csv_path: Path | str,
    client: _ClientT,
    model: str,
    api_timeout: int,
    content_sample: str,
    vector_store: "VectorStore | None" = None,
) -> dict:
    import pandas as pd

    filename = layer1.get("filename", "")
    fallback = _make_fallback(agent_mode=True)

    # If vector_store provided, skip CSV load — all search goes through the store.
    if vector_store is not None:
        df = pd.DataFrame()
    else:
        try:
            df = pd.read_csv(csv_path, low_memory=False) if Path(csv_path).exists() else pd.DataFrame()
            if not df.empty and "filename" in df.columns:
                df = df[df["filename"] != filename].reset_index(drop=True)
            for col in ("folder", "keywords", "short_summary"):
                if col not in df.columns:
                    df[col] = ""
        except Exception as e:
            logger.warning("layer3 agent: CSV load failed (%s) — continuing without search context", e)
            df = pd.DataFrame()

    evidence: list[dict] = []
    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user",   "content": _build_agent_user_msg(layer1, layer2, project, content_sample)},
    ]

    try:
        for _round in range(MAX_ROUNDS):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_AGENT_TOOLS,
                tool_choice="auto",
                temperature=0,
                timeout=api_timeout,
            )
            choice = resp.choices[0]
            messages.append(choice.message)

            if choice.finish_reason == "stop" or not getattr(choice.message, "tool_calls", None):
                break

            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_args = {}

                tool_result = _execute_tool(tool_name, tool_args, df, vector_store=vector_store)
                count = tool_result.get("count", 0)
                evidence.append({"tool": tool_name, "args": tool_args, "result_summary": f"{count} file(s) returned"})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result)})

        final_text = ""
        for msg in reversed(messages):
            role    = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if content:
                    final_text = content
                    break

        result = _parse_json(final_text)
        if not result:
            logger.warning("layer3 agent: no JSON in final response for %s", filename)
            fallback["agent_evidence"] = evidence
            return fallback

        for dim in _ALL_DIMS:
            result[dim] = normalize_level(dim, result.get(dim))
            result.setdefault(f"{dim}_reason", "Not assessed.")
        result.setdefault("decision_priority_reason", "")
        result["agent_evidence"] = evidence
        return result

    except Exception as exc:
        logger.warning("layer3 agent failed for %s [%s: %s]", filename, type(exc).__name__, exc)
        fallback["agent_evidence"] = evidence
        return fallback


# ── Public entry point ────────────────────────────────────────────────────────

def layer3_priority(
    layer1: dict,
    layer2: dict,
    project: dict,
    client: _ClientT,
    model: str,
    api_timeout: int = 30,
    content_sample: str = "",
    project_intel: dict | None = None,
    use_agent: bool = False,
    csv_path: "Path | str | None" = None,
    vector_store: "VectorStore | None" = None,
) -> dict:
    """
    Assess a file's design decision priority using five rubric-graded dimensions.

    use_agent=True  — agent mode: LLM calls search_similar_files /
                      get_folder_siblings before grading.
                      Prefers vector_store for search; falls back to csv_path.
    use_agent=False — standard mode: single LLM call.
                      If vector_store is provided, similar-file context is
                      appended to the prompt to help calibrate uniqueness.

    Returns dict with keys (each dimension holds a named level from LEVELS):
        authority, authority_reason,
        scope, scope_reason,
        urgency, urgency_reason,
        coverage, coverage_reason,
        accessibility, accessibility_reason,
        decision_priority_reason

    Files are ranked lexicographically via priority_sort_key(result).
    Agent mode also returns: agent_evidence.
    """
    if use_agent:
        if vector_store is None and csv_path is None:
            raise ValueError("csv_path or vector_store is required when use_agent=True")
        return _run_agent(
            layer1=layer1, layer2=layer2, project=project,
            csv_path=csv_path or "", client=client, model=model,
            api_timeout=api_timeout, content_sample=content_sample,
            vector_store=vector_store,
        )
    return _run_standard(
        layer1=layer1, layer2=layer2, project=project,
        client=client, model=model, api_timeout=api_timeout,
        content_sample=content_sample, project_intel=project_intel,
        vector_store=vector_store,
    )
