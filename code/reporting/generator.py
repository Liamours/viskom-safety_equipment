"""
Rule-based report text generator.

Input  : FrameResult.to_dict() + context metadata
Output : ReportData dict consumed by renderer.py

Severity tiers (used to pick language intensity):
  CRITICAL — head protection (helmet)
  HIGH     — high-visibility vest, safety-vest
  MEDIUM   — gloves, face protection, hearing, eye protection
  LOW      — shoes, coverall, suit, tools
"""
from datetime import datetime
from typing import Dict, List, Optional

SEVERITY: Dict[str, str] = {
    "helmet":             "CRITICAL",
    "hard-hat":           "CRITICAL",
    "vest":               "HIGH",
    "safety-vest":        "HIGH",
    # Fix #12: coverall is primary biohazard barrier → HIGH (was LOW)
    "coverall":           "HIGH",
    # Fix #12: mask is respiratory protection → HIGH (was MEDIUM)
    "mask":               "HIGH",
    "gloves":             "MEDIUM",
    "face-guard":         "MEDIUM",
    "face_shield":        "MEDIUM",
    "face-mask-medical":  "MEDIUM",
    "goggles":            "MEDIUM",
    "glasses":            "MEDIUM",
    "earmuffs":           "MEDIUM",
    "ear":                "LOW",
    "shoes":              "LOW",
    "foot":               "LOW",
    "medical-suit":       "LOW",
    "safety-suit":        "LOW",
    "hands":              "LOW",
    "head":               "LOW",
    "face":               "LOW",
    "tools":              "LOW",
}

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

PPE_LABELS: Dict[str, str] = {
    "helmet":             "Safety Helmet",
    "hard-hat":           "Hard Hat",
    "vest":               "High-Visibility Vest",
    "safety-vest":        "Safety Vest",
    "gloves":             "Protective Gloves",
    "face-guard":         "Face Guard",
    "face_shield":        "Face Shield",
    "face-mask-medical":  "Medical Face Mask",
    "mask":               "Face Mask",
    "goggles":            "Safety Goggles",
    "glasses":            "Protective Glasses",
    "earmuffs":           "Ear Muffs",
    "ear":                "Ear Protection",
    "shoes":              "Safety Shoes",
    "foot":               "Foot Protection",
    "coverall":           "Coverall Suit",
    "medical-suit":       "Medical Suit",
    "safety-suit":        "Safety Suit",
    "hands":              "Hand Coverage",
    "head":               "Head Coverage",
    "face":               "Face Coverage",
    "tools":              "Tools",
}

CONTEXT_LABELS: Dict[str, str] = {
    "chv":              "Construction Site",
    "construction":     "Construction Site",
    "construction_full":"Construction Site (Full)",
    "medical":          "Medical / Healthcare",
    "cppe5":            "Medical / Healthcare",
    "sh17":             "Industrial Workplace",
}


def _label(ppe: str) -> str:
    return PPE_LABELS.get(ppe, ppe.replace("-", " ").replace("_", " ").title())


def _severity(ppe: str) -> str:
    return SEVERITY.get(ppe, "LOW")


def _sort_by_severity(items: List[str]) -> List[str]:
    return sorted(items, key=lambda x: SEVERITY_ORDER.get(_severity(x), 3))


def _person_narrative(person: dict, rule: str) -> str:
    pid      = person["person_id"] + 1
    worn     = person.get("worn", [])
    missing  = person.get("missing", [])
    score    = person.get("compliance_score", 1.0 if not missing else 0.0)
    compliant = person["compliant"]

    if compliant:
        worn_str = ", ".join(_label(w) for w in worn) if worn else "all required items"
        return f"Worker {pid} is fully compliant — {worn_str} confirmed."

    missing_sorted = _sort_by_severity(missing)
    missing_str    = ", ".join(_label(m) for m in missing_sorted)
    score_pct      = f"{score * 100:.0f}%"

    parts = [f"Worker {pid} ({score_pct} compliant) — missing: {missing_str}."]
    if worn:
        parts.append(f"Worn: {', '.join(_label(w) for w in worn)}.")
    return " ".join(parts)


def _overall_assessment(frame: dict, rule: str) -> str:
    n        = frame["num_persons"]
    compliant = frame["num_compliant"]
    rate      = frame.get("compliance_rate")
    context  = CONTEXT_LABELS.get(rule, "workplace")

    if n == 0:
        return "No workers detected — no PPE assessment possible."
    rate_str = f"{rate * 100:.0f}%" if rate is not None else "N/A"
    if compliant == n:
        return f"All {n} worker(s) fully compliant ({rate_str}) in {context} environment."
    violations = n - compliant
    return (
        f"{compliant}/{n} fully compliant, overall PPE score {rate_str} ({context}). "
        f"{violations} worker(s) need corrective action."
    )


def _action_items(frame: dict) -> List[str]:
    # Group by PPE type: { ppe -> [worker_ids] }
    ppe_workers: Dict[str, List[int]] = {}
    for p in frame.get("persons", []):
        pid = p["person_id"] + 1
        for m in p.get("missing", []):
            ppe_workers.setdefault(m, []).append(pid)

    items = []
    for ppe in _sort_by_severity(list(ppe_workers.keys())):
        workers = ppe_workers[ppe]
        sev     = _severity(ppe)
        prefix  = "[URGENT]" if sev in ("CRITICAL", "HIGH") else "[ADVISORY]"
        who     = ", ".join(f"Worker {w}" for w in sorted(workers))
        items.append(f"{prefix} Provide {_label(ppe)} — {who}")
    return items


def generate_report_data(
    frame_dict: dict,
    rule: str = "construction",
    dataset: str = "",
    image_path: str = "",
    inspector: str = "Automated Vision System",
) -> dict:
    """
    Convert a FrameResult.to_dict() payload into a structured report dict.

    Returns
    -------
    {
      "meta":        { timestamp, rule, context, dataset, image_path, inspector },
      "summary":     { num_persons, num_compliant, num_violations, compliance_rate },
      "persons":     [ { person_id, compliant, worn, missing, narrative } ],
      "assessment":  str,
      "action_items": [ str ],
    }
    """
    n         = frame_dict.get("num_persons", 0)
    compliant = frame_dict.get("num_compliant", 0)
    # Use partial compliance_rate from pipeline if available, else fall back to binary
    if "compliance_rate" in frame_dict:
        rate = frame_dict["compliance_rate"]
    else:
        rate = round(compliant / n, 4) if n > 0 else None

    persons_out = []
    for p in frame_dict.get("persons", []):
        persons_out.append({
            "person_id":        p["person_id"],
            "compliant":        p["compliant"],
            "compliance_score": p.get("compliance_score"),
            "worn":             p.get("worn", []),
            "missing":          p.get("missing", []),
            "person_bbox":      p.get("person_bbox", []),
            "narrative":        _person_narrative(p, rule),
        })

    return {
        "meta": {
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rule":        rule,
            "context":     CONTEXT_LABELS.get(rule, "Workplace"),
            "dataset":     dataset,
            "image_path":  image_path,
            "inspector":   inspector,
        },
        "summary": {
            "num_persons":    n,
            "num_compliant":  compliant,
            "num_violations": n - compliant,
            "compliance_rate": rate,
        },
        "persons":      persons_out,
        "assessment":   _overall_assessment(frame_dict, rule),
        "action_items": _action_items(frame_dict),
    }
