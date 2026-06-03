"""
Export each figure panel as individual PNG.

Output: results/phase_2/components/<dataset>_<col>.png

Cols:
  01_raw          — raw image
  02_landmarks    — pose skeleton only, no text
  03_pact         — PPE boxes only, no text labels
  04_protocol     — company protocol (PNG)
  05_json         — PACT JSON (PNG)
  06_report       — compliance report (PNG)
"""
import json
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline
from landmarks.core import draw_pose_landmarks
from landmarks.base import COCO_CONNECTIONS
from reporting.generator import generate_report_data

OUT_DIR = ROOT / "results/phase_2/components"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PPE_COLOR = (255, 140, 0)   # orange boxes, no text

SAMPLES = [
    {
        "tag":     "sh17",
        "image":   ROOT / "dataset/SH17/pexels-photo-18110372.jpeg",
        "dataset": "sh17",
        "weights": ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
        "rule":    "sh17",
        "protocol": {
            "context":  "Industrial / Construction Workplace",
            "standard": "OSHA 29 CFR 1926 / ISO 45001",
            "required": [
                ("Safety Helmet",        "Head protection against falling objects"),
                ("Safety Vest (Hi-Vis)", "High-visibility for traffic/machinery zones"),
                ("Protective Gloves",    "Hand protection from sharp/hot surfaces"),
            ],
            "recommended": ["Safety Shoes / Boots", "Eye Protection", "Hearing Protection (Earmuffs)"],
        },
    },
    {
        "tag":     "cppe5",
        "image":   ROOT / "dataset/CPPE-5/130.png",
        "dataset": "cppe5",
        "weights": ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
        "rule":    "cppe5",
        "protocol": {
            "context":  "Healthcare / Clinical Environment",
            "standard": "WHO PPE Guidelines / CDC Infection Control",
            "required": [
                ("Coverall Suit",    "Full-body barrier against biohazards"),
                ("Face Mask",        "Respiratory droplet protection (N95/surgical)"),
                ("Protective Gloves","Barrier against pathogen contact"),
            ],
            "recommended": ["Face Shield / Goggles", "Boot Covers", "Hair Cover / Cap"],
        },
    },
    {
        "tag":     "chv",
        "image":   ROOT / "dataset/CHV/ppe_0387.jpg",
        "dataset": "chv",
        "weights": ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
        "rule":    "chv",
        "protocol": {
            "context":  "Construction / Building Site",
            "standard": "OSHA 1926.100 / EN 397 / ANSI 107-2020",
            "required": [
                ("Safety Helmet",        "Head protection — mandatory on all active sites"),
                ("High-Visibility Vest", "Worker visibility to vehicle operators"),
            ],
            "recommended": ["Safety Shoes / Steel-Toe Boots", "Protective Gloves", "Safety Glasses"],
        },
    },
]


# ── image panel helpers ───────────────────────────────────────────────────────

def save_img(arr_rgb: np.ndarray, path: Path):
    cv2.imwrite(str(path), cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    print(f"  saved: {path.name}")


def panel_raw(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb.copy()


def panel_landmarks(img_rgb: np.ndarray, frame_result) -> np.ndarray:
    canvas = img_rgb.copy()
    for person in frame_result.persons:
        if person.landmark_result is not None:
            canvas = draw_pose_landmarks(
                canvas, person.landmark_result,
                landmark_color=(0, 255, 128),
                connection_color=(255, 255, 255),
                landmark_radius=5,
                connection_thickness=2,
                connections=COCO_CONNECTIONS,
            )
    return canvas


def panel_pact(img_rgb: np.ndarray, frame_result) -> np.ndarray:
    canvas = img_rgb.copy()
    for person in frame_result.persons:
        # skeleton, no text
        if person.landmark_result is not None:
            canvas = draw_pose_landmarks(
                canvas, person.landmark_result,
                landmark_color=(0, 255, 128),
                connection_color=(255, 255, 255),
                landmark_radius=5,
                connection_thickness=2,
                connections=COCO_CONNECTIONS,
            )
        # PPE boxes only, no text labels
        for ppe in person.ppe_detections:
            x1, y1, x2, y2 = ppe["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), PPE_COLOR, 3)
    return canvas


# ── text panel helpers (matplotlib → PNG) ────────────────────────────────────

def save_text_panel(render_fn, path: Path, w_in=5, h_in=7, dpi=200):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor("#1a1a2e" if "protocol" in str(path) or "json" in str(path) else "#f4f6f8")
    render_fn(ax)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {path.name}")


def render_protocol(ax, protocol):
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y = 0.96
    ax.text(0.5, y, "COMPANY PROTOCOL", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#ffffff",
            fontfamily="monospace", transform=ax.transAxes)
    y -= 0.08
    ax.text(0.05, y, protocol["context"], ha="left", va="top",
            fontsize=9, color="#aab4c8", style="italic", transform=ax.transAxes)
    y -= 0.07
    ax.text(0.05, y, f"Standard: {protocol['standard']}", ha="left", va="top",
            fontsize=7.5, color="#7a8499", transform=ax.transAxes)
    y -= 0.09

    ax.text(0.05, y, "MANDATORY PPE", ha="left", va="top",
            fontsize=9.5, fontweight="bold", color="#ff6b6b", transform=ax.transAxes)
    y -= 0.07

    for name, desc in protocol["required"]:
        ax.text(0.06, y, f"▶  {name}", ha="left", va="top",
                fontsize=9, color="#ffffff", fontweight="bold", transform=ax.transAxes)
        y -= 0.07
        for line in textwrap.wrap(desc, 42):
            ax.text(0.10, y, line, ha="left", va="top",
                    fontsize=8, color="#aab4c8", transform=ax.transAxes)
            y -= 0.06
        y -= 0.01

    y -= 0.03
    ax.text(0.05, y, "RECOMMENDED", ha="left", va="top",
            fontsize=9, fontweight="bold", color="#ffd93d", transform=ax.transAxes)
    y -= 0.07
    for item in protocol["recommended"]:
        ax.text(0.06, y, f"○  {item}", ha="left", va="top",
                fontsize=8.5, color="#c8d0e0", transform=ax.transAxes)
        y -= 0.07


def render_json(ax, frame_dict):
    ax.set_facecolor("#0d1117"); ax.axis("off")
    compact = {
        "num_persons":   frame_dict["num_persons"],
        "num_compliant": frame_dict["num_compliant"],
        "persons": [
            {"id": p["person_id"], "compliant": p["compliant"],
             "worn": p["worn"], "missing": p["missing"]}
            for p in frame_dict["persons"]
        ],
    }
    lines = json.dumps(compact, indent=2).split("\n")
    y = 0.97; lh = 0.055
    ax.text(0.03, y, "// PACT JSON Output", ha="left", va="top",
            fontsize=7.5, color="#6a9955", fontfamily="monospace", transform=ax.transAxes)
    y -= lh
    for line in lines:
        if y < 0.02: break
        if any(k in line for k in ['"worn"', '"missing"', '"compliant"', '"num_']):
            color = "#9cdcfe"
        elif ': true' in line:  color = "#4ec9b0"
        elif ': false' in line: color = "#f44747"
        elif line.strip().startswith('"'): color = "#ce9178"
        else: color = "#d4d4d4"
        ax.text(0.03, y, line, ha="left", va="top",
                fontsize=7.5, color=color, fontfamily="monospace", transform=ax.transAxes)
        y -= lh


def render_report(ax, report_data):
    ax.set_facecolor("#f4f6f8"); ax.axis("off")
    meta    = report_data["meta"]
    summary = report_data["summary"]
    persons = report_data["persons"]
    n = summary["num_persons"]; ok = summary["num_compliant"]
    viol = summary["num_violations"]
    rate = summary["compliance_rate"]
    pct  = f"{rate*100:.0f}%" if rate is not None else "N/A"
    ok_c = "#2e7d32"; fail_c = "#b71c1c"; warn_c = "#e65100"
    main_c = ok_c if viol == 0 else (fail_c if ok == 0 else warn_c)

    y = 0.97
    ax.text(0.5, y, "COMPLIANCE REPORT", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#1a1a2e", transform=ax.transAxes)
    y -= 0.08
    ax.text(0.5, y, meta["context"], ha="center", va="top",
            fontsize=8, color="#555", style="italic", transform=ax.transAxes)
    y -= 0.08

    for label, val, col in [
        ("Workers Detected", str(n), "#333"),
        ("Compliant",        str(ok), ok_c),
        ("Violations",       str(viol), fail_c if viol > 0 else ok_c),
        ("Compliance Rate",  pct, main_c),
    ]:
        ax.text(0.06, y, label, ha="left", va="top",
                fontsize=8.5, color="#555", transform=ax.transAxes)
        ax.text(0.70, y, val, ha="left", va="top",
                fontsize=8.5, fontweight="bold", color=col, transform=ax.transAxes)
        y -= 0.07

    ax.axhline(y, color="#dde3ea", linewidth=0.8, xmin=0.03, xmax=0.97)
    y -= 0.05

    for p in persons:
        pid = p["person_id"] + 1
        comp = p["compliant"]
        badge_c = ok_c if comp else fail_c
        ax.text(0.06, y, f"Worker {pid}", ha="left", va="top",
                fontsize=9, fontweight="bold", color="#1a1a2e", transform=ax.transAxes)
        ax.text(0.55, y, "COMPLIANT" if comp else "NON-COMPLIANT", ha="left", va="top",
                fontsize=8, fontweight="bold", color=badge_c, transform=ax.transAxes)
        y -= 0.07
        if p["worn"]:
            s = ", ".join(w.replace("-"," ").title() for w in p["worn"])
            for ln in textwrap.wrap(f"✓  {s}", 40):
                ax.text(0.06, y, ln, ha="left", va="top",
                        fontsize=7.5, color=ok_c, transform=ax.transAxes); y -= 0.058
        if p["missing"]:
            s = ", ".join(m.replace("-"," ").title() for m in p["missing"])
            for ln in textwrap.wrap(f"✗  {s}", 40):
                ax.text(0.06, y, ln, ha="left", va="top",
                        fontsize=7.5, color=fail_c, transform=ax.transAxes); y -= 0.058
        narr = p.get("narrative","").split(".")[0] + "."
        for ln in textwrap.wrap(narr, 46):
            if y < 0.05: break
            ax.text(0.06, y, ln, ha="left", va="top",
                    fontsize=7, color="#555", style="italic", transform=ax.transAxes)
            y -= 0.053
        y -= 0.02

    actions = report_data.get("action_items", [])
    if actions and y > 0.12:
        ax.axhline(y, color="#dde3ea", linewidth=0.8, xmin=0.03, xmax=0.97)
        y -= 0.04
        ax.text(0.06, y, "ACTION ITEMS", ha="left", va="top",
                fontsize=9, fontweight="bold", color="#333", transform=ax.transAxes)
        y -= 0.07
        for item in actions:
            if y < 0.03: break
            col = fail_c if "[URGENT]" in item else warn_c
            for ln in textwrap.wrap(item, 46):
                ax.text(0.06, y, ln, ha="left", va="top",
                        fontsize=7.5, color=col, transform=ax.transAxes); y -= 0.055


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    for cfg in SAMPLES:
        tag = cfg["tag"]
        print(f"\n[{tag.upper()}]")

        img_bgr = cv2.imread(str(cfg["image"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pipeline = PACTPipeline(
            det_weights  = str(cfg["weights"]),
            pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
            dataset      = cfg["dataset"],
            rule         = cfg["rule"],
            device       = "0",
        )
        frame_result = pipeline.run(img_rgb, image_path=str(cfg["image"]))
        frame_dict   = frame_result.to_dict()
        report_data  = generate_report_data(
            frame_dict, rule=cfg["rule"], dataset=cfg["dataset"],
            image_path=str(cfg["image"]),
        )

        # image panels
        save_img(panel_raw(img_rgb),                      OUT_DIR / f"{tag}_01_raw.png")
        save_img(panel_landmarks(img_rgb, frame_result),  OUT_DIR / f"{tag}_02_landmarks.png")
        save_img(panel_pact(img_rgb, frame_result),       OUT_DIR / f"{tag}_03_pact.png")

        # text files — plain, no styling
        proto = cfg["protocol"]
        lines = [
            f"Context: {proto['context']}",
            f"Standard: {proto['standard']}",
            "",
            "Mandatory PPE:",
        ]
        for name, desc in proto["required"]:
            lines.append(f"  - {name}: {desc}")
        lines += ["", "Recommended:"]
        for item in proto["recommended"]:
            lines.append(f"  - {item}")
        p = OUT_DIR / f"{tag}_04_protocol.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved: {p.name}")

        p = OUT_DIR / f"{tag}_05_json.json"
        p.write_text(json.dumps(frame_dict, indent=2), encoding="utf-8")
        print(f"  saved: {p.name}")

        summary = report_data["summary"]
        lines = [
            f"Context: {report_data['meta']['context']}",
            f"Rule: {report_data['meta']['rule']}",
            f"Image: {report_data['meta']['image_path']}",
            f"Timestamp: {report_data['meta']['timestamp']}",
            "",
            f"Workers detected: {summary['num_persons']}",
            f"Compliant: {summary['num_compliant']}",
            f"Violations: {summary['num_violations']}",
            "Compliance rate: " + (f"{summary['compliance_rate']*100:.0f}%" if summary['compliance_rate'] is not None else "N/A"),
            "",
            "Overall assessment:",
            report_data["assessment"],
            "",
            "Workers:",
        ]
        for p_data in report_data["persons"]:
            pid    = p_data["person_id"] + 1
            status = "COMPLIANT" if p_data["compliant"] else "NON-COMPLIANT"
            score  = p_data.get("compliance_score")
            score_str = f" ({score*100:.0f}%)" if score is not None else ""
            lines.append(f"  Worker {pid}: {status}{score_str}")
            if p_data["worn"]:
                lines.append(f"    Worn: {', '.join(p_data['worn'])}")
            if p_data["missing"]:
                lines.append(f"    Missing: {', '.join(p_data['missing'])}")
            lines.append(f"    {p_data['narrative']}")
        if report_data["action_items"]:
            lines += ["", "Action items:"]
            for item in report_data["action_items"]:
                lines.append(f"  {item}")
        p = OUT_DIR / f"{tag}_06_report.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved: {p.name}")

    print(f"\nAll components → {OUT_DIR}")


if __name__ == "__main__":
    main()
