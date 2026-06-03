"""
End-to-End Pipeline Figure
===========================
Produces a 3-row × 6-column figure:

  Col 1 : Raw image
  Col 2 : Raw image + pose landmarks
  Col 3 : Raw image + landmarks + detected PPE (PACT output)
  Col 4 : Company protocol (required PPE for dataset context)
  Col 5 : PACT JSON output
  Col 6 : Compliance report summary

Rows correspond to the 3 datasets: SH17, CPPE-5, CHV.

Output
------
results/phase_2/figure_pipeline.png
results/phase_2/figure_pipeline.pdf
"""
import json
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
sys.path.insert(0, str(ROOT / "code"))

from compliance.pipeline import PACTPipeline
from compliance.visualizer import draw_pact_result
from landmarks.core import draw_pose_landmarks
from landmarks.base import COCO_CONNECTIONS
from reporting.generator import generate_report_data, CONTEXT_LABELS

# ── dataset configs ───────────────────────────────────────────────────────────

SAMPLES = [
    {
        "image":   ROOT / "dataset/SH17/pexels-photo-18110372.jpeg",
        "dataset": "sh17",
        "split":   "original",
        "weights": ROOT / "runs/detect/phase1_sh17_original/weights/best.pt",
        "rule":    "sh17",
        "label":   "Industrial Workplace (SH17)",
        "protocol": {
            "context":  "Industrial / Construction Workplace",
            "standard": "OSHA 29 CFR 1926 / ISO 45001",
            "required": [
                ("Safety Helmet",         "Head protection against falling objects"),
                ("Safety Vest (Hi-Vis)",  "High-visibility for traffic/machinery zones"),
                ("Protective Gloves",     "Hand protection from sharp/hot surfaces"),
            ],
            "recommended": [
                "Safety Shoes / Boots",
                "Eye Protection (Glasses / Goggles)",
                "Hearing Protection (Earmuffs)",
            ],
        },
    },
    {
        "image":   ROOT / "dataset/CPPE-5/130.png",
        "dataset": "cppe5",
        "split":   "original",
        "weights": ROOT / "runs/detect/phase1_cppe_5_original/weights/best.pt",
        "rule":    "cppe5",
        "label":   "Medical / Healthcare (CPPE-5)",
        "protocol": {
            "context":  "Healthcare / Clinical Environment",
            "standard": "WHO PPE Guidelines / CDC Infection Control",
            "required": [
                ("Coverall Suit",   "Full-body barrier against biohazards"),
                ("Face Mask",       "Respiratory droplet protection (N95/surgical)"),
                ("Protective Gloves","Barrier against pathogen contact"),
            ],
            "recommended": [
                "Face Shield / Goggles",
                "Boot Covers",
                "Hair Cover / Cap",
            ],
        },
    },
    {
        "image":   ROOT / "dataset/CHV/ppe_0387.jpg",
        "dataset": "chv",
        "split":   "original",
        "weights": ROOT / "runs/detect/phase1_chv_original/weights/best.pt",
        "rule":    "chv",
        "label":   "Construction Site (CHV)",
        "protocol": {
            "context":  "Construction / Building Site",
            "standard": "OSHA 1910.135 / EN 397 / ISO 20471",
            "required": [
                ("Safety Helmet",        "Head protection — mandatory on all active sites"),
                ("High-Visibility Vest", "Worker visibility to vehicle operators"),
            ],
            "recommended": [
                "Safety Shoes / Steel-Toe Boots",
                "Protective Gloves",
                "Safety Glasses",
            ],
        },
    },
]

COL_TITLES = [
    "Raw Image",
    "Pose Landmarks",
    "PACT Detection",
    "Company Protocol",
    "JSON Output",
    "Compliance Report",
]

# ── panel renderers ───────────────────────────────────────────────────────────

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
        # light person bbox
        x1, y1, x2, y2 = person.person_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 200, 255), 1)
    return canvas


def render_protocol_ax(ax, protocol: dict):
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.95
    ax.text(0.5, y, "COMPANY PROTOCOL", ha="center", va="top",
            fontsize=6.5, fontweight="bold", color="#ffffff",
            fontfamily="monospace", transform=ax.transAxes)
    y -= 0.07

    ax.text(0.05, y, protocol["context"], ha="left", va="top",
            fontsize=5.5, color="#aab4c8", transform=ax.transAxes,
            style="italic")
    y -= 0.06

    ax.text(0.05, y, f"Std: {protocol['standard']}", ha="left", va="top",
            fontsize=4.8, color="#7a8499", transform=ax.transAxes)
    y -= 0.08

    # required PPE
    ax.text(0.05, y, "MANDATORY PPE", ha="left", va="top",
            fontsize=5.8, fontweight="bold", color="#ff6b6b",
            transform=ax.transAxes)
    y -= 0.06

    for name, desc in protocol["required"]:
        ax.text(0.06, y, f"▶  {name}", ha="left", va="top",
                fontsize=5.5, color="#ffffff", fontweight="bold",
                transform=ax.transAxes)
        y -= 0.055
        for line in textwrap.wrap(desc, 30):
            ax.text(0.10, y, line, ha="left", va="top",
                    fontsize=4.8, color="#aab4c8",
                    transform=ax.transAxes)
            y -= 0.048
        y -= 0.01

    y -= 0.02
    ax.text(0.05, y, "RECOMMENDED", ha="left", va="top",
            fontsize=5.5, fontweight="bold", color="#ffd93d",
            transform=ax.transAxes)
    y -= 0.055

    for item in protocol["recommended"]:
        ax.text(0.06, y, f"○  {item}", ha="left", va="top",
                fontsize=5.0, color="#c8d0e0",
                transform=ax.transAxes)
        y -= 0.052


def render_json_ax(ax, frame_dict: dict):
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # build compact JSON string
    compact = {
        "num_persons":   frame_dict["num_persons"],
        "num_compliant": frame_dict["num_compliant"],
        "persons": [
            {
                "id":        p["person_id"],
                "compliant": p["compliant"],
                "worn":      p["worn"],
                "missing":   p["missing"],
            }
            for p in frame_dict["persons"]
        ],
    }
    raw = json.dumps(compact, indent=2)

    # syntax color by line type
    lines = raw.split("\n")
    y = 0.97
    lh = 0.052
    ax.text(0.03, y, "// PACT JSON Output", ha="left", va="top",
            fontsize=4.5, color="#6a9955", fontfamily="monospace",
            transform=ax.transAxes)
    y -= lh

    for line in lines:
        if y < 0.02:
            ax.text(0.03, y, "  ...", ha="left", va="top",
                    fontsize=4.5, color="#6a9955", fontfamily="monospace",
                    transform=ax.transAxes)
            break
        stripped = line.strip()

        if stripped.startswith('"worn"') or stripped.startswith('"missing"') or \
           stripped.startswith('"compliant"') or stripped.startswith('"num_'):
            color = "#9cdcfe"
        elif '"true"' in stripped.lower() or ': true' in stripped:
            color = "#4ec9b0"
        elif ': false' in stripped or '"missing"' in stripped:
            color = "#f44747"
        elif stripped.startswith('"'):
            color = "#ce9178"
        else:
            color = "#d4d4d4"

        ax.text(0.03, y, line, ha="left", va="top",
                fontsize=4.5, color=color, fontfamily="monospace",
                transform=ax.transAxes)
        y -= lh


def render_report_ax(ax, report_data: dict):
    ax.set_facecolor("#f4f6f8")
    ax.axis("off")

    meta    = report_data["meta"]
    summary = report_data["summary"]
    persons = report_data["persons"]

    n    = summary["num_persons"]
    ok   = summary["num_compliant"]
    viol = summary["num_violations"]
    rate = summary["compliance_rate"]
    pct  = f"{rate*100:.0f}%" if rate is not None else "N/A"

    ok_color   = "#2e7d32"
    fail_color = "#b71c1c"
    warn_color = "#e65100"
    main_color = ok_color if viol == 0 else (fail_color if ok == 0 else warn_color)

    y = 0.97
    # header
    ax.text(0.5, y, "COMPLIANCE REPORT", ha="center", va="top",
            fontsize=6.5, fontweight="bold", color="#1a1a2e",
            transform=ax.transAxes)
    y -= 0.07

    ax.text(0.5, y, meta["context"], ha="center", va="top",
            fontsize=5.0, color="#555", style="italic",
            transform=ax.transAxes)
    y -= 0.06

    # summary stats
    ax.text(0.05, y, f"Workers:    {n}", ha="left", va="top",
            fontsize=5.5, color="#333", transform=ax.transAxes)
    y -= 0.055
    ax.text(0.05, y, f"Compliant:  {ok}", ha="left", va="top",
            fontsize=5.5, color=ok_color, fontweight="bold",
            transform=ax.transAxes)
    y -= 0.055
    ax.text(0.05, y, f"Violations: {viol}", ha="left", va="top",
            fontsize=5.5, color=fail_color if viol > 0 else ok_color,
            fontweight="bold", transform=ax.transAxes)
    y -= 0.055
    ax.text(0.05, y, f"Rate:       {pct}", ha="left", va="top",
            fontsize=5.5, color=main_color, fontweight="bold",
            transform=ax.transAxes)
    y -= 0.08

    # divider
    ax.axhline(y, color="#dde3ea", linewidth=0.8, xmin=0.03, xmax=0.97)
    y -= 0.04

    # per-person entries
    for p in persons:
        pid  = p["person_id"] + 1
        comp = p["compliant"]
        badge_color = ok_color if comp else fail_color
        badge_text  = "COMPLIANT" if comp else "NON-COMPLIANT"

        ax.text(0.05, y, f"Worker {pid}", ha="left", va="top",
                fontsize=5.8, fontweight="bold", color="#1a1a2e",
                transform=ax.transAxes)
        ax.text(0.60, y, badge_text, ha="left", va="top",
                fontsize=5.0, fontweight="bold", color=badge_color,
                transform=ax.transAxes)
        y -= 0.055

        if p["worn"]:
            worn_str = ", ".join(w.replace("-", " ").title() for w in p["worn"])
            for line in textwrap.wrap(f"  ✓ {worn_str}", 34):
                ax.text(0.05, y, line, ha="left", va="top",
                        fontsize=4.5, color=ok_color, transform=ax.transAxes)
                y -= 0.047

        if p["missing"]:
            miss_str = ", ".join(m.replace("-", " ").title() for m in p["missing"])
            for line in textwrap.wrap(f"  ✗ {miss_str}", 34):
                ax.text(0.05, y, line, ha="left", va="top",
                        fontsize=4.5, color=fail_color, transform=ax.transAxes)
                y -= 0.047

        # short narrative (first sentence only)
        narrative = p.get("narrative", "")
        first_sent = narrative.split(".")[0] + "." if "." in narrative else narrative
        for line in textwrap.wrap(first_sent, 36):
            if y < 0.05:
                break
            ax.text(0.05, y, line, ha="left", va="top",
                    fontsize=4.3, color="#555", style="italic",
                    transform=ax.transAxes)
            y -= 0.045
        y -= 0.02

    # action items
    actions = report_data.get("action_items", [])
    if actions and y > 0.15:
        ax.axhline(y, color="#dde3ea", linewidth=0.8, xmin=0.03, xmax=0.97)
        y -= 0.04
        ax.text(0.05, y, "ACTION ITEMS", ha="left", va="top",
                fontsize=5.5, fontweight="bold", color="#333",
                transform=ax.transAxes)
        y -= 0.055
        for item in actions:
            if y < 0.03:
                break
            color = fail_color if "[URGENT]" in item else warn_color
            for line in textwrap.wrap(item, 36):
                ax.text(0.05, y, line, ha="left", va="top",
                        fontsize=4.3, color=color, transform=ax.transAxes)
                y -= 0.045


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = ROOT / "results/phase_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(SAMPLES)
    n_cols = 6

    fig = plt.figure(figsize=(36, 15 * n_rows / 3 + 1))
    fig.patch.set_facecolor("#ffffff")

    # column title widths: image cols wider, text cols slightly narrower
    col_widths = [2.2, 2.2, 2.2, 2.0, 2.0, 2.0]
    gs = GridSpec(
        n_rows + 1, n_cols,
        figure=fig,
        height_ratios=[0.12] + [1.0] * n_rows,
        width_ratios=col_widths,
        hspace=0.06,
        wspace=0.04,
    )

    # column headers
    for c, title in enumerate(COL_TITLES):
        ax = fig.add_subplot(gs[0, c])
        ax.set_facecolor("#1a1a2e")
        ax.text(0.5, 0.5, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#ffffff",
                transform=ax.transAxes)
        ax.axis("off")

    for row_idx, cfg in enumerate(SAMPLES):
        r = row_idx + 1
        img_path = cfg["image"]

        print(f"\n[{row_idx+1}/{n_rows}] {cfg['label']}")

        # load image
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # run PACT
        print("  Running PACT pipeline...")
        pipeline = PACTPipeline(
            det_weights  = str(cfg["weights"]),
            pose_weights = str(ROOT / "models/yolov8x-pose.pt"),
            dataset      = cfg["dataset"],
            rule         = cfg["rule"],
            device       = "0",
        )
        frame_result = pipeline.run(img_rgb, image_path=str(img_path))
        frame_dict   = frame_result.to_dict()
        report_data  = generate_report_data(
            frame_dict, rule=cfg["rule"], dataset=cfg["dataset"],
            image_path=str(img_path),
        )

        print(f"  Persons: {frame_dict['num_persons']}  "
              f"Compliant: {frame_dict['num_compliant']}")

        # ── col 1: raw image ─────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[r, 0])
        ax1.imshow(img_rgb)
        ax1.axis("off")
        ax1.set_title(cfg["label"], fontsize=7, color="#333",
                      pad=3, loc="left", fontstyle="italic")

        # ── col 2: landmarks ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[r, 1])
        lm_img = panel_landmarks(img_rgb, frame_result)
        ax2.imshow(lm_img)
        ax2.axis("off")

        # ── col 3: PACT output ───────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[r, 2])
        pact_img = draw_pact_result(img_rgb, frame_result)
        ax3.imshow(pact_img)
        ax3.axis("off")

        # ── col 4: protocol ──────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[r, 3])
        render_protocol_ax(ax4, cfg["protocol"])

        # ── col 5: JSON ──────────────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[r, 4])
        render_json_ax(ax5, frame_dict)

        # ── col 6: report ────────────────────────────────────────────────────
        ax6 = fig.add_subplot(gs[r, 5])
        render_report_ax(ax6, report_data)

        # row label on left spine
        ax1.annotate(
            f"Row {row_idx+1}", xy=(0, 0.5), xytext=(-18, 0),
            xycoords="axes fraction", textcoords="offset points",
            ha="right", va="center", fontsize=7, color="#888",
            rotation=90,
        )

    fig.suptitle(
        "End-to-End PPE Compliance Pipeline  —  PACT (Pose-Anchored Compliance Tracker)",
        fontsize=11, fontweight="bold", color="#1a1a2e", y=0.995,
    )

    for fmt in ("png", "pdf"):
        out_path = out_dir / f"figure_pipeline.{fmt}"
        fig.savefig(str(out_path), dpi=250, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nSaved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
