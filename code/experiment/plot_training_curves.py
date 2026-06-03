"""
Training curve plots — 8020 split only.

Inputs : runs/detect/phase1_{chv|cppe_5|sh17}_8020/results.csv
Outputs: results/figures/training_map50.png
         results/figures/training_val_loss.png
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pathlib import Path

# ── global font ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "Times", "DejaVu Serif"],
    "font.weight":     "normal",
    "axes.titleweight":"normal",
    "axes.labelweight":"normal",
})

ROOT    = Path("C:/Users/lulay/Desktop/viskom-safety_equipment")
OUT_DIR = ROOT / "results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "CHV":    ROOT / "runs/detect/phase1_chv_8020/results.csv",
    "CPPE-5": ROOT / "runs/detect/phase1_cppe_5_8020/results.csv",
    "SH17":   ROOT / "runs/detect/phase1_sh17_8020/results.csv",
}

COLORS = {
    "CHV":    "#2563EB",   # blue
    "CPPE-5": "#DC2626",   # red
    "SH17":   "#16A34A",   # green
}

# ── load ──────────────────────────────────────────────────────────────────────
data = {}
for name, path in RUNS.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["val_loss"] = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
    data[name] = df
    print(f"{name}: {len(df)} epochs")


def _style_ax(ax, title=""):
    ax.set_title(title, fontsize=11, pad=5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, framealpha=0.5, edgecolor="none")


# ── Figure 1: mAP50 ───────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

for ax, (name, df) in zip(axes1, data.items()):
    color = COLORS[name]
    ax.plot(df["epoch"], df["metrics/mAP50(B)"],
            color=color, linewidth=1.0, alpha=0.3, label="Actual")
    smoothed = df["metrics/mAP50(B)"].rolling(5, min_periods=1).mean()
    ax.plot(df["epoch"], smoothed,
            color=color, linewidth=2.0, label="Smoothed")
    ax.set_ylim(0.0, 1.0)
    _style_ax(ax, title=name)

fig1.tight_layout(w_pad=0.5)
out1 = OUT_DIR / "training_map50.png"
fig1.savefig(out1, dpi=200, bbox_inches="tight")
print(f"saved: {out1}")
plt.close(fig1)


# ── Figure 2: Val Loss ────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

for ax, (name, df) in zip(axes2, data.items()):
    color = COLORS[name]
    ax.plot(df["epoch"], df["val_loss"],
            color=color, linewidth=1.0, alpha=0.3, label="Actual")
    smoothed = df["val_loss"].rolling(5, min_periods=1).mean()
    ax.plot(df["epoch"], smoothed,
            color=color, linewidth=2.0, label="Smoothed")
    ax.set_ylim(1.5, 4.2)
    _style_ax(ax, title=name)

fig2.tight_layout(w_pad=0.5)
out2 = OUT_DIR / "training_val_loss.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
print(f"saved: {out2}")
plt.close(fig2)


# ── Figure 3: mAP50 combined ──────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(5, 3))

for name, df in data.items():
    color = COLORS[name]
    ax3.plot(df["epoch"], df["metrics/mAP50(B)"],
             color=color, linewidth=1.0, alpha=0.3)
    smoothed = df["metrics/mAP50(B)"].rolling(5, min_periods=1).mean()
    ax3.plot(df["epoch"], smoothed,
             color=color, linewidth=2.0, label=name)

ax3.set_ylim(0.0, 1.0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
ax3.tick_params(labelsize=9)
ax3.legend(fontsize=9, framealpha=0.5, edgecolor="none")

fig3.tight_layout()
out3 = OUT_DIR / "training_map50_combined.png"
fig3.savefig(out3, dpi=200, bbox_inches="tight")
print(f"saved: {out3}")
plt.close(fig3)


# ── Figure 4: Val Loss combined ───────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(5, 3))

for name, df in data.items():
    color = COLORS[name]
    ax4.plot(df["epoch"], df["val_loss"],
             color=color, linewidth=1.0, alpha=0.3)
    smoothed = df["val_loss"].rolling(5, min_periods=1).mean()
    ax4.plot(df["epoch"], smoothed,
             color=color, linewidth=2.0, label=name)

ax4.set_ylim(1.5, 4.2)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
ax4.tick_params(labelsize=9)
ax4.legend(fontsize=9, framealpha=0.5, edgecolor="none")

fig4.tight_layout()
out4 = OUT_DIR / "training_val_loss_combined.png"
fig4.savefig(out4, dpi=200, bbox_inches="tight")
print(f"saved: {out4}")
plt.close(fig4)
