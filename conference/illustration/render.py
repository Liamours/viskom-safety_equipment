#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "requests",
#   "pymupdf",
# ]
# ///

"""
Render TikZ .tex files → PDF (via tectonic) → PNG (via pymupdf).
Tectonic binary is downloaded automatically from GitHub releases.
"""

import os
import sys
import zipfile
import tempfile
import subprocess
import urllib.request
import json
from pathlib import Path

ILLUS_DIR = Path(__file__).parent
TEX_FILES = [
    ILLUS_DIR / "flow_ppe_detection.tex",
    ILLUS_DIR / "flow_landmark_detection.tex",
]
TECTONIC_BIN = ILLUS_DIR / "tectonic.exe"
DPI = 300


# ── 1. Get tectonic ───────────────────────────────────────────────────────────

def get_tectonic():
    if TECTONIC_BIN.exists():
        print(f"[tectonic] already exists: {TECTONIC_BIN}")
        return

    print("[tectonic] fetching latest release info from GitHub...")
    api = "https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest"
    with urllib.request.urlopen(api) as r:
        release = json.loads(r.read())

    asset_url = None
    for asset in release["assets"]:
        if "x86_64-pc-windows-msvc" in asset["name"] and asset["name"].endswith(".zip"):
            asset_url = asset["browser_download_url"]
            break

    if not asset_url:
        raise RuntimeError("Could not find Windows tectonic binary in latest release assets")

    print(f"[tectonic] downloading {asset_url} ...")
    zip_path = ILLUS_DIR / "tectonic_tmp.zip"
    urllib.request.urlretrieve(asset_url, zip_path)

    print("[tectonic] extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("tectonic.exe"):
                data = zf.read(name)
                TECTONIC_BIN.write_bytes(data)
                break

    zip_path.unlink()
    print(f"[tectonic] ready at {TECTONIC_BIN}")


# ── 2. Compile .tex → .pdf ────────────────────────────────────────────────────

def compile_tex(tex_path: Path) -> Path:
    pdf_path = tex_path.with_suffix(".pdf")
    print(f"[compile] {tex_path.name} -> {pdf_path.name}")

    result = subprocess.run(
        [str(TECTONIC_BIN), str(tex_path)],
        cwd=str(ILLUS_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[compile] STDOUT:", result.stdout[-2000:])
        print("[compile] STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"tectonic failed on {tex_path.name}")

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not produced for {tex_path.name}")

    print(f"[compile] OK -> {pdf_path}")
    return pdf_path


# ── 3. PDF → PNG via pymupdf ──────────────────────────────────────────────────

def pdf_to_png(pdf_path: Path, dpi: int = DPI) -> Path:
    import fitz  # pymupdf

    png_path = pdf_path.with_suffix(".png")
    print(f"[render] {pdf_path.name} -> {png_path.name} @ {dpi} dpi")

    doc = fitz.open(str(pdf_path))
    page = doc[0]

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(png_path))
    doc.close()

    print(f"[render] OK -> {png_path}")
    return png_path


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    get_tectonic()

    for tex in TEX_FILES:
        if not tex.exists():
            print(f"[skip] {tex} not found")
            continue
        pdf = compile_tex(tex)
        png = pdf_to_png(pdf)

    print("\nDone. PNGs saved to:", ILLUS_DIR)
