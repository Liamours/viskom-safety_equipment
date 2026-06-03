"""
HTML report renderer.

Takes report_data from generator.py + original image (numpy BGR or path)
and produces a self-contained HTML file with embedded base64 person crops.
"""
import base64
import io
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# Fix #10: import canonical label map so HTML card matches generator output
from reporting.generator import PPE_LABELS as _PPE_LABELS


def _ppe_label(key: str) -> str:
    return _PPE_LABELS.get(key, key.replace("-", " ").replace("_", " ").title())


# ── image helpers ─────────────────────────────────────────────────────────────

def _load_image(image: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if not path.exists():
        return None
    return cv2.imread(str(path))


def _bgr_to_b64(img_bgr: np.ndarray, max_dim: int = 800) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _crop_person(img_bgr: np.ndarray, bbox: list, pad: float = 0.05) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    return img_bgr[y1:y2, x1:x2]


# ── HTML building blocks ──────────────────────────────────────────────────────

_CSS = """
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f4f6f8;
    margin: 0; padding: 0;
    color: #1a1a2e;
}
.page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px;
}
.header {
    background: #1a1a2e;
    color: #fff;
    border-radius: 8px;
    padding: 28px 32px 20px;
    margin-bottom: 24px;
}
.header h1 { margin: 0 0 6px; font-size: 1.5rem; letter-spacing: 0.04em; }
.header .meta-row { font-size: 0.82rem; color: #aab4c8; margin-top: 4px; }
.header .meta-row span { margin-right: 24px; }
.summary-box {
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 32px;
}
.summary-box.ok   { background: #e8f5e9; border-left: 5px solid #2e7d32; }
.summary-box.warn { background: #fff3e0; border-left: 5px solid #e65100; }
.summary-box.fail { background: #fce4ec; border-left: 5px solid #b71c1c; }
.stat { text-align: center; }
.stat .num { font-size: 2rem; font-weight: 700; line-height: 1; }
.stat .lbl { font-size: 0.75rem; color: #555; margin-top: 2px; }
.ok   .num { color: #2e7d32; }
.warn .num { color: #e65100; }
.fail .num { color: #b71c1c; }
.assessment {
    background: #fff;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 24px;
    border: 1px solid #dde3ea;
    font-size: 0.92rem;
    line-height: 1.6;
}
.assessment h2 { margin: 0 0 8px; font-size: 0.85rem; text-transform: uppercase;
                 letter-spacing: 0.08em; color: #666; }
.scene-img {
    background: #fff;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
    border: 1px solid #dde3ea;
    text-align: center;
}
.scene-img h2 { margin: 0 0 12px; font-size: 0.85rem; text-transform: uppercase;
                letter-spacing: 0.08em; color: #666; text-align: left; }
.scene-img img { max-width: 100%; border-radius: 4px; }
.section-title {
    font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #666;
    margin-bottom: 12px; margin-top: 0;
}
.persons-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.person-card {
    background: #fff;
    border-radius: 8px;
    border: 1px solid #dde3ea;
    overflow: hidden;
}
.person-card .card-top {
    display: flex;
    gap: 14px;
    padding: 14px;
    align-items: flex-start;
}
.person-card .crop-wrap {
    flex-shrink: 0;
    width: 90px; height: 110px;
    background: #eee;
    border-radius: 4px;
    overflow: hidden;
    display: flex; align-items: center; justify-content: center;
}
.person-card .crop-wrap img { width: 100%; height: 100%; object-fit: cover; }
.person-card .crop-wrap .no-img { font-size: 0.7rem; color: #999; text-align: center; }
.person-card .card-info { flex: 1; }
.person-card .pid { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
.badge {
    display: inline-block;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.06em; padding: 2px 8px;
    border-radius: 3px; text-transform: uppercase;
    margin-bottom: 8px;
}
.badge.ok   { background: #e8f5e9; color: #2e7d32; }
.badge.fail { background: #fce4ec; color: #b71c1c; }
.ppe-list { list-style: none; padding: 0; margin: 0; font-size: 0.8rem; }
.ppe-list li { padding: 2px 0; }
.ppe-list .worn    { color: #2e7d32; }
.ppe-list .missing { color: #b71c1c; }
.ppe-list .icon { margin-right: 5px; }
.narrative {
    background: #f9fafb;
    border-top: 1px solid #eee;
    padding: 12px 14px;
    font-size: 0.8rem;
    line-height: 1.6;
    color: #333;
}
.actions {
    background: #fff;
    border-radius: 8px;
    border: 1px solid #dde3ea;
    padding: 16px 20px;
    margin-bottom: 24px;
}
.actions ul { margin: 8px 0 0; padding-left: 18px; font-size: 0.85rem; line-height: 1.8; }
.actions .urgent  { color: #b71c1c; font-weight: 600; }
.actions .advisory { color: #e65100; }
.footer {
    border-top: 1px solid #dde3ea;
    padding-top: 14px;
    font-size: 0.75rem;
    color: #999;
}
"""

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPE Compliance Report</title>
<style>{css}</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <h1>PPE Compliance Inspection Report</h1>
    <div class="meta-row">
      <span>Date &amp; Time: {timestamp}</span>
      <span>Context: {context}</span>
      <span>Inspector: {inspector}</span>
    </div>
    <div class="meta-row">
      <span>Image: {image_name}</span>
      <span>Dataset: {dataset}</span>
      <span>Rule: {rule}</span>
    </div>
  </div>

  <!-- SUMMARY BOX -->
  <div class="summary-box {summary_cls}">
    <div class="stat">
      <div class="num">{num_persons}</div>
      <div class="lbl">Workers Detected</div>
    </div>
    <div class="stat">
      <div class="num">{num_compliant}</div>
      <div class="lbl">Compliant</div>
    </div>
    <div class="stat">
      <div class="num">{num_violations}</div>
      <div class="lbl">Violations</div>
    </div>
    <div class="stat">
      <div class="num">{compliance_pct}</div>
      <div class="lbl">Compliance Rate</div>
    </div>
  </div>

  <!-- OVERALL ASSESSMENT -->
  <div class="assessment">
    <h2>Overall Assessment</h2>
    {assessment}
  </div>

  <!-- SCENE IMAGE -->
  {scene_section}

  <!-- PERSON CARDS -->
  {persons_section}

  <!-- ACTION ITEMS -->
  {actions_section}

  <!-- FOOTER -->
  <div class="footer">
    Generated by PACT (Pose-Anchored Compliance Tracker) &mdash;
    End-to-End PPE Compliance System &mdash; {timestamp}
  </div>

</div>
</body>
</html>
"""


def _summary_cls(report_data: dict) -> str:
    s = report_data["summary"]
    n = s["num_persons"]
    if n == 0:
        return "warn"
    if s["num_violations"] == 0:
        return "ok"
    if s["num_compliant"] == 0:
        return "fail"
    return "warn"


def _scene_section(img_bgr: Optional[np.ndarray]) -> str:
    if img_bgr is None:
        return ""
    b64 = _bgr_to_b64(img_bgr, max_dim=900)
    return (
        '<div class="scene-img">'
        '<h2>Scene</h2>'
        f'<img src="data:image/jpeg;base64,{b64}" alt="scene">'
        "</div>"
    )


def _person_card(person: dict, img_bgr: Optional[np.ndarray]) -> str:
    pid       = person["person_id"] + 1
    compliant = person["compliant"]
    worn      = person.get("worn", [])
    missing   = person.get("missing", [])
    bbox      = person.get("person_bbox", [])
    narrative = person.get("narrative", "")

    badge = '<span class="badge ok">Compliant</span>' if compliant else \
            '<span class="badge fail">Non-Compliant</span>'

    # PPE lists — use canonical PPE_LABELS dict (fix #10)
    ppe_items = ""
    for w in worn:
        ppe_items += f'<li class="worn"><span class="icon">&#10003;</span>{_ppe_label(w)}</li>'
    for m in missing:
        ppe_items += f'<li class="missing"><span class="icon">&#10007;</span>{_ppe_label(m)}</li>'

    # Person crop
    crop_html = '<div class="no-img">No image</div>'
    if img_bgr is not None and bbox and len(bbox) == 4:
        crop = _crop_person(img_bgr, bbox)
        if crop.size > 0:
            b64 = _bgr_to_b64(crop, max_dim=300)
            crop_html = f'<img src="data:image/jpeg;base64,{b64}" alt="Worker {pid}">'

    return f"""
<div class="person-card">
  <div class="card-top">
    <div class="crop-wrap">{crop_html}</div>
    <div class="card-info">
      <div class="pid">Worker {pid}</div>
      {badge}
      <ul class="ppe-list">{ppe_items}</ul>
    </div>
  </div>
  <div class="narrative">{narrative}</div>
</div>"""


def _persons_section(report_data: dict, img_bgr: Optional[np.ndarray]) -> str:
    persons = report_data.get("persons", [])
    if not persons:
        return '<p style="color:#888;font-size:0.9rem;">No persons detected in this frame.</p>'
    cards = "\n".join(_person_card(p, img_bgr) for p in persons)
    return (
        f'<h2 class="section-title">Worker Analysis</h2>'
        f'<div class="persons-grid">{cards}</div>'
    )


def _actions_section(report_data: dict) -> str:
    items = report_data.get("action_items", [])
    if not items:
        return (
            '<div class="actions">'
            '<h2 class="section-title">Action Items</h2>'
            '<p style="color:#2e7d32;font-size:0.85rem;">No corrective actions required.</p>'
            "</div>"
        )
    li_tags = ""
    for item in items:
        cls = "urgent" if item.startswith("[URGENT]") else "advisory"
        li_tags += f'<li class="{cls}">{item}</li>'
    return (
        '<div class="actions">'
        '<h2 class="section-title">Action Items</h2>'
        f"<ul>{li_tags}</ul>"
        "</div>"
    )


# ── public API ────────────────────────────────────────────────────────────────

def render_html_report(
    report_data: dict,
    image: Union[str, Path, np.ndarray, None] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Render a self-contained HTML report.

    Parameters
    ----------
    report_data  : output of generator.generate_report_data()
    image        : original scene image (path or BGR numpy array); optional
    output_path  : save to this path if given; always returns HTML string

    Returns
    -------
    HTML string
    """
    img_bgr = _load_image(image) if image is not None else None

    meta    = report_data["meta"]
    summary = report_data["summary"]

    n         = summary["num_persons"]
    rate      = summary["compliance_rate"]
    pct_str   = f"{rate * 100:.0f}%" if rate is not None else "N/A"
    image_name = Path(meta["image_path"]).name if meta["image_path"] else "—"

    html = _HTML_TEMPLATE.format(
        css            = _CSS,
        timestamp      = meta["timestamp"],
        context        = meta["context"],
        inspector      = meta["inspector"],
        image_name     = image_name,
        dataset        = meta["dataset"] or "—",
        rule           = meta["rule"],
        summary_cls    = _summary_cls(report_data),
        num_persons    = n,
        num_compliant  = summary["num_compliant"],
        num_violations = summary["num_violations"],
        compliance_pct = pct_str,
        assessment     = report_data["assessment"],
        scene_section  = _scene_section(img_bgr),
        persons_section= _persons_section(report_data, img_bgr),
        actions_section= _actions_section(report_data),
    )

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
