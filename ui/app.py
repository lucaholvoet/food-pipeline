import os
import requests
from io import BytesIO

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ── Constants ─────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000/analyze")

BBOX_COLORS = [
    "#FF6B6B",  # coral
    "#4ECDC4",  # teal
    "#45B7D1",  # sky blue
    "#96CEB4",  # sage
    "#F8B500",  # amber
    "#DDA0DD",  # plum
    "#98D8C8",  # mint
    "#F7DC6F",  # gold
]

WARNING_LABELS = {
    "no_plate_detected": (
        "No plate detected",
        "Portion estimates may be less accurate",
    ),
    "low_confidence_vlm_triggered": (
        "Low confidence — AI refinement used",
        "CV confidence was below 70%",
    ),
    "no_food_detected": (
        "No food detected",
        "No food items were found in this image",
    ),
}

CUSTOM_CSS = """
.food-table { width: 100%; border-collapse: collapse; margin: 8px 0; }
.food-table th {
    background: #2d3748; color: #a0aec0;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
    padding: 10px 14px; text-align: left; font-weight: 600;
}
.food-table td {
    padding: 11px 14px; border-bottom: 1px solid #edf2f7;
    font-size: 0.9rem; color: #2d3748;
}
.food-table tr:last-child td { border-bottom: none; }
.food-table tr:hover td { background: #f7fafc; }

.totals-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 24px 32px; color: white; margin: 16px 0;
}
.totals-label {
    font-size: 0.82rem; color: #a0aec0;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.totals-calories {
    font-size: 3.2rem; font-weight: 800; color: #f8b500; line-height: 1.1; margin: 4px 0;
}
.macros-row { display: flex; gap: 14px; margin-top: 16px; flex-wrap: wrap; }
.macro-pill {
    background: rgba(255,255,255,0.08); border-radius: 10px;
    padding: 10px 18px; text-align: center; flex: 1; min-width: 72px;
}
.macro-value { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
.macro-name {
    font-size: 0.68rem; color: #718096;
    text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px;
}
.proc-time { font-size: 0.75rem; color: #a0aec0; text-align: right; margin-top: 10px; }

.badge { border-radius: 12px; padding: 3px 10px; font-size: 0.78rem; font-weight: 600; }
.badge-green  { background: #c6f6d5; color: #276749; }
.badge-yellow { background: #fefcbf; color: #744210; }
.badge-red    { background: #fed7d7; color: #742a2a; }

.warning-banner {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    padding: 10px 16px; border-radius: 0 6px 6px 0;
    color: #78350f; font-size: 0.88rem; margin: 4px 0;
}

.section-title {
    font-size: 0.85rem; font-weight: 700; color: #4a5568;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 20px 0 10px; padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
}

.vlm-header {
    background: linear-gradient(90deg, #4a00e0, #8e2de2);
    color: white; padding: 12px 20px; border-radius: 10px;
    font-weight: 700; margin-bottom: 12px; font-size: 0.92rem;
}
.vlm-card {
    border: 1px solid #e9d8fd; border-radius: 10px;
    padding: 16px; margin: 8px 0; background: #faf5ff;
}
.vlm-note {
    background: #f3e8ff; border-left: 3px solid #805ad5;
    padding: 8px 14px; border-radius: 0 6px 6px 0;
    font-size: 0.83rem; color: #553c9a; margin-top: 6px;
}
.action-badge {
    display: inline-block; border-radius: 4px;
    padding: 2px 8px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.04em; margin-right: 8px;
}
.action-corrected { background: #fed7d7; color: #c53030; }
.action-confirmed { background: #c6f6d5; color: #276749; }
.action-unknown   { background: #fefcbf; color: #744210; }

.color-dot {
    display: inline-block; border-radius: 50%;
    vertical-align: middle; margin-right: 6px;
}
.status-error { color: #e53e3e; font-size: 0.9rem; padding: 4px 0; }
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_dark(hex_color: str) -> bool:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.55


def confidence_badge(conf: float) -> str:
    pct = int(round(conf * 100))
    if conf >= 0.80:
        cls = "badge-green"
    elif conf >= 0.60:
        cls = "badge-yellow"
    else:
        cls = "badge-red"
    return f'<span class="badge {cls}">{pct}%</span>'


def is_vlm_response(data: dict) -> bool:
    # CVPipelineOutput has "status"; VLMResponse has "refinement_status"
    return "refinement_status" in data


# ── Bbox drawing ──────────────────────────────────────────────────────────────

def draw_bboxes(image_pil: Image.Image, items: list) -> Image.Image:
    annotated = image_pil.copy()
    draw = ImageDraw.Draw(annotated)

    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, size=18)
            break
        except (IOError, OSError):
            pass
    if font is None:
        font = ImageFont.load_default()

    for idx, item in enumerate(items):
        bbox = item.get("bbox", [])
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = BBOX_COLORS[idx % len(BBOX_COLORS)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        conf_pct = int(round(float(item.get("classification_confidence", 0)) * 100))
        label = f"{item.get('display_name', 'Unknown')} {conf_pct}%"

        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        label_y0 = max(0, y1 - th - 8)
        label_y1 = max(th + 6, y1)
        draw.rectangle([x1, label_y0, x1 + tw + 10, label_y1], fill=color)
        draw.text(
            (x1 + 5, label_y0 + 3),
            label,
            fill="white" if _is_dark(color) else "#1a1a1a",
            font=font,
        )

    return annotated


# ── HTML builders ─────────────────────────────────────────────────────────────

def build_warnings_html(warnings: list) -> str:
    if not warnings:
        return ""
    parts = []
    for w in warnings:
        if w in WARNING_LABELS:
            title, detail = WARNING_LABELS[w]
        else:
            title = w.replace("_", " ").title()
            detail = ""
        detail_str = f" — {detail}" if detail else ""
        parts.append(
            f'<div class="warning-banner">⚠ <strong>{title}</strong>{detail_str}</div>'
        )
    return "\n".join(parts)


def build_totals_html(totals: dict, proc_ms: int) -> str:
    cal   = f"{float(totals.get('calories_kcal', 0)):,.1f}"
    prot  = f"{float(totals.get('protein_g', 0)):.1f}g"
    fat   = f"{float(totals.get('fat_g', 0)):.1f}g"
    carbs = f"{float(totals.get('carbs_g', 0)):.1f}g"
    fiber = f"{float(totals.get('fiber_g', 0)):.1f}g"

    return f"""
<div class="totals-card">
  <div class="totals-label">Total Calories</div>
  <div class="totals-calories">{cal} kcal</div>
  <div class="macros-row">
    <div class="macro-pill">
      <div class="macro-value">{prot}</div>
      <div class="macro-name">Protein</div>
    </div>
    <div class="macro-pill">
      <div class="macro-value">{fat}</div>
      <div class="macro-name">Fat</div>
    </div>
    <div class="macro-pill">
      <div class="macro-value">{carbs}</div>
      <div class="macro-name">Carbs</div>
    </div>
    <div class="macro-pill">
      <div class="macro-value">{fiber}</div>
      <div class="macro-name">Fiber</div>
    </div>
  </div>
  <div class="proc-time">Processed in {proc_ms:,}ms</div>
</div>"""


def _table_header(conf_label: str = "Confidence") -> str:
    return f"""
<table class="food-table">
<thead><tr>
  <th>#</th>
  <th>Food</th>
  <th>Grams</th>
  <th>Calories</th>
  <th>Protein</th>
  <th>Fat</th>
  <th>Carbs</th>
  <th>Fiber</th>
  <th>{conf_label}</th>
</tr></thead>
<tbody>"""


def build_cv_items_table(items: list) -> str:
    if not items:
        return '<p style="color:#718096;font-size:0.9rem;padding:8px 0">No food items detected.</p>'

    rows = _table_header("Confidence")
    for idx, item in enumerate(items):
        color = BBOX_COLORS[idx % len(BBOX_COLORS)]
        dot = f'<span class="color-dot" style="width:12px;height:12px;background:{color}"></span>'
        n = item.get("nutrition_total", {})
        conf = float(item.get("classification_confidence", 0))
        grams = float(item.get("estimated_grams", 0))
        rows += f"""
<tr>
  <td>{dot}{item.get("item_id", idx + 1)}</td>
  <td><strong>{item.get("display_name", "")}</strong></td>
  <td>{grams:.0f}g</td>
  <td>{float(n.get("calories_kcal", 0)):.0f}</td>
  <td>{float(n.get("protein_g", 0)):.1f}g</td>
  <td>{float(n.get("fat_g", 0)):.1f}g</td>
  <td>{float(n.get("carbs_g", 0)):.1f}g</td>
  <td>{float(n.get("fiber_g", 0)):.1f}g</td>
  <td>{confidence_badge(conf)}</td>
</tr>"""
    return rows + "</tbody></table>"


def build_vlm_items_table(items: list) -> str:
    if not items:
        return ""

    rows = _table_header("VLM Conf")
    for idx, item in enumerate(items):
        color = BBOX_COLORS[idx % len(BBOX_COLORS)]
        dot = f'<span class="color-dot" style="width:12px;height:12px;background:{color}"></span>'
        refined  = item.get("refined", {})
        portion  = item.get("portion", {})
        original = item.get("original", {})
        n = item.get("nutrition_total", {})
        conf = float(refined.get("vlm_confidence", 0))
        grams = float(portion.get("estimated_grams", 0))

        if item.get("action") == "corrected":
            orig_name = original.get("food_name", "").replace("_", " ").title()
            food_cell = (
                f'<s style="color:#a0aec0;font-size:0.82rem">{orig_name}</s><br>'
                f'<strong>{refined.get("display_name", "")}</strong>'
            )
        else:
            food_cell = f'<strong>{refined.get("display_name", "")}</strong>'

        rows += f"""
<tr>
  <td>{dot}{item.get("item_id", idx + 1)}</td>
  <td>{food_cell}</td>
  <td>{grams:.0f}g</td>
  <td>{float(n.get("calories_kcal", 0)):.0f}</td>
  <td>{float(n.get("protein_g", 0)):.1f}g</td>
  <td>{float(n.get("fat_g", 0)):.1f}g</td>
  <td>{float(n.get("carbs_g", 0)):.1f}g</td>
  <td>{float(n.get("fiber_g", 0)):.1f}g</td>
  <td>{confidence_badge(conf)}</td>
</tr>"""
    return rows + "</tbody></table>"


def build_vlm_panel(data: dict) -> str:
    model     = data.get("vlm_model", "VLM")
    threshold = int(float(data.get("confidence_threshold_used", 0.70)) * 100)
    status    = data.get("refinement_status", "completed")

    header = (
        f'<div class="vlm-header">'
        f'🤖 AI Refinement &nbsp;·&nbsp; {model} &nbsp;·&nbsp; '
        f'threshold: {threshold}% &nbsp;·&nbsp; {status}'
        f'</div>'
    )

    cards = []
    for item in data.get("items", []):
        action   = item.get("action", "unknown")
        refined  = item.get("refined", {})
        original = item.get("original", {})
        portion  = item.get("portion", {})

        orig_conf_pct = int(round(float(original.get("classification_confidence", 0)) * 100))
        vlm_conf_pct  = int(round(float(refined.get("vlm_confidence", 0)) * 100))
        orig_name     = original.get("food_name", "").replace("_", " ").title()
        refined_name  = refined.get("display_name", "")

        if action == "corrected":
            food_line = (
                f'<span style="color:#e53e3e;text-decoration:line-through">{orig_name}</span>'
                f'<span style="color:#805ad5;font-weight:bold;margin:0 8px">→</span>'
                f'<strong style="color:#276749">{refined_name}</strong>'
            )
        elif action == "confirmed":
            food_line = f'<strong style="color:#276749">{refined_name}</strong>'
        else:
            food_line = f'<strong style="color:#2d3748">{refined_name}</strong>'

        desc = refined.get("food_description", "")
        desc_html = (
            f'<div style="color:#4a5568;font-size:0.85rem;margin:6px 0">{desc}</div>'
            if desc else ""
        )

        portion_method = portion.get("portion_method", "").replace("_", " ")
        portion_conf   = int(round(float(portion.get("vlm_confidence", 0)) * 100))
        grams          = float(portion.get("estimated_grams", 0))

        cards.append(f"""
<div class="vlm-card">
  <div style="margin-bottom:8px">
    <span class="action-badge action-{action}">{action}</span>
    <span style="color:#718096;font-size:0.85rem">Item #{item.get("item_id", "")}</span>
    <span style="float:right;color:#718096;font-size:0.82rem">
      CV {orig_conf_pct}% → VLM {vlm_conf_pct}%
    </span>
  </div>
  <div style="font-size:1rem;margin:6px 0">{food_line}</div>
  {desc_html}
  <div style="color:#718096;font-size:0.82rem;margin-top:4px">
    Portion: {grams:.0f}g via {portion_method} ({portion_conf}% confidence)
  </div>
</div>""")

    notes = data.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    notes_html = ""
    if notes:
        note_items = "".join(f'<div class="vlm-note">{n}</div>' for n in notes)
        notes_html = f'<div class="section-title">Notes</div>{note_items}'

    return header + "\n".join(cards) + notes_html


# ── Gradio callback ───────────────────────────────────────────────────────────

def _empty_return(msg: str = "") -> tuple:
    hidden = gr.update(visible=False)
    status = f'<p class="status-error">{msg}</p>' if msg else ""
    return (
        gr.update(value=None, visible=False),  # output_image
        gr.update(value=""),                   # warnings_html
        hidden,                                 # warnings_panel
        gr.update(value=""),                   # results_html
        hidden,                                 # results_panel
        gr.update(value=""),                   # vlm_html
        hidden,                                 # vlm_panel
        status,                                 # status_html
    )


def analyze_image(input_image: Image.Image) -> tuple:
    if input_image is None:
        return _empty_return("Please upload or capture an image first.")

    buf = BytesIO()
    input_image.save(buf, format="JPEG", quality=92)
    buf.seek(0)

    try:
        resp = requests.post(
            API_URL,
            files={"file": ("image.jpg", buf, "image/jpeg")},
            timeout=180,
        )
    except requests.ConnectionError:
        return _empty_return(
            f"Cannot connect to backend at <code>{API_URL}</code>. Is it running?"
        )
    except requests.Timeout:
        return _empty_return("Request timed out — the pipeline may still be loading.")

    if resp.status_code == 503:
        return _empty_return("Pipeline is loading. Please wait ~30s and try again.")
    if resp.status_code == 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return _empty_return(f"Invalid image: {detail}")
    if resp.status_code != 200:
        return _empty_return(f"Backend error (HTTP {resp.status_code}).")

    try:
        data = resp.json()
    except Exception:
        return _empty_return("Backend returned an invalid response.")

    try:
        vlm      = is_vlm_response(data)
        items    = data.get("items", [])
        warnings = data.get("warnings", [])

        if not vlm and items:
            try:
                annotated = draw_bboxes(input_image, items)
            except Exception:
                annotated = None
        else:
            annotated = None

        w_html       = build_warnings_html(warnings)
        totals_html  = build_totals_html(
            data.get("totals", {}), data.get("processing_time_ms", 0)
        )
        table_html   = build_vlm_items_table(items) if vlm else build_cv_items_table(items)
        results_html = (
            totals_html
            + '<div class="section-title">Per-item Breakdown</div>'
            + table_html
        )
        vlm_html = build_vlm_panel(data) if vlm else ""

    except Exception as e:
        return _empty_return(f"Unexpected response format: {e}")

    return (
        gr.update(value=annotated, visible=annotated is not None),
        gr.update(value=w_html),
        gr.update(visible=bool(warnings)),
        gr.update(value=results_html),
        gr.update(visible=True),
        gr.update(value=vlm_html),
        gr.update(visible=vlm),
        "",
    )


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="Food Calorie Estimator") as demo:

    gr.HTML("""
        <div style="text-align:center;padding:28px 0 12px">
          <h1 style="font-size:2rem;font-weight:800;color:#1a202c;margin:0">
            Food Calorie Estimator
          </h1>
          <p style="color:#718096;margin:8px 0 0;font-size:1rem">
            Upload or photograph your meal for instant nutrition analysis
          </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                sources=["upload", "webcam"],
                type="pil",
                label="Your meal",
                height=380,
            )
            analyze_btn = gr.Button("Analyze Meal", variant="primary", size="lg")
            status_html = gr.HTML("")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detected items",
                interactive=False,
                height=380,
                visible=False,
            )

    with gr.Column(visible=False) as warnings_panel:
        warnings_html = gr.HTML("")

    with gr.Column(visible=False) as results_panel:
        results_html = gr.HTML("")

    with gr.Column(visible=False) as vlm_panel:
        vlm_html = gr.HTML("")

    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[
            output_image,
            warnings_html,
            warnings_panel,
            results_html,
            results_panel,
            vlm_html,
            vlm_panel,
            status_html,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        show_error=True,
        show_api=False,
    )
