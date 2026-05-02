import os
import json
import requests
from io import BytesIO
from datetime import date
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from profile import (init_profile_db, save_profile, get_profile,
                     update_weight, calculate_profile, ACTIVITY_MULTIPLIERS,
                     weeks_until)
from db import (init_db, log_meal, get_today_meals, get_today_totals,
                get_weekly_data, get_meals_for_date, get_totals_for_date,
                delete_meal, generate_user_id)

from auth import init_auth_db, register_user, login_user

from llm_chat import chat_correction, chat_manual, MAX_TURNS

init_auth_db()


init_db()
init_profile_db()

# ── Constants ─────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000/analyze")

BBOX_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#F8B500", "#DDA0DD", "#98D8C8", "#F7DC6F",
]

WARNING_LABELS = {
    "no_plate_detected": ("No plate detected", "Portion estimates may be less accurate"),
    "low_confidence_vlm_triggered": ("Low confidence — AI refinement used", "CV confidence was below 70%"),
    "no_food_detected": ("No food detected", "No food items were found in this image"),
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
.totals-label { font-size: 0.82rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.06em; }
.totals-calories { font-size: 3.2rem; font-weight: 800; color: #f8b500; line-height: 1.1; margin: 4px 0; }
.macros-row { display: flex; gap: 14px; margin-top: 16px; flex-wrap: wrap; }
.macro-pill {
    background: rgba(255,255,255,0.08); border-radius: 10px;
    padding: 10px 18px; text-align: center; flex: 1; min-width: 72px;
}
.macro-value { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
.macro-name { font-size: 0.68rem; color: #718096; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px; }
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
.vlm-card { border: 1px solid #e9d8fd; border-radius: 10px; padding: 16px; margin: 8px 0; background: #faf5ff; }
.vlm-note { background: #f3e8ff; border-left: 3px solid #805ad5; padding: 8px 14px; border-radius: 0 6px 6px 0; font-size: 0.83rem; color: #553c9a; margin-top: 6px; }
.action-badge { display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 8px; }
.action-corrected { background: #fed7d7; color: #c53030; }
.action-confirmed { background: #c6f6d5; color: #276749; }
.action-unknown   { background: #fefcbf; color: #744210; }
.color-dot { display: inline-block; border-radius: 50%; vertical-align: middle; margin-right: 6px; }
.status-error { color: #e53e3e; font-size: 0.9rem; padding: 4px 0; }
.status-success { color: #276749; font-size: 0.9rem; padding: 4px 0; font-weight: 600; }

.log-btn { margin-top: 8px; }
.dashboard-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 20px 28px; color: white; margin: 8px 0;
    text-align: center;
}
.dashboard-number { font-size: 2.4rem; font-weight: 800; color: #f8b500; }
.dashboard-label { font-size: 0.78rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }
.meal-row { padding: 12px 0; border-bottom: 1px solid #edf2f7; display: flex; justify-content: space-between; align-items: center; }
.meal-row:last-child { border-bottom: none; }
.bar-container { background: #edf2f7; border-radius: 8px; height: 24px; margin: 4px 0; overflow: hidden; position: relative; }
.bar-fill { background: linear-gradient(90deg, #f8b500, #f97316); height: 100%; border-radius: 8px; transition: width 0.3s; }
.bar-label { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 0.75rem; font-weight: 600; color: #2d3748; }
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_dark(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.55

def confidence_badge(conf):
    pct = int(round(conf * 100))
    cls = "badge-green" if conf >= 0.80 else "badge-yellow" if conf >= 0.60 else "badge-red"
    return f'<span class="badge {cls}">{pct}%</span>'

def is_vlm_response(data):
    return "refinement_status" in data


# ── Bbox drawing ──────────────────────────────────────────────────────────────

def draw_bboxes(image_pil, items):
    annotated = image_pil.copy()
    draw = ImageDraw.Draw(annotated)
    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
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
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly0 = max(0, y1 - th - 8)
        ly1 = max(th + 6, y1)
        draw.rectangle([x1, ly0, x1 + tw + 10, ly1], fill=color)
        draw.text((x1 + 5, ly0 + 3), label,
                  fill="white" if _is_dark(color) else "#1a1a1a", font=font)
    return annotated


# ── HTML builders ─────────────────────────────────────────────────────────────

def build_warnings_html(warnings):
    if not warnings:
        return ""
    parts = []
    for w in warnings:
        if w in WARNING_LABELS:
            title, detail = WARNING_LABELS[w]
        else:
            title, detail = w.replace("_", " ").title(), ""
        detail_str = f" — {detail}" if detail else ""
        parts.append(f'<div class="warning-banner">⚠ <strong>{title}</strong>{detail_str}</div>')
    return "\n".join(parts)


def build_totals_html(totals, proc_ms):
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
    <div class="macro-pill"><div class="macro-value">{prot}</div><div class="macro-name">Protein</div></div>
    <div class="macro-pill"><div class="macro-value">{fat}</div><div class="macro-name">Fat</div></div>
    <div class="macro-pill"><div class="macro-value">{carbs}</div><div class="macro-name">Carbs</div></div>
    <div class="macro-pill"><div class="macro-value">{fiber}</div><div class="macro-name">Fiber</div></div>
  </div>
  <div class="proc-time">Processed in {proc_ms:,}ms</div>
</div>"""


def _table_header(conf_label="Confidence"):
    return f"""
<table class="food-table"><thead><tr>
  <th>#</th><th>Food</th><th>Grams</th><th>Calories</th>
  <th>Protein</th><th>Fat</th><th>Carbs</th><th>Fiber</th><th>{conf_label}</th>
</tr></thead><tbody>"""


def build_cv_items_table(items):
    if not items:
        return '<p style="color:#718096;font-size:0.9rem;padding:8px 0">No food items detected.</p>'
    rows = _table_header()
    for idx, item in enumerate(items):
        color = BBOX_COLORS[idx % len(BBOX_COLORS)]
        dot = f'<span class="color-dot" style="width:12px;height:12px;background:{color}"></span>'
        n = item.get("nutrition_total", {})
        conf = float(item.get("classification_confidence", 0))
        grams = float(item.get("estimated_grams", 0))
        rows += f"""<tr>
  <td>{dot}{item.get("item_id", idx+1)}</td>
  <td><strong>{item.get("display_name","")}</strong></td>
  <td>{grams:.0f}g</td>
  <td>{float(n.get("calories_kcal",0)):.0f}</td>
  <td>{float(n.get("protein_g",0)):.1f}g</td>
  <td>{float(n.get("fat_g",0)):.1f}g</td>
  <td>{float(n.get("carbs_g",0)):.1f}g</td>
  <td>{float(n.get("fiber_g",0)):.1f}g</td>
  <td>{confidence_badge(conf)}</td>
</tr>"""
    return rows + "</tbody></table>"


def build_vlm_items_table(items):
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
            food_cell = f'<s style="color:#a0aec0;font-size:0.82rem">{orig_name}</s><br><strong>{refined.get("display_name","")}</strong>'
        else:
            food_cell = f'<strong>{refined.get("display_name","")}</strong>'
        rows += f"""<tr>
  <td>{dot}{item.get("item_id", idx+1)}</td>
  <td>{food_cell}</td>
  <td>{grams:.0f}g</td>
  <td>{float(n.get("calories_kcal",0)):.0f}</td>
  <td>{float(n.get("protein_g",0)):.1f}g</td>
  <td>{float(n.get("fat_g",0)):.1f}g</td>
  <td>{float(n.get("carbs_g",0)):.1f}g</td>
  <td>{float(n.get("fiber_g",0)):.1f}g</td>
  <td>{confidence_badge(conf)}</td>
</tr>"""
    return rows + "</tbody></table>"


def build_vlm_panel(data):
    model     = data.get("vlm_model", "VLM")
    threshold = int(float(data.get("confidence_threshold_used", 0.70)) * 100)
    status    = data.get("refinement_status", "completed")
    header = f'<div class="vlm-header">🤖 AI Refinement &nbsp;·&nbsp; {model} &nbsp;·&nbsp; threshold: {threshold}% &nbsp;·&nbsp; {status}</div>'
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
            food_line = f'<span style="color:#e53e3e;text-decoration:line-through">{orig_name}</span><span style="color:#805ad5;font-weight:bold;margin:0 8px">→</span><strong style="color:#276749">{refined_name}</strong>'
        elif action == "confirmed":
            food_line = f'<strong style="color:#276749">{refined_name}</strong>'
        else:
            food_line = f'<strong style="color:#2d3748">{refined_name}</strong>'
        desc = refined.get("food_description", "")
        desc_html = f'<div style="color:#4a5568;font-size:0.85rem;margin:6px 0">{desc}</div>' if desc else ""
        portion_method = portion.get("portion_method", "").replace("_", " ")
        portion_conf   = int(round(float(portion.get("vlm_confidence", 0)) * 100))
        grams          = float(portion.get("estimated_grams", 0))
        cards.append(f"""<div class="vlm-card">
  <div style="margin-bottom:8px">
    <span class="action-badge action-{action}">{action}</span>
    <span style="color:#718096;font-size:0.85rem">Item #{item.get("item_id","")}</span>
    <span style="float:right;color:#718096;font-size:0.82rem">CV {orig_conf_pct}% → VLM {vlm_conf_pct}%</span>
  </div>
  <div style="font-size:1rem;margin:6px 0">{food_line}</div>
  {desc_html}
  <div style="color:#718096;font-size:0.82rem;margin-top:4px">Portion: {grams:.0f}g via {portion_method} ({portion_conf}% confidence)</div>
</div>""")
    notes = data.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    notes_html = ""
    if notes:
        note_items = "".join(f'<div class="vlm-note">{n}</div>' for n in notes)
        notes_html = f'<div class="section-title">Notes</div>{note_items}'
    return header + "\n".join(cards) + notes_html


# ── Dashboard builders ────────────────────────────────────────────────────────

def build_dashboard_html(user_id: str = "default"):
    today_totals = get_today_totals(user_id)
    today_meals  = get_today_meals(user_id)
    weekly       = get_weekly_data(user_id)  # now (iso_date, label, calories)
    profile      = get_profile(user_id)

    calories_eaten = today_totals["calories_kcal"]
    daily_target   = profile["daily_calories"] if profile else 2000
    calories_left  = max(0, daily_target - calories_eaten)
    progress_pct   = min(100, int((calories_eaten / daily_target) * 100)) if daily_target else 0
    bar_color      = "#e53e3e" if progress_pct > 100 else "#f8b500"
    count          = today_totals["meal_count"]

    # goal date progress
    goal_info = ""
    if profile and profile.get("goal_date"):
        goal_date = date.fromisoformat(profile["goal_date"])
        days_left = (goal_date - date.today()).days
        goal_str  = goal_date.strftime("%d %b %Y")
        if days_left > 0:
            goal_info = f'<div style="font-size:0.78rem;color:#a0aec0;margin-top:8px">🎯 Goal date: <strong style="color:#f8b500">{goal_str}</strong> — {days_left} days remaining</div>'
        elif days_left == 0:
            goal_info = f'<div style="font-size:0.78rem;color:#48bb78;margin-top:8px">🎉 Goal date is today! ({goal_str})</div>'
        else:
            goal_info = f'<div style="font-size:0.78rem;color:#e53e3e;margin-top:8px">⚠ Goal date passed ({goal_str})</div>'

    summary = f"""
<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:16px;padding:24px;color:white;margin-bottom:16px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
    <div>
      <div style="font-size:0.78rem;color:#a0aec0;text-transform:uppercase;letter-spacing:0.06em">Calories today</div>
      <div style="font-size:2.8rem;font-weight:800;color:#f8b500;line-height:1">{calories_eaten:,.0f}</div>
      <div style="font-size:0.85rem;color:#a0aec0">of {daily_target:,.0f} target</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:2rem;font-weight:700;color:{'#48bb78' if calories_left > 0 else '#e53e3e'}">{calories_left:,.0f}</div>
      <div style="font-size:0.78rem;color:#a0aec0">{'remaining' if calories_left > 0 else 'over target'}</div>
    </div>
  </div>
  <div style="background:rgba(255,255,255,0.1);border-radius:8px;height:8px;overflow:hidden">
    <div style="background:{bar_color};height:100%;width:{progress_pct}%;border-radius:8px;transition:width 0.3s"></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:16px">
    <div style="text-align:center"><div style="font-weight:700">{today_totals['protein_g']:.0f}g</div><div style="font-size:0.68rem;color:#a0aec0">PROTEIN</div></div>
    <div style="text-align:center"><div style="font-weight:700">{today_totals['fat_g']:.0f}g</div><div style="font-size:0.68rem;color:#a0aec0">FAT</div></div>
    <div style="text-align:center"><div style="font-weight:700">{today_totals['carbs_g']:.0f}g</div><div style="font-size:0.68rem;color:#a0aec0">CARBS</div></div>
    <div style="text-align:center"><div style="font-weight:700">{count}</div><div style="font-size:0.68rem;color:#a0aec0">MEALS</div></div>
  </div>
  {goal_info}
</div>"""

    # weekly bar chart — now uses label with date
    max_cal = max((c for _, _, c in weekly), default=1) or 1
    bars = ""
    for iso_date, label, cal_val in weekly:
        pct = int((cal_val / max_cal) * 100)
        bars += f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#4a5568;margin-bottom:3px">
    <span>{label}</span><span>{cal_val:,.0f} kcal</span>
  </div>
  <div class="bar-container">
    <div class="bar-fill" style="width:{pct}%"></div>
  </div>
</div>"""

    weekly_section = f"""
<div class="section-title">Last 7 days</div>
<div style="background:white;border-radius:12px;padding:16px;border:1px solid #e2e8f0">
{bars}
</div>"""

    # today's meals list with full date
    today_str = date.today().strftime("%A %d %B %Y")
    if today_meals:
        meal_rows = ""
        for m in today_meals:
            time_str = m["logged_at"][11:16]
            meal_rows += f"""
<div class="meal-row">
  <div>
    <div style="font-weight:600;color:#2d3748">{m["meal_name"]}</div>
    <div style="font-size:0.78rem;color:#718096">{time_str} &nbsp;·&nbsp; {m["protein_g"]:.0f}g protein &nbsp;·&nbsp; {m["carbs_g"]:.0f}g carbs</div>
  </div>
  <div style="text-align:right">
    <div style="font-weight:700;color:#f8b500">{m["calories_kcal"]:.0f} kcal</div>
  </div>
</div>"""
        meals_section = f'<div class="section-title">Today\'s meals — {today_str}</div><div style="background:white;border-radius:12px;padding:16px;border:1px solid #e2e8f0">{meal_rows}</div>'
    else:
        meals_section = f'<div style="color:#718096;text-align:center;padding:24px">No meals logged today ({today_str}). Analyze a meal and click Log Meal!</div>'

    return summary + weekly_section + meals_section


# ── Gradio callbacks ──────────────────────────────────────────────────────────

# Store last result for logging
_last_result = {"data": None}

def _empty_return(msg=""):
    hidden = gr.update(visible=False)
    status = f'<p class="status-error">{msg}</p>' if msg else ""
    return (
        gr.update(value=None, visible=False),
        gr.update(value=""),
        hidden,
        gr.update(value=""),
        hidden,
        gr.update(value=""),
        hidden,
        status,
        gr.update(visible=False),  # log_btn
        gr.update(visible=False),  # disagree_btn
    )


def analyze_image(input_image):
    if input_image is None:
        return (
        gr.update(value=annotated, visible=annotated is not None),
        gr.update(value=w_html),
        gr.update(visible=bool(warnings)),
        gr.update(value=results_html),
        gr.update(visible=True),
        gr.update(value=vlm_html),
        gr.update(visible=vlm),
        "",
        gr.update(visible=True),   # log_btn
        gr.update(visible=True),   # disagree_btn
        )

    buf = BytesIO()
    input_image.save(buf, format="JPEG", quality=92)
    buf.seek(0)

    try:
        resp = requests.post(API_URL, files={"file": ("image.jpg", buf, "image/jpeg")}, timeout=180)
    except requests.ConnectionError:
        return _empty_return(f"Cannot connect to backend at <code>{API_URL}</code>. Is it running?")
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
        _last_result["data"] = data
        vlm      = is_vlm_response(data)
        items    = data.get("items", [])
        warnings = data.get("warnings", [])

        annotated = None
        if not vlm and items:
            try:
                annotated = draw_bboxes(input_image, items)
            except Exception:
                pass

        w_html       = build_warnings_html(warnings)
        totals_html  = build_totals_html(data.get("totals", {}), data.get("processing_time_ms", 0))
        table_html   = build_vlm_items_table(items) if vlm else build_cv_items_table(items)
        results_html = totals_html + '<div class="section-title">Per-item Breakdown</div>' + table_html
        vlm_html     = build_vlm_panel(data) if vlm else ""

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
        gr.update(visible=True),  # show log button
    )


def do_log_meal(user_id):
    data = _last_result.get("data")
    if not data:
        return '<p class="status-error">No meal to log.</p>', build_dashboard_html(user_id)
    totals = data.get("totals", {})
    items  = data.get("items", [])
    vlm    = is_vlm_response(data)
    source = "vlm" if vlm else "cv"
    log_meal(totals, items, user_id=user_id, source=source)
    _last_result["data"] = None
    return '<p class="status-success">✓ Meal logged!</p>', build_dashboard_html(user_id)


def refresh_dashboard(user_id):
    return build_dashboard_html(user_id)

def build_profile_result_html(calc, profile):
    direction = "lose" if calc["weekly_change_kg"] < 0 else "gain"
    arrow = "↓" if calc["weekly_change_kg"] < 0 else "↑"
    color = "#48bb78" if direction == "lose" else "#f6ad55"

    goal_date = calc.get("goal_date", "")
    try:
        goal_display = date.fromisoformat(goal_date).strftime("%d %b %Y")
    except Exception:
        goal_display = goal_date

    return f"""
<div style="margin-top:16px">
  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:16px">
    <div class="dashboard-card">
      <div class="dashboard-number" style="color:#f8b500">{calc['daily_calories']}</div>
      <div class="dashboard-label">daily calorie target</div>
    </div>
    <div class="dashboard-card">
      <div class="dashboard-number" style="color:{color}">{arrow} {abs(calc['weekly_change_kg'])}kg</div>
      <div class="dashboard-label">expected per week</div>
    </div>
    <div class="dashboard-card">
      <div class="dashboard-number" style="font-size:1.4rem">{goal_display}</div>
      <div class="dashboard-label">goal date ({calc['timeframe_weeks']}w away)</div>
    </div>
    <div class="dashboard-card">
      <div class="dashboard-number" style="font-size:1.8rem">{calc['tdee']}</div>
      <div class="dashboard-label">maintenance calories</div>
    </div>
  </div>
  <div style="background:white;border-radius:12px;padding:16px;border:1px solid #e2e8f0">
    <div class="section-title">Daily macro targets</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:8px">
      <div style="text-align:center;padding:12px;background:#f7fafc;border-radius:8px">
        <div style="font-size:1.5rem;font-weight:700;color:#3182ce">{calc['protein_g']}g</div>
        <div style="font-size:0.75rem;color:#718096;text-transform:uppercase">Protein</div>
      </div>
      <div style="text-align:center;padding:12px;background:#f7fafc;border-radius:8px">
        <div style="font-size:1.5rem;font-weight:700;color:#e53e3e">{calc['fat_g']}g</div>
        <div style="font-size:0.75rem;color:#718096;text-transform:uppercase">Fat</div>
      </div>
      <div style="text-align:center;padding:12px;background:#f7fafc;border-radius:8px">
        <div style="font-size:1.5rem;font-weight:700;color:#38a169">{calc['carbs_g']}g</div>
        <div style="font-size:0.75rem;color:#718096;text-transform:uppercase">Carbs</div>
      </div>
    </div>
    <div style="margin-top:12px;font-size:0.78rem;color:#718096;text-align:center">
      BMR: {calc['bmr']} kcal &nbsp;·&nbsp; TDEE: {calc['tdee']} kcal &nbsp;·&nbsp; 
      Adjustment: {calc['daily_adjustment']:+d} kcal/day
    </div>
  </div>
</div>"""

def on_save_profile(age, gender, height, weight, goal_weight, goal_date, activity, user_id):
    # validate date format
    try:
        date.fromisoformat(goal_date)
    except ValueError:
        return '<p class="status-error">Invalid date format. Use YYYY-MM-DD (e.g. 2025-12-31)</p>', "", gr.update()
    try:
        calc = save_profile(int(age), gender, float(height), float(weight),
                            float(goal_weight), goal_date, activity,
                            user_id=user_id)
        profile = get_profile(user_id)
        return (
            '<p class="status-success">✓ Profile saved!</p>',
            build_profile_result_html(calc, profile),
            build_dashboard_html(user_id)
        )
    except Exception as e:
        return f'<p class="status-error">Error: {e}</p>', "", gr.update()

def on_update_weight(new_weight, user_id):
    try:
        calc = update_weight(float(new_weight), user_id=user_id)
        if not calc:
            return '<p class="status-error">Set up your profile first.</p>'
        return f'<p class="status-success">✓ Weight updated! New target: {calc["daily_calories"]} kcal</p>'
    except Exception as e:
        return f'<p class="status-error">Error: {e}</p>'
    

def do_login(username, password):
    success, result = login_user(username, password)
    if not success:
        return (
            gr.update(),           # current_user unchanged
            gr.update(),           # auth_panel unchanged
            gr.update(),           # main_panel unchanged
            f'<p class="status-error">{result}</p>',
            gr.update(),
        )
    # load profile for welcome message
    profile = get_profile(result)
    welcome = f'<div style="text-align:right;padding:4px 0;font-size:0.85rem;color:#718096">Logged in as <strong>{result}</strong></div>'
    return (
        result,                                    # current_user = username
        gr.update(visible=False),                  # hide auth_panel
        gr.update(visible=True),                   # show main_panel
        "",                                        # clear login status
        welcome,                                   # welcome message
    )

def do_register(username, password, confirm):
    if password != confirm:
        return gr.update(), f'<p class="status-error">Passwords do not match.</p>'
    success, result = register_user(username, password)
    if not success:
        return gr.update(), f'<p class="status-error">{result}</p>'
    return gr.update(), f'<p class="status-success">✓ Account created! Go to Login tab.</p>'

def do_logout():
    return (
        "",                       # clear current_user
        gr.update(visible=True),  # show auth_panel
        gr.update(visible=False), # hide main_panel
        "",                       # clear welcome
    )

def load_dashboard(username):
    if not username:
        return ""
    return build_dashboard_html(username)

def load_profile_tab(username):
    if not username:
        return "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""
    profile = get_profile(username)
    result_html = ""
    if profile:
        calc = calculate_profile(
            profile["age"], profile["gender"], profile["height_cm"],
            profile["current_weight_kg"], profile["goal_weight_kg"],
            profile.get("goal_date") or (date.today().replace(year=date.today().year + 1)).isoformat(),
            profile["activity_level"]
        )
        result_html = build_profile_result_html(calc, profile)
    return (
        result_html,
        gr.update(value=profile["age"] if profile else 25),
        gr.update(value=profile["gender"] if profile else "Male"),
        gr.update(value=profile["height_cm"] if profile else 175),
        gr.update(value=profile["current_weight_kg"] if profile else 70),
        gr.update(value=profile["goal_weight_kg"] if profile else 65),
        gr.update(value=profile.get("goal_date") if profile else (date.today().replace(year=date.today().year + 1)).isoformat()),
    )

# ── LLM Chat callbacks ────────────────────────────────────────────────────────

def open_correction_chat(user_id):
    """Open the correction chat panel with initial message."""
    data = _last_result.get("data")
    if not data:
        return (
            gr.update(visible=False),
            [],
            gr.update(value=""),
        )
    items = data.get("items", [])
    is_vlm = is_vlm_response(data)

    # Build summary for first message
    lines = ["I see the pipeline detected:"]
    for item in items:
        if is_vlm:
            name = item.get("refined", {}).get("display_name", "Unknown")
            grams = item.get("portion", {}).get("estimated_grams", 0)
        else:
            name = item.get("display_name", "Unknown")
            grams = item.get("estimated_grams", 0)
        lines.append(f"• {name} ({grams:.0f}g)")
    lines.append("\nWhat would you like to change?")

    initial_msg = "\n".join(lines)
    history = [{"role": "assistant", "content": initial_msg}]

    return (
        gr.update(visible=True),
        history,
        gr.update(value=_history_to_html(history)),
    )

def send_correction_message(user_message, history, user_id):
    """Send a message in the correction chat."""
    if not user_message.strip():
        return history, gr.update(), gr.update(visible=False), gr.update(value="")

    data = _last_result.get("data")
    if not data:
        return history, gr.update(), gr.update(visible=False), gr.update(value="")

    # Check turn limit
    user_turns = sum(1 for m in history if m["role"] == "user")
    if user_turns >= MAX_TURNS:
        history = history + [{"role": "assistant", "content": "We've reached the maximum number of exchanges. Please confirm what you'd like to log or start over."}]
        return history, gr.update(value=_history_to_html(history)), gr.update(visible=False), gr.update(value="")

    try:
        new_history, log_data = chat_correction(history, user_message, data)
    except Exception as e:
        new_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"Sorry, I had an error: {str(e)}"}
        ]
        log_data = None

    show_confirm = log_data is not None
    confirm_html = ""
    if log_data:
        t = log_data.get("totals", {})
        confirm_html = f"""
<div style="background:#f0fff4;border:2px solid #48bb78;border-radius:12px;padding:16px;margin-top:8px">
  <div style="font-weight:700;color:#276749;margin-bottom:8px">✓ Ready to log: {log_data.get('meal_name', 'Meal')}</div>
  <div style="font-size:0.85rem;color:#4a5568">{log_data.get('items_description', '')}</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px">
    <div style="text-align:center"><div style="font-weight:700;color:#f8b500">{t.get('calories_kcal', 0):.0f}</div><div style="font-size:0.7rem;color:#718096">KCAL</div></div>
    <div style="text-align:center"><div style="font-weight:700">{t.get('protein_g', 0):.0f}g</div><div style="font-size:0.7rem;color:#718096">PROTEIN</div></div>
    <div style="text-align:center"><div style="font-weight:700">{t.get('fat_g', 0):.0f}g</div><div style="font-size:0.7rem;color:#718096">FAT</div></div>
    <div style="text-align:center"><div style="font-weight:700">{t.get('carbs_g', 0):.0f}g</div><div style="font-size:0.7rem;color:#718096">CARBS</div></div>
  </div>
</div>"""

    return (
        new_history,
        gr.update(value=_history_to_html(new_history)),
        gr.update(visible=show_confirm, value=confirm_html),
        gr.update(value=""),
    )

def confirm_correction_log(history, user_id):
    """Log the corrected meal from chat."""
    data = _last_result.get("data")
    # Find the last log_data in history
    for msg in reversed(history):
        if msg["role"] == "assistant":
            from llm_chat import _extract_log_data
            log_data = _extract_log_data(msg["content"])
            if log_data:
                totals = log_data.get("totals", {})
                meal_name = log_data.get("meal_name", "Corrected meal")
                log_meal(totals, [], user_id=user_id, source="llm_correction", meal_name=meal_name)
                return (
                    gr.update(visible=False),   # hide chat panel
                    [],                          # clear history
                    gr.update(value=""),         # clear chat display
                    gr.update(visible=False),    # hide confirm
                    '<p class="status-success">✓ Corrected meal logged!</p>',
                    build_dashboard_html(user_id),
                )
    return gr.update(), history, gr.update(), gr.update(), gr.update(), gr.update()

# Manual logging chat
def open_manual_chat():
    return gr.update(visible=True), [], gr.update(value="")

def send_manual_message(user_message, history, user_id):
    if not user_message.strip():
        return history, gr.update(), gr.update(visible=False), gr.update(value="")

    user_turns = sum(1 for m in history if m["role"] == "user")
    if user_turns >= MAX_TURNS:
        history = history + [{"role": "assistant", "content": "We've reached the maximum exchanges. Please confirm to log."}]
        return history, gr.update(value=_history_to_html(history)), gr.update(visible=False), gr.update(value="")

    default_date = date.today().isoformat()
    try:
        new_history, log_data = chat_manual(history, user_message, default_date)
    except Exception as e:
        new_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"Sorry, I had an error: {str(e)}"}
        ]
        log_data = None

    show_confirm = log_data is not None
    confirm_html = ""
    if log_data:
        t = log_data.get("totals", {})
        log_date = log_data.get("log_date", date.today().isoformat())
        confirm_html = f"""
<div style="background:#f0fff4;border:2px solid #48bb78;border-radius:12px;padding:16px;margin-top:8px">
  <div style="font-weight:700;color:#276749;margin-bottom:8px">✓ Ready to log: {log_data.get('meal_name', 'Meal')}</div>
  <div style="font-size:0.85rem;color:#4a5568">{log_data.get('items_description', '')} — {log_date}</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px">
    <div style="text-align:center"><div style="font-weight:700;color:#f8b500">{t.get('calories_kcal', 0):.0f}</div><div style="font-size:0.7rem;color:#718096">KCAL</div></div>
    <div style="text-align:center"><div style="font-weight:700">{t.get('protein_g', 0):.0f}g</div><div style="font-size:0.7rem;color:#718096">PROTEIN</div></div>
    <div style="text-align:center"><div style="font-weight:700">{t.get('fat_g', 0):.0f}g</div><div style="font-size:0.7rem;color:#718096">FAT</div></div>
    <div style="text-align:center"><div style="font-weight:700">{t.get('carbs_g', 0):.0f}g</div><div style="font-size:0.7rem;color:#718096">CARBS</div></div>
  </div>
</div>"""

    return (
        new_history,
        gr.update(value=_history_to_html(new_history)),
        gr.update(visible=show_confirm, value=confirm_html),
        gr.update(value=""),
    )

def confirm_manual_log(history, user_id):
    for msg in reversed(history):
        if msg["role"] == "assistant":
            from llm_chat import _extract_log_data
            log_data = _extract_log_data(msg["content"])
            if log_data:
                totals = log_data.get("totals", {})
                meal_name = log_data.get("meal_name", "Manual meal")
                log_date = log_data.get("log_date", date.today().isoformat())
                # log with custom date
                import json as _json
                from datetime import datetime as _dt
                logged_at = f"{log_date}T12:00:00"
                from db import get_conn
                with get_conn() as conn:
                    conn.execute("""
                        INSERT INTO meals
                        (user_id, logged_at, meal_name, calories_kcal, protein_g,
                         fat_g, carbs_g, fiber_g, items_json, source)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (
                        user_id, logged_at, meal_name,
                        totals.get("calories_kcal", 0),
                        totals.get("protein_g", 0),
                        totals.get("fat_g", 0),
                        totals.get("carbs_g", 0),
                        totals.get("fiber_g", 0),
                        _json.dumps([]),
                        "manual",
                    ))
                return (
                    gr.update(visible=False),
                    [],
                    gr.update(value=""),
                    gr.update(visible=False),
                    '<p class="status-success">✓ Meal logged manually!</p>',
                    build_dashboard_html(user_id),
                )
    return gr.update(), history, gr.update(), gr.update(), gr.update(), gr.update()

def _history_to_html(history: list) -> str:
    """Convert chat history to HTML display."""
    if not history:
        return ""
    parts = []
    for msg in history:
        if msg["role"] == "user":
            parts.append(f"""
<div style="display:flex;justify-content:flex-end;margin:8px 0">
  <div style="background:#4a00e0;color:white;border-radius:12px 12px 2px 12px;padding:10px 14px;max-width:80%;font-size:0.88rem">
    {msg["content"]}
  </div>
</div>""")
        else:
            # Strip JSON blocks from display
            display_text = msg["content"]
            if "```json" in display_text:
                display_text = display_text[:display_text.find("```json")].strip()
                if not display_text:
                    display_text = "I've prepared the nutrition summary below — does this look correct?"
            display_text = display_text.replace("\n", "<br>")
            parts.append(f"""
<div style="display:flex;justify-content:flex-start;margin:8px 0">
  <div style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:12px 12px 12px 2px;padding:10px 14px;max-width:80%;font-size:0.88rem;color:#2d3748">
    {display_text}
  </div>
</div>""")
    return f'<div style="height:300px;overflow-y:auto;padding:8px">' + "".join(parts) + "</div>"

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="Food Calorie Estimator") as demo:

    # session state — stores logged-in username, empty string = not logged in
    current_user = gr.State("")
    correction_history = gr.State([])
    manual_history     = gr.State([])

    gr.HTML("""
        <div style="text-align:center;padding:28px 0 12px">
          <h1 style="font-size:2rem;font-weight:800;color:#1a202c;margin:0">Food Calorie Estimator</h1>
          <p style="color:#718096;margin:8px 0 0;font-size:1rem">Upload or photograph your meal for instant nutrition analysis</p>
        </div>
    """)

    # ── Auth panel (shown when not logged in) ─────────────────────────────────
    with gr.Column(visible=True) as auth_panel:
        with gr.Tabs():
            with gr.Tab("🔑 Login"):
                login_user_input = gr.Textbox(label="Username", placeholder="your username")
                login_pass_input = gr.Textbox(label="Password", type="password", placeholder="your password")
                login_btn        = gr.Button("Login", variant="primary")
                login_status     = gr.HTML("")

            with gr.Tab("📝 Register"):
                reg_user_input = gr.Textbox(label="Choose a username", placeholder="min 3 characters")
                reg_pass_input = gr.Textbox(label="Choose a password", type="password", placeholder="min 6 characters")
                reg_pass_confirm = gr.Textbox(label="Confirm password", type="password")
                reg_btn        = gr.Button("Create Account", variant="primary")
                reg_status     = gr.HTML("")

    # ── Main app (shown when logged in) ──────────────────────────────────────
    with gr.Column(visible=False) as main_panel:
        
        welcome_html = gr.HTML("")

        with gr.Tabs():
            with gr.Tab("📷 Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(sources=["upload", "webcam"], type="pil", label="Your meal", height=380)
                        analyze_btn = gr.Button("Analyze Meal", variant="primary", size="lg")
                        log_btn      = gr.Button("📋 Log Meal", variant="secondary", size="sm", visible=False)
                        disagree_btn = gr.Button("✏️ Disagree? Adjust with AI", variant="secondary", size="sm", visible=False)
                        status_html = gr.HTML("")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Detected items", interactive=False, height=380, visible=False)

                with gr.Column(visible=False) as warnings_panel:
                    warnings_html = gr.HTML("")

                with gr.Column(visible=False) as results_panel:
                    results_html = gr.HTML("")

                with gr.Column(visible=False) as vlm_panel:
                    vlm_html = gr.HTML("")

                with gr.Column(visible=False) as correction_chat_panel:
                    gr.HTML("<div class='section-title'>💬 Adjust with AI</div>")
                    correction_chat_display = gr.HTML("")
                    correction_input = gr.Textbox(
                        placeholder="Tell me what's wrong or what you actually ate...",
                        label="",
                        show_label=False
                    )
                    with gr.Row():
                        correction_send_btn    = gr.Button("Send", variant="primary", size="sm")
                        correction_close_btn   = gr.Button("Cancel", size="sm")
                    correction_confirm_html = gr.HTML("", visible=False)
                    correction_confirm_btn  = gr.Button("✓ Log this meal", variant="primary")

            with gr.Tab("📊 Dashboard"):
                with gr.Row():
                    refresh_btn    = gr.Button("🔄 Refresh", size="sm")
                    manual_log_btn = gr.Button("➕ Log food manually", variant="secondary", size="sm")
                dashboard_html = gr.HTML("")

                with gr.Column(visible=False) as manual_chat_panel:
                    gr.HTML("<div class='section-title'>💬 Log food manually</div>")
                    manual_chat_display = gr.HTML("")
                    manual_input = gr.Textbox(
                        placeholder="Describe what you ate, e.g. 'bowl of oatmeal with banana for breakfast'",
                        label="",
                        show_label=False
                    )
                    with gr.Row():
                        manual_send_btn  = gr.Button("Send", variant="primary", size="sm")
                        manual_close_btn = gr.Button("Cancel", size="sm")
                    manual_confirm_html = gr.HTML("", visible=False)
                    manual_confirm_btn  = gr.Button("✓ Log this meal", variant="primary")
                    manual_status_html  = gr.HTML("")

            with gr.Tab("👤 Profile"):
                profile_intro = gr.HTML("<div style='margin-bottom:16px;font-size:0.9rem;color:#718096'>Set up your profile to get personalized calorie targets.</div>")

                with gr.Row():
                    age_input    = gr.Number(label="Age", value=25, precision=0)
                    gender_input = gr.Radio(["Male", "Female"], label="Gender", value="Male")

                with gr.Row():
                    height_input = gr.Number(label="Height (cm)", value=175, precision=0)
                    weight_input = gr.Number(label="Current weight (kg)", value=70)

                with gr.Row():
                    goal_weight_input = gr.Number(label="Goal weight (kg)", value=65)
                    goal_date_input   = gr.Textbox(
                        label="Goal date (YYYY-MM-DD)",
                        placeholder="e.g. 2025-12-31",
                        value=(date.today().replace(year=date.today().year + 1)).isoformat()
                    )

                activity_input = gr.Dropdown(
                    list(ACTIVITY_MULTIPLIERS.keys()),
                    label="Activity level",
                    value=list(ACTIVITY_MULTIPLIERS.keys())[2]
                )

                save_profile_btn = gr.Button("💾 Save Profile & Calculate", variant="primary")
                profile_status   = gr.HTML("")
                profile_result   = gr.HTML("")

                gr.HTML("<div class='section-title' style='margin-top:24px'>Update current weight</div>")
                with gr.Row():
                    new_weight_input  = gr.Number(label="New weight (kg)", precision=1)
                    update_weight_btn = gr.Button("Update weight", size="sm")
                update_weight_status = gr.HTML("")

            with gr.Tab("🚪 Logout"):
                logout_btn = gr.Button("Logout", variant="stop")

    # Auth handlers
    login_btn.click(
        fn=do_login,
        inputs=[login_user_input, login_pass_input],
        outputs=[current_user, auth_panel, main_panel, login_status, welcome_html],
    )

    reg_btn.click(
        fn=do_register,
        inputs=[reg_user_input, reg_pass_input, reg_pass_confirm],
        outputs=[current_user, reg_status],
    )

    logout_btn.click(
        fn=do_logout,
        inputs=[],
        outputs=[current_user, auth_panel, main_panel, welcome_html],
    )

    # Load dashboard when tab is opened
    refresh_btn.click(
        fn=load_dashboard,
        inputs=[current_user],
        outputs=[dashboard_html],
    )

    log_btn.click(
        fn=do_log_meal,
        inputs=[current_user],
        outputs=[status_html, dashboard_html],
    )

    save_profile_btn.click(
        fn=on_save_profile,
        inputs=[age_input, gender_input, height_input, weight_input,
                    goal_weight_input, goal_date_input, activity_input, current_user],
        outputs=[profile_status, profile_result, dashboard_html],
        )

    update_weight_btn.click(
        fn=on_update_weight,
        inputs=[new_weight_input, current_user],
        outputs=[update_weight_status],
    )

    # Correction chat
    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[
            output_image, warnings_html, warnings_panel,
            results_html, results_panel,
            vlm_html, vlm_panel,
            status_html, log_btn, disagree_btn,
        ],
    )

    disagree_btn.click(
        fn=open_correction_chat,
        inputs=[current_user],
        outputs=[correction_chat_panel, correction_history, correction_chat_display],
    )

    correction_send_btn.click(
        fn=send_correction_message,
        inputs=[correction_input, correction_history, current_user],
        outputs=[correction_history, correction_chat_display,
                 correction_confirm_html, correction_input],
    )

    correction_confirm_btn.click(
        fn=confirm_correction_log,
        inputs=[correction_history, current_user],
        outputs=[correction_chat_panel, correction_history, correction_chat_display,
                 correction_confirm_html, status_html, dashboard_html],
    )

    correction_close_btn.click(
        fn=lambda: (gr.update(visible=False), [], gr.update(value="")),
        inputs=[],
        outputs=[correction_chat_panel, correction_history, correction_chat_display],
    )

    # Manual logging chat
    manual_log_btn.click(
        fn=open_manual_chat,
        inputs=[],
        outputs=[manual_chat_panel, manual_history, manual_chat_display],
    )

    manual_send_btn.click(
        fn=send_manual_message,
        inputs=[manual_input, manual_history, current_user],
        outputs=[manual_history, manual_chat_display,
                 manual_confirm_html, manual_input],
    )

    manual_confirm_btn.click(
        fn=confirm_manual_log,
        inputs=[manual_history, current_user],
        outputs=[manual_chat_panel, manual_history, manual_chat_display,
                 manual_confirm_html, manual_status_html, dashboard_html],
    )

    manual_close_btn.click(
        fn=lambda: (gr.update(visible=False), [], gr.update(value="")),
        inputs=[],
        outputs=[manual_chat_panel, manual_history, manual_chat_display],
    )



if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        show_error=True,
        show_api=False,
    )
