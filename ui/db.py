import sqlite3
import os
import uuid
from datetime import datetime, date, timedelta
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", "/app/meals.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                logged_at TEXT NOT NULL,
                meal_name TEXT,
                calories_kcal REAL,
                protein_g REAL,
                fat_g REAL,
                carbs_g REAL,
                fiber_g REAL,
                items_json TEXT,
                source TEXT DEFAULT 'cv'
            )
        """)
        try:
            conn.execute("ALTER TABLE meals ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
        except Exception:
            pass

def generate_user_id() -> str:
    return str(uuid.uuid4())

def log_meal(totals: dict, items: list, user_id: str = "default",
             source: str = "cv", meal_name: str = None):
    import json
    now = datetime.now().isoformat()
    name = meal_name or _auto_name(items)
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO meals
            (user_id, logged_at, meal_name, calories_kcal, protein_g, fat_g,
             carbs_g, fiber_g, items_json, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, now, name,
            totals.get("calories_kcal", 0),
            totals.get("protein_g", 0),
            totals.get("fat_g", 0),
            totals.get("carbs_g", 0),
            totals.get("fiber_g", 0),
            json.dumps(items),
            source,
        ))

def _auto_name(items: list) -> str:
    names = []
    for item in items[:3]:
        name = item.get("refined", {}).get("display_name") or item.get("display_name", "")
        if name:
            names.append(name)
    return ", ".join(names) if names else "Meal"

def get_today_meals(user_id: str = "default"):
    today = date.today().isoformat()
    return get_meals_for_date(today, user_id)

def get_meals_for_date(target_date: str, user_id: str = "default"):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM meals
            WHERE DATE(logged_at) = ? AND user_id = ?
            ORDER BY logged_at DESC
        """, (target_date, user_id)).fetchall()
    return [dict(r) for r in rows]

def get_today_totals(user_id: str = "default"):
    today = date.today().isoformat()
    return get_totals_for_date(today, user_id)

def get_totals_for_date(target_date: str, user_id: str = "default"):
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COALESCE(SUM(calories_kcal), 0) as calories_kcal,
                COALESCE(SUM(protein_g), 0) as protein_g,
                COALESCE(SUM(fat_g), 0) as fat_g,
                COALESCE(SUM(carbs_g), 0) as carbs_g,
                COALESCE(SUM(fiber_g), 0) as fiber_g,
                COUNT(*) as meal_count
            FROM meals
            WHERE DATE(logged_at) = ? AND user_id = ?
        """, (target_date, user_id)).fetchone()
    return dict(row)

def get_weekly_data(user_id: str = "default"):
    today = date.today()
    days = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DATE(logged_at) as day,
                   COALESCE(SUM(calories_kcal), 0) as calories
            FROM meals
            WHERE DATE(logged_at) >= ? AND user_id = ?
            GROUP BY DATE(logged_at)
        """, (days[0], user_id)).fetchall()
    daily = {r["day"]: r["calories"] for r in rows}
    result = []
    for d in days:
        d_obj = date.fromisoformat(d)
        if d_obj == today:
            label = f"Today {d_obj.strftime('%d %b')}"
        else:
            label = d_obj.strftime("%a %d %b")
        result.append((d, label, daily.get(d, 0)))
    return result

def delete_meal(meal_id: int, user_id: str = "default"):
    with get_conn() as conn:
        conn.execute("DELETE FROM meals WHERE id = ? AND user_id = ?",
                     (meal_id, user_id))