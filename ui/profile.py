import sqlite3
import os
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

def init_profile_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profile (
                user_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                height_cm REAL,
                current_weight_kg REAL,
                goal_weight_kg REAL,
                timeframe_weeks INTEGER,
                activity_level TEXT,
                bmr REAL,
                tdee REAL,
                daily_calories REAL,
                protein_g REAL,
                fat_g REAL,
                carbs_g REAL,
                updated_at TEXT
            )
        """)

ACTIVITY_MULTIPLIERS = {
    "Sedentary (desk job, little exercise)": 1.2,
    "Lightly active (1-3 days/week)": 1.375,
    "Moderately active (3-5 days/week)": 1.55,
    "Very active (6-7 days/week)": 1.725,
    "Extremely active (athlete, physical job)": 1.9,
}

def calculate_profile(age, gender, height_cm, current_weight_kg,
                       goal_weight_kg, timeframe_weeks, activity_level):
    if gender == "Male":
        bmr = 10 * current_weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * current_weight_kg + 6.25 * height_cm - 5 * age - 161

    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    tdee = bmr * multiplier

    weight_diff_kg = goal_weight_kg - current_weight_kg
    total_kcal_needed = weight_diff_kg * 7700
    days = timeframe_weeks * 7
    daily_adjustment = total_kcal_needed / days if days > 0 else 0
    daily_adjustment = max(-1000, min(500, daily_adjustment))
    daily_calories = max(1200, round(tdee + daily_adjustment))

    protein_g = round(current_weight_kg * 1.8)
    fat_g     = round(daily_calories * 0.25 / 9)
    carbs_g   = round((daily_calories - protein_g * 4 - fat_g * 9) / 4)

    actual_adjustment = daily_calories - tdee
    weekly_change_kg  = round((actual_adjustment * 7) / 7700, 2)
    est_weeks = round(abs(total_kcal_needed) / abs(actual_adjustment * 7), 1) \
                if abs(actual_adjustment) > 0 else 0

    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "daily_calories": daily_calories,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carbs_g": carbs_g,
        "weekly_change_kg": weekly_change_kg,
        "est_weeks": est_weeks,
        "daily_adjustment": round(daily_adjustment),
    }

def save_profile(age, gender, height_cm, current_weight_kg,
                 goal_weight_kg, timeframe_weeks, activity_level,
                 user_id: str = "default"):
    from datetime import datetime
    calc = calculate_profile(age, gender, height_cm, current_weight_kg,
                             goal_weight_kg, timeframe_weeks, activity_level)
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO profile
            (user_id, age, gender, height_cm, current_weight_kg, goal_weight_kg,
             timeframe_weeks, activity_level, bmr, tdee, daily_calories,
             protein_g, fat_g, carbs_g, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                age=excluded.age, gender=excluded.gender,
                height_cm=excluded.height_cm,
                current_weight_kg=excluded.current_weight_kg,
                goal_weight_kg=excluded.goal_weight_kg,
                timeframe_weeks=excluded.timeframe_weeks,
                activity_level=excluded.activity_level,
                bmr=excluded.bmr, tdee=excluded.tdee,
                daily_calories=excluded.daily_calories,
                protein_g=excluded.protein_g, fat_g=excluded.fat_g,
                carbs_g=excluded.carbs_g, updated_at=excluded.updated_at
        """, (
            user_id, age, gender, height_cm, current_weight_kg, goal_weight_kg,
            timeframe_weeks, activity_level,
            calc["bmr"], calc["tdee"], calc["daily_calories"],
            calc["protein_g"], calc["fat_g"], calc["carbs_g"],
            datetime.now().isoformat()
        ))
    return calc

def get_profile(user_id: str = "default"):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM profile WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None

def update_weight(new_weight_kg, user_id: str = "default"):
    profile = get_profile(user_id)
    if not profile:
        return None
    return save_profile(
        profile["age"], profile["gender"], profile["height_cm"],
        new_weight_kg, profile["goal_weight_kg"],
        profile["timeframe_weeks"], profile["activity_level"],
        user_id=user_id
    )