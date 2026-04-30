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
                id INTEGER PRIMARY KEY CHECK (id = 1),
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
    # BMR — Mifflin-St Jeor
    if gender == "Male":
        bmr = 10 * current_weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * current_weight_kg + 6.25 * height_cm - 5 * age - 161

    # TDEE
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    tdee = bmr * multiplier

    # Daily calorie target
    weight_diff_kg = goal_weight_kg - current_weight_kg
    total_kcal_needed = weight_diff_kg * 7700  # kcal per kg
    days = timeframe_weeks * 7
    daily_adjustment = total_kcal_needed / days if days > 0 else 0

    # Cap adjustment to safe range (-1000 to +500)
    daily_adjustment = max(-1000, min(500, daily_adjustment))
    daily_calories = round(tdee + daily_adjustment)
    daily_calories = max(1200, daily_calories)  # never below 1200

    # Macros (standard split)
    protein_g = round(current_weight_kg * 1.8)        # 1.8g per kg bodyweight
    fat_g     = round(daily_calories * 0.25 / 9)       # 25% of calories from fat
    carbs_g   = round((daily_calories - protein_g * 4 - fat_g * 9) / 4)

    # Weekly weight change
    actual_adjustment = daily_calories - tdee
    weekly_change_kg  = round((actual_adjustment * 7) / 7700, 2)

    # Estimated weeks to goal
    if abs(actual_adjustment) > 0:
        est_weeks = round(abs(total_kcal_needed) / abs(actual_adjustment * 7), 1)
    else:
        est_weeks = 0

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
                 goal_weight_kg, timeframe_weeks, activity_level):
    from datetime import datetime
    calc = calculate_profile(age, gender, height_cm, current_weight_kg,
                             goal_weight_kg, timeframe_weeks, activity_level)
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO profile
            (id, age, gender, height_cm, current_weight_kg, goal_weight_kg,
             timeframe_weeks, activity_level, bmr, tdee, daily_calories,
             protein_g, fat_g, carbs_g, updated_at)
            VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
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
            age, gender, height_cm, current_weight_kg, goal_weight_kg,
            timeframe_weeks, activity_level,
            calc["bmr"], calc["tdee"], calc["daily_calories"],
            calc["protein_g"], calc["fat_g"], calc["carbs_g"],
            datetime.now().isoformat()
        ))
    return calc

def get_profile():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM profile WHERE id = 1").fetchone()
    return dict(row) if row else None

def update_weight(new_weight_kg):
    profile = get_profile()
    if not profile:
        return None
    return save_profile(
        profile["age"], profile["gender"], profile["height_cm"],
        new_weight_kg, profile["goal_weight_kg"],
        profile["timeframe_weeks"], profile["activity_level"]
    )