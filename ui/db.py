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
    # return with formatted label AND iso date
    result = []
    for d in days:
        d_obj = date.fromisoformat(d)
        if d_obj == today:
            label = f"Today {d_obj.strftime('%d %b')}"
        else:
            label = d_obj.strftime("%a %d %b")
        result.append((d, label, daily.get(d, 0)))
    return result