# Food Calorie Estimator — User Manual

**Version:** 1.0  
**Last Updated:** May 2026

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Analyzing a Meal](#2-analyzing-a-meal)
3. [Understanding Results](#3-understanding-results)
4. [Logging Meals](#4-logging-meals)
5. [AI Correction Chat](#5-ai-correction-chat)
6. [Manual Meal Logging](#6-manual-meal-logging)
7. [Dashboard](#7-dashboard)
8. [Profile Setup](#8-profile-setup)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Getting Started

### Accessing the App

Open your browser and navigate to:
- **Local deployment:** `http://localhost:7860`
- **Live server:** `http://65.109.133.173:7860`

### Creating an Account

1. On the login screen, click the **📝 Register** tab
2. Enter a username (minimum 3 characters)
3. Enter a password (minimum 6 characters)
4. Confirm your password
5. Click **Create Account**
6. Switch to the **🔑 Login** tab to sign in

> **Note:** Your password is stored as a salted SHA-256 hash. It is never stored in plain text.

### Logging In

1. Enter your username and password
2. Click **Login**
3. You will see the main app with your name in the top right

> **Session Note:** Due to Gradio's architecture, your session resets if you close the browser tab. You will need to log in again when you return.

---

## 2. Analyzing a Meal

### Step 1: Go to the Analyze Tab

Click the **📷 Analyze** tab in the main navigation.

### Step 2: Provide a Meal Image

You have two options:

**Upload a photo:**
- Drag and drop an image onto the upload area, or
- Click the upload area to browse your files
- Supported formats: JPG, PNG

**Take a photo (mobile):**
- On a phone, tap the camera icon to use your device camera
- Position the meal so the plate is clearly visible
- Take the photo

### Step 3: Click "Analyze Meal"

Press the **Analyze Meal** button. The system will:

1. Detect food items and the plate in your image
2. Classify each food item
3. Estimate portion sizes in grams
4. Look up nutrition information
5. If confidence is low, run AI refinement

**Wait time:** Usually 1–5 seconds for CV results, 5–15 seconds if VLM refinement is triggered.

### Tips for Best Results

| ✅ Do | ❌ Don't |
|-------|---------|
| Include the plate in the frame | Photograph food from extreme angles |
| Use good lighting | Use dark/blurry photos |
| Keep the camera steady | Have multiple overlapping plates |
| Show distinct food items | Pile everything into an unrecognizable mound |

---

## 3. Understanding Results

### Annotated Image

After analysis, you'll see your image with colored bounding boxes around detected food items. Each box shows:
- The food name
- A confidence percentage (e.g., 87%)

### Nutrition Totals Card

A card showing total nutrition for the entire meal:

```
┌─────────────────────────────┐
│  Total Calories              │
│  625 kcal                    │
│                              │
│  32.5g    30.0g    60.0g    4.8g   │
│  Protein   Fat     Carbs   Fiber  │
│                              │
│  Processed in 1,243ms        │
└─────────────────────────────┘
```

### Per-Item Breakdown Table

A detailed table showing each detected food item:

| Column | Description |
|--------|-------------|
| **#** | Item number with color dot matching the bounding box |
| **Food** | Identified food name |
| **Grams** | Estimated portion size |
| **Calories** | Estimated calories for this item |
| **Protein/Fat/Carbs/Fiber** | Per-item macro values |
| **Confidence** | Color-coded badge: 🟢 ≥80%, 🟡 ≥60%, 🔴 <60% |

### Warning Banners

You may see yellow warning banners:

| Warning | Meaning |
|---------|---------|
| **No plate detected** | Portion estimates may be less accurate |
| **Low confidence — AI refinement used** | The CV pipeline was uncertain, so the VLM was called to improve results |

### AI Refinement Panel

When the VLM is triggered, a purple panel appears showing:

- **Model used** (e.g., Gemini Flash)
- **Per-item cards** with:
  - Action: **confirmed** (CV was right) or **corrected** (VLM changed it)
  - Original vs. refined food name (e.g., ~~fried_rice~~ → **Bibimbap**)
  - CV confidence → VLM confidence comparison
  - Food description
  - Portion estimate and method
- **Notes** from the VLM about its reasoning

---

## 4. Logging Meals

### From Analysis Results

After analyzing a meal:
1. Review the results
2. Click the **📋 Log Meal** button
3. The meal is saved to your daily log with today's date and time
4. A "✓ Meal logged!" confirmation appears

### What Gets Logged

| Field | Value |
|-------|-------|
| Date/Time | Current timestamp |
| Meal name | Auto-generated from detected items |
| Calories | Total kcal from analysis |
| Macros | Protein, fat, carbs, fiber |
| Source | `cv`, `vlm`, `llm_correction`, or `manual` |

### Viewing Logged Meals

Go to the **📊 Dashboard** tab. Today's meals are listed at the bottom with:
- Meal name
- Time logged
- Calories and key macros

---

## 5. AI Correction Chat

If you disagree with the analysis results, you can correct them using natural language.

### Starting a Correction

1. After analysis, click **✏️ Disagree? Adjust with AI**
2. A chat panel opens showing what was detected
3. Type what's wrong or what you actually ate

### Example Corrections

| You Type | What Happens |
|----------|-------------|
| "That's not pizza, it's a quesadilla" | AI updates the food and recalculates nutrition |
| "The portion is too large, it was about 200g" | AI adjusts the grams and nutrition |
| "I only ate half of that" | AI halves the portion size |
| "That's actually two items: rice and chicken" | AI splits into separate items |

### Confirming the Correction

1. After your correction, the AI shows a green confirmation card with updated nutrition
2. Click **✓ Log this meal** to save the corrected values
3. The corrected meal is logged with source `llm_correction`

### Limits

- Maximum 6 back-and-forth messages per correction session
- The AI aims to produce a loggable result within 1–2 exchanges

---

## 6. Manual Meal Logging

Don't have a photo? You can log meals by describing them.

### Starting Manual Logging

1. Go to the **📊 Dashboard** tab
2. Click **➕ Log food manually**
3. A chat panel opens

### Example Descriptions

| Description | AI Understands |
|-------------|---------------|
| "Bowl of oatmeal with banana and honey for breakfast" | Estimates oats + banana + honey nutrition |
| "Grilled chicken breast 200g with side salad" | Standard chicken breast + salad nutrition |
| "Two slices of pepperoni pizza and a coke" | Pizza + cola nutrition |
| "I had a croissant and a latte this morning" | Croissant + latte nutrition |

### Logging for Past Dates

You can specify a date in your description:
- "Yesterday I had pasta bolognese for dinner"
- "Last Monday I ate a burger for lunch"

The AI will use the correct date automatically.

### Confirming

Same as correction — the AI shows a confirmation card, and you click **✓ Log this meal**.

---

## 7. Dashboard

The dashboard provides a daily overview of your nutrition intake.

### Calorie Summary

```
┌─────────────────────────────────────┐
│  Calories today          Remaining  │
│  1,450                   550        │
│  of 2,000 target                    │
│  ███████████████████░░░░░  72%      │
│                                     │
│  85g      45g      120g     3       │
│  PROTEIN   FAT     CARBS   MEALS   │
└─────────────────────────────────────┘
```

### 7-Day History

A bar chart showing your daily calorie intake for the past 7 days, with the most recent day at the bottom.

### Today's Meals List

All meals logged today, showing:
- Meal name
- Time logged
- Calories
- Key macros

### Refreshing

Click **🔄 Refresh** to update the dashboard with the latest data.

---

## 8. Profile Setup

### Why Set Up Your Profile

Without a profile, the system uses a default 2,000 kcal daily target. Setting up your profile gives you a **personalized calorie target** based on your body, goals, and activity level.

### Setting Up Your Profile

1. Go to the **👤 Profile** tab
2. Fill in the fields:

| Field | Description | Example |
|-------|-------------|---------|
| **Age** | Your age in years | 22 |
| **Gender** | Male or Female | Male |
| **Height (cm)** | Your height in centimeters | 175 |
| **Current weight (kg)** | Your current weight | 80 |
| **Goal weight (kg)** | Your target weight | 72 |
| **Goal date** | When you want to reach your goal | 2026-12-31 |
| **Activity level** | Your typical weekly activity | Moderately active |

3. Click **💾 Save Profile & Calculate**

### Activity Levels

| Level | Description | Multiplier |
|-------|-------------|-----------|
| Sedentary | Desk job, little to no exercise | 1.2× |
| Lightly active | Light exercise 1–3 days/week | 1.375× |
| Moderately active | Moderate exercise 3–5 days/week | 1.55× |
| Very active | Hard exercise 6–7 days/week | 1.725× |
| Extremely active | Athlete or physical labor job | 1.9× |

### Profile Results

After saving, you'll see:

- **Daily calorie target** — personalized based on your goals
- **Expected weekly change** — how fast you'll reach your goal (e.g., ↓ 0.5 kg/week)
- **Goal date progress** — days remaining and time frame
- **Maintenance calories** — your TDEE (Total Daily Energy Expenditure)
- **Macro targets** — daily protein, fat, and carbs in grams

### How It's Calculated

1. **BMR** (Basal Metabolic Rate) = Mifflin-St Jeor equation
   - Males: `10 × weight + 6.25 × height − 5 × age + 5`
   - Females: `10 × weight + 6.25 × height − 5 × age − 161`
2. **TDEE** = BMR × activity multiplier
3. **Daily adjustment** = (weight_diff × 7700 kcal/kg) / (days until goal)
4. **Daily target** = TDEE + adjustment (clamped to min 1,200 kcal)
5. **Protein** = 1.8g per kg body weight
6. **Fat** = 25% of daily calories
7. **Carbs** = remaining calories

### Updating Your Weight

As you progress, update your weight in the "Update current weight" section at the bottom of the Profile tab. This recalculates your targets automatically.

---

## 9. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|---------|
| **"Cannot connect to backend"** | Make sure the CV pipeline container is running. If using Docker: `docker-compose up` |
| **Request times out** | First request may take 30–60 seconds while models load. Try again. |
| **No food detected** | Ensure the image clearly shows food on a plate. Try a different angle or better lighting. |
| **Wrong food identified** | Use the "Disagree? Adjust with AI" button to correct it |
| **Portion seems wrong** | Use the AI correction chat to adjust the gram estimate |
| **Session expired** | Gradio sessions reset on page reload. Log in again. |
| **Dashboard is empty** | You need to analyze and log meals first |

### Browser Compatibility

| Browser | Supported |
|---------|-----------|
| Chrome / Edge | ✅ Full support |
| Firefox | ✅ Full support |
| Safari | ✅ Full support |
| Mobile Chrome/Safari | ✅ Full support (including camera) |

### Performance Notes

| Metric | Typical Value |
|--------|--------------|
| CV analysis (no VLM) | 1–3 seconds |
| CV + VLM refinement | 5–15 seconds |
| Image upload (5MB) | < 1 second on broadband |
| Dashboard refresh | < 1 second |

---

## Keyboard Shortcuts

The app runs in a browser, so standard browser shortcuts apply. There are no custom keyboard shortcuts.

---

## Data Privacy

- All data is stored in a local SQLite database on the server
- Images are processed in memory and not permanently stored
- Passwords are salted and hashed (SHA-256)
- No data is shared with third parties beyond Google AI API calls for VLM refinement
