"""
config.py ─ Central configuration for Disaster Risk Prediction System
=====================================================================
All constants, thresholds, API settings, and city lists live here.
Change OWM_API_KEY to your real key before running the app.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# 1.  OPENWEATHERMAP  (PyOWM backend)
# ──────────────────────────────────────────────────────────────────────────────
# FREE key  →  https://openweathermap.org/api  (activates within 2 hours)
# PyOWM uses this key internally for every API call — no raw URLs needed.
OWM_API_KEY = os.environ.get("OWM_API_KEY", "268ced99c734e4434570fc08ec92d95a")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  MODEL SETTINGS  (Random Forest)
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_FOREST_PARAMS = {
    "n_estimators":    300,    # more trees → more stable predictions
    "max_depth":       14,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "class_weight":   "balanced",   # handles class imbalance automatically
    "random_state":    42,
    "n_jobs":         -1,           # use all CPU cores
}

MODEL_SAVE_PATH = "models/disaster_models.pkl"

# Feature columns fed into the ML models  (order matters!)
FEATURE_COLUMNS = [
    "rainfall",
    "humidity",
    "wind_speed",
    "temperature",
    "pressure",
]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  RISK THRESHOLDS  (IMD-based)
# ──────────────────────────────────────────────────────────────────────────────
# Used BOTH for synthetic label generation AND for the rule-based baseline.

# IMD heavy-rainfall classification:
#   Heavy Rain       ≥  64.5 mm/day  →  ~8 mm/3h
#   Very Heavy Rain  ≥ 115.6 mm/day  →  ~14 mm/3h
#   Extremely Heavy  ≥ 204.4 mm/day  →  ~25 mm/3h
FLOOD_THRESHOLDS = {
    "HIGH":   {"rainfall": 25.0, "humidity": 88},
    "MEDIUM": {"rainfall":  8.0, "humidity": 72},
}

# IMD cyclone wind-speed classification:
#   Depression        <  17 m/s
#   Cyclonic Storm    17–24 m/s
#   Severe Cyclone    24–32 m/s
#   Very Severe       32–46 m/s
CYCLONE_WIND_THRESHOLD    = 17.2   # m/s  (Cyclonic Storm lower bound)
CYCLONE_PRESSURE_THRESHOLD= 990    # hPa  (surface low pressure)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  INDIAN CITIES
# ──────────────────────────────────────────────────────────────────────────────
INDIA_CITIES = sorted([
    "Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad",
    "Bengaluru", "Ahmedabad", "Pune", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Patna", "Vadodara",
    "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad",
    "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad",
    "Coimbatore", "Vijayawada", "Madurai", "Bhubaneswar", "Kochi",
    "Mangaluru", "Thiruvananthapuram", "Guwahati", "Chandigarh", "Raipur",
    "Ranchi", "Jodhpur", "Amritsar", "Jabalpur", "Gwalior",
    "Tiruchirappalli", "Hubli", "Bareilly", "Moradabad", "Mysuru",
])

# ──────────────────────────────────────────────────────────────────────────────
# 5.  DATA GENERATION SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SYNTHETIC_SAMPLES = 8000
SYNTHETIC_SEED            = 42

# ──────────────────────────────────────────────────────────────────────────────
# 6.  REAL DATASET  (optional — place CSV here and set path)
# ──────────────────────────────────────────────────────────────────────────────
# If you have a real CSV, set REAL_DATASET_PATH to its file path.
# The CSV must contain these columns (exact names):
#   rainfall, humidity, wind_speed, temperature, pressure,
#   flood_risk (LOW/MEDIUM/HIGH),
#   cyclone_risk (YES/NO),
#   overall_risk (LOW/MEDIUM/HIGH)
REAL_DATASET_PATH = None   # e.g.  "data/india_weather_disaster.csv"