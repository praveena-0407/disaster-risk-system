"""
data_generator.py ─ Training Dataset for Disaster Risk Prediction
=================================================================
This module handles BOTH synthetic data generation AND real dataset loading.

WHY SYNTHETIC DATA?
───────────────────
Labeled disaster datasets (weather features + risk labels together) are rare.
Real datasets from IMD, NDMA etc. provide raw weather measurements but rarely
include pre-labeled risk levels. We simulate those labels using IMD thresholds,
creating a statistically realistic training corpus.

HOW TO GET REAL DATASETS (Complete Guide)
──────────────────────────────────────────

Option 1 ── Kaggle  [EASIEST, no registration needed to browse]
    Search: https://www.kaggle.com/datasets/search?q=flood+india+weather
    Recommended datasets:
      • "Kerala Flood Dataset 2018"
      • "India Weather Historical Data"
      • "Cyclone Data India"
    Download via CLI:
        pip install kaggle
        # place kaggle.json from your Kaggle account → ~/.kaggle/kaggle.json
        kaggle datasets download -d <owner>/<dataset-name>
        unzip *.zip -d data/

Option 2 ── IMD (India Meteorological Department)
    URL:   https://mausam.imd.gov.in/
    URL:   https://imdpune.gov.in/lrfindex.php
    Data:  District-wise daily rainfall, station-wise surface observations
    Steps:
        1. Go to the URL above
        2. Navigate to "Data Services" or "Climatological Data"
        3. Register (free for students/researchers)
        4. Download CSV for your state/district

Option 3 ── NASA Giovanni (Satellite Rainfall)
    URL:   https://giovanni.gsfc.nasa.gov/giovanni/
    Data:  GPM IMERG precipitation, 30-min intervals, 0.1° grid, all India
    Steps:
        1. Select "Time Averaged Map"
        2. Date range → your period
        3. Variable → "GPM IMERG Final Precipitation"
        4. Draw bounding box over India
        5. Export as CSV/NetCDF

Option 4 ── EM-DAT (Global Disaster Database)
    URL:   https://www.emdat.be/
    Data:  Historical disaster events (type, date, deaths, damage, location)
    Steps:
        1. Register (free, academic email preferred)
        2. Filter: Country=India, Type=Flood / Storm
        3. Export → CSV

Option 5 ── NOAA ISD (Hourly Surface Observations)
    URL:   https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
    Data:  Hourly temperature, wind, pressure, precipitation from airports
    Steps:
        1. Use ISD Lite format for easy parsing
        2. Filter by country code = IN
        3. Download per-station CSV files

REQUIRED CSV FORMAT (for load_real_dataset)
───────────────────────────────────────────
Your CSV must have these exact column names:

    rainfall      float   mm per 3 hours       0 – 200
    humidity      float   percent               0 – 100
    wind_speed    float   metres per second     0 – 60
    temperature   float   degrees Celsius      -5 – 48
    pressure      float   hPa                940 – 1040
    flood_risk    str     LOW / MEDIUM / HIGH
    cyclone_risk  str     YES / NO
    overall_risk  str     LOW / MEDIUM / HIGH

If your CSV has raw weather but no risk labels, use label_raw_dataset().
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from config import (
    FEATURE_COLUMNS,
    FLOOD_THRESHOLDS,
    CYCLONE_WIND_THRESHOLD,
    CYCLONE_PRESSURE_THRESHOLD,
    DEFAULT_SYNTHETIC_SAMPLES,
    SYNTHETIC_SEED,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL FUNCTIONS  (encode IMD domain knowledge)
# ─────────────────────────────────────────────────────────────────────────────

def flood_label(rainfall: float, humidity: float) -> str:
    """
    Assign flood risk label based on IMD heavy-rainfall classification.

    IMD Daily Rainfall Thresholds (converted to ~3h equivalent):
        Extremely Heavy ≥ 204 mm/day  →  ~25 mm/3h  → HIGH
        Very Heavy      ≥ 115 mm/day  →  ~14 mm/3h
        Heavy           ≥  64 mm/day  →  ~8 mm/3h   → MEDIUM boundary

    Humidity amplifies risk (sustained moisture = longer flood duration).
    """
    high   = FLOOD_THRESHOLDS["HIGH"]
    medium = FLOOD_THRESHOLDS["MEDIUM"]

    if rainfall >= high["rainfall"] or (rainfall >= 15 and humidity >= high["humidity"]):
        return "HIGH"
    elif rainfall >= medium["rainfall"] or (rainfall >= 3 and humidity >= medium["humidity"]):
        return "MEDIUM"
    return "LOW"


def cyclone_label(wind_speed: float, pressure: float) -> str:
    """
    Assign cyclone risk label based on IMD wind speed + pressure criteria.

    IMD Cyclone Classification:
        Depression        < 17 m/s
        Cyclonic Storm   17–24 m/s   ← threshold used here
        Severe Cyclone   24–32 m/s
        Very Severe      32–46 m/s

    Low pressure (< 990 hPa) combined with high wind confirms cyclonic system.
    """
    if wind_speed >= CYCLONE_WIND_THRESHOLD and pressure < CYCLONE_PRESSURE_THRESHOLD:
        return "YES"
    return "NO"


def overall_label(flood: str, cyclone: str, wind: float, rainfall: float) -> str:
    """
    Aggregate risk score from individual hazard labels.

    Scoring:
        Flood HIGH   → +3
        Cyclone YES  → +3
        Flood MEDIUM → +1
        Rainfall >40 → +1
        Wind > 20    → +1

    Total ≥ 4 → HIGH | ≥ 2 → MEDIUM | else → LOW
    """
    score = 0
    if flood == "HIGH":     score += 3
    elif flood == "MEDIUM": score += 1
    if cyclone == "YES":    score += 3
    if rainfall > 40:       score += 1
    if wind > 20:           score += 1

    if   score >= 4: return "HIGH"
    elif score >= 2: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_data(
    n_samples: int = DEFAULT_SYNTHETIC_SAMPLES,
    seed:      int = SYNTHETIC_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic weather + risk-label dataset for training.

    Three seasonal regimes are mixed to replicate Indian climate:

        Monsoon (40%)    — June–September
            Heavy rainfall, very high humidity, moderate-high wind,
            lower pressure, moderate temperature.

        Transition (30%) — March–May and October–November
            Moderate rain, rising humidity, elevated wind (cyclone risk
            on eastern coast), variable pressure.

        Winter (30%)     — December–February
            Low rainfall, low humidity, gentle wind, stable high pressure,
            cooler temperature.

    Args:
        n_samples: Number of training rows to generate.
        seed:      Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            rainfall, humidity, wind_speed, temperature, pressure,
            flood_risk, cyclone_risk, overall_risk, season
    """
    rng    = np.random.default_rng(seed)
    season = rng.choice(
        ["monsoon", "transition", "winter"],
        size=n_samples,
        p=[0.40, 0.30, 0.30],
    )

    rainfall    = np.zeros(n_samples)
    humidity    = np.zeros(n_samples)
    wind_speed  = np.zeros(n_samples)
    temperature = np.zeros(n_samples)
    pressure    = np.zeros(n_samples)

    mon  = season == "monsoon"
    tra  = season == "transition"
    win  = season == "winter"

    # ── Monsoon ───────────────────────────────────────────────────────────────
    n_m = mon.sum()
    # Gamma distribution gives right-skewed rain (lots of light, some heavy)
    rainfall[mon]    = rng.gamma(shape=2.8, scale=11.0, size=n_m)
    humidity[mon]    = rng.normal(loc=83,   scale=7.0,  size=n_m)
    wind_speed[mon]  = rng.gamma(shape=2.2, scale=3.8,  size=n_m)
    temperature[mon] = rng.normal(loc=29,   scale=2.5,  size=n_m)
    pressure[mon]    = rng.normal(loc=1001, scale=7.0,  size=n_m)

    # ── Transition (cyclone risk) ──────────────────────────────────────────────
    n_t = tra.sum()
    rainfall[tra]    = rng.gamma(shape=1.5, scale=7.0,  size=n_t)
    humidity[tra]    = rng.normal(loc=68,   scale=13.0, size=n_t)
    wind_speed[tra]  = rng.gamma(shape=2.8, scale=5.5,  size=n_t)   # higher for cyclones
    temperature[tra] = rng.normal(loc=33,   scale=4.0,  size=n_t)
    pressure[tra]    = rng.normal(loc=1004, scale=13.0, size=n_t)   # wider range — deep lows

    # ── Winter ────────────────────────────────────────────────────────────────
    n_w = win.sum()
    rainfall[win]    = rng.gamma(shape=0.4, scale=1.8,  size=n_w)
    humidity[win]    = rng.normal(loc=50,   scale=14.0, size=n_w)
    wind_speed[win]  = rng.gamma(shape=1.4, scale=2.2,  size=n_w)
    temperature[win] = rng.normal(loc=20,   scale=5.0,  size=n_w)
    pressure[win]    = rng.normal(loc=1016, scale=4.5,  size=n_w)

    # ── Clip to physical bounds ───────────────────────────────────────────────
    rainfall    = np.clip(rainfall,      0,  200)
    humidity    = np.clip(humidity,      0,  100)
    wind_speed  = np.clip(wind_speed,    0,   60)
    temperature = np.clip(temperature,  -5,   48)
    pressure    = np.clip(pressure,    940, 1040)

    # ── Generate labels ───────────────────────────────────────────────────────
    f_risk = [flood_label(r, h)   for r, h in zip(rainfall, humidity)]
    c_risk = [cyclone_label(w, p) for w, p in zip(wind_speed, pressure)]
    o_risk = [
        overall_label(f, c, w, r)
        for f, c, w, r in zip(f_risk, c_risk, wind_speed, rainfall)
    ]

    df = pd.DataFrame({
        "rainfall":    np.round(rainfall,    2),
        "humidity":    np.round(humidity,    1),
        "wind_speed":  np.round(wind_speed,  2),
        "temperature": np.round(temperature, 1),
        "pressure":    np.round(pressure,    1),
        "flood_risk":   f_risk,
        "cyclone_risk": c_risk,
        "overall_risk": o_risk,
        "season":       season,
    })

    logger.info(
        f"[DataGen] Generated {n_samples:,} samples. "
        f"Flood: {pd.Series(f_risk).value_counts().to_dict()} | "
        f"Cyclone YES: {sum(1 for c in c_risk if c=='YES')}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# REAL DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_real_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a real CSV dataset and validate it has the required columns.

    Args:
        csv_path: Path to your CSV file.

    Returns:
        Validated DataFrame with FEATURE_COLUMNS + label columns.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ValueError:        If required columns are missing.
    """
    import os
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"[DataGen] Loaded real dataset: {csv_path} → {df.shape}")

    required = FEATURE_COLUMNS + ["flood_risk", "cyclone_risk", "overall_risk"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            "Required: rainfall, humidity, wind_speed, temperature, pressure,\n"
            "          flood_risk (LOW/MEDIUM/HIGH),\n"
            "          cyclone_risk (YES/NO),\n"
            "          overall_risk (LOW/MEDIUM/HIGH)"
        )

    # Drop rows with nulls in critical columns
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        logger.warning(f"[DataGen] Dropped {before - len(df)} rows with nulls.")

    return df[required].reset_index(drop=True)


def label_raw_dataset(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Take a raw weather CSV (no risk labels) and auto-generate labels
    using IMD thresholds. Useful for IMD / NOAA / Kaggle raw CSVs.

    Args:
        csv_path:    Path to CSV with columns: rainfall, humidity,
                     wind_speed, temperature, pressure
        output_path: If given, save the labeled CSV here.

    Returns:
        DataFrame with added flood_risk, cyclone_risk, overall_risk columns.
    """
    df = pd.read_csv(csv_path)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Raw dataset missing column: '{col}'")

    df["flood_risk"]   = df.apply(lambda r: flood_label(r["rainfall"], r["humidity"]),   axis=1)
    df["cyclone_risk"] = df.apply(lambda r: cyclone_label(r["wind_speed"], r["pressure"]), axis=1)
    df["overall_risk"] = df.apply(
        lambda r: overall_label(r["flood_risk"], r["cyclone_risk"],
                                r["wind_speed"], r["rainfall"]), axis=1
    )

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"[DataGen] Labeled dataset saved to: {output_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK DEMO
#  Run:  python data_generator.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = generate_training_data(n_samples=2000)
    print(df.head(10).to_string())

    print("\n── Flood Risk Distribution ──")
    print(df["flood_risk"].value_counts())

    print("\n── Cyclone Risk Distribution ──")
    print(df["cyclone_risk"].value_counts())

    print("\n── Overall Risk Distribution ──")
    print(df["overall_risk"].value_counts())

    print("\n── Season Distribution ──")
    print(df["season"].value_counts())