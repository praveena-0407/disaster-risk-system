"""
utils.py ─ Display Helpers
===========================
Color coding, emoji mapping, advisory messages, and
forecast DataFrame formatting for the Streamlit UI.
"""

import pandas as pd
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR & EMOJI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "HIGH":   "#ff4444",
    "MEDIUM": "#ffaa00",
    "LOW":    "#00cc88",
    "YES":    "#ff4444",
    "NO":     "#00cc88",
}

RISK_EMOJIS = {
    "HIGH":   "🔴",
    "MEDIUM": "🟡",
    "LOW":    "🟢",
    "YES":    "🌀",
    "NO":     "✅",
}


def risk_color(level: str) -> str:
    """Return hex color string for a risk level label."""
    return RISK_COLORS.get(level, "#aabbcc")


def risk_emoji(level: str) -> str:
    """Return emoji for a risk level label."""
    return RISK_EMOJIS.get(level, "❓")


def style_risk_cell(val: str) -> str:
    """CSS style for a DataFrame cell — used with df.style.applymap()."""
    color = RISK_COLORS.get(val, "#aabbcc")
    return f"color: {color}; font-weight: 600;"


# ─────────────────────────────────────────────────────────────────────────────
# ADVISORY MESSAGES
# ─────────────────────────────────────────────────────────────────────────────

def build_advisories(result: Dict) -> List[Dict]:
    """
    Generate safety advisory messages based on prediction result.

    Returns:
        List of dicts with keys: type ('error'|'warning'|'info'|'success'),
                                 message (str)
    """
    advisories = []

    flood   = result.get("flood_risk",   "LOW")
    cyclone = result.get("cyclone_risk", "NO")
    overall = result.get("overall_risk", "LOW")

    if flood == "HIGH":
        advisories.append({
            "type": "error",
            "message": (
                "🚨 **FLOOD ALERT — HIGH RISK** | "
                "Evacuate low-lying areas immediately. Avoid flooded roads and "
                "river banks. Contact NDRF: 011-24363260."
            ),
        })
    elif flood == "MEDIUM":
        advisories.append({
            "type": "warning",
            "message": (
                "⚠️ **FLOOD WATCH — MEDIUM RISK** | "
                "Monitor local water levels. Prepare emergency supplies and "
                "an evacuation plan. Stay tuned to IMD alerts."
            ),
        })

    if cyclone == "YES":
        advisories.append({
            "type": "error",
            "message": (
                "🌀 **CYCLONE WARNING** | "
                "Secure loose objects and structures. Stay indoors away from "
                "windows. Follow NDMA guidelines. Emergency: 1078."
            ),
        })

    if overall == "HIGH":
        advisories.append({
            "type": "error",
            "message": (
                "🆘 **HIGH OVERALL DISASTER RISK** | "
                "Contact local emergency services immediately. "
                "National Disaster Helpline: 1078. NDMA: ndma.gov.in"
            ),
        })
    elif overall == "MEDIUM":
        advisories.append({
            "type": "warning",
            "message": (
                "⚠️ **MODERATE RISK** | "
                "Stay alert. Keep emergency contacts handy. "
                "Follow IMD updates at mausam.imd.gov.in"
            ),
        })

    if not advisories:
        advisories.append({
            "type": "success",
            "message": (
                "✅ **CONDITIONS CLEAR** | "
                "Current weather indicates low disaster risk. "
                "Continue monitoring IMD forecasts regularly."
            ),
        })

    return advisories


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST TABLE
# ─────────────────────────────────────────────────────────────────────────────

def format_forecast_table(forecast_results: List[Dict]) -> pd.DataFrame:
    """
    Convert forecast prediction results into a display-ready DataFrame.

    Args:
        forecast_results: Output of DisasterPredictor.predict_forecast()

    Returns:
        Styled DataFrame with one row per 3-hour forecast slot.
    """
    rows = []
    for item in forecast_results:
        w = item.get("weather", {})
        rows.append({
            "Date / Time":      str(item.get("datetime", ""))[:16],
            "Rainfall (mm)":    f"{w.get('rainfall',    0):.1f}",
            "Humidity (%)":     f"{w.get('humidity',    0):.0f}",
            "Wind (m/s)":       f"{w.get('wind_speed',  0):.1f}",
            "Temp (°C)":        f"{w.get('temperature', 0):.1f}",
            "Pressure (hPa)":   f"{w.get('pressure',    0):.0f}",
            "Condition":        w.get("description", ""),
            "Flood Risk":       item.get("flood_risk",   ""),
            "Cyclone Risk":     item.get("cyclone_risk", ""),
            "Overall Risk":     item.get("overall_risk", ""),
        })
    return pd.DataFrame(rows)