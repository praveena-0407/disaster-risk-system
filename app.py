"""
app.py ─ Streamlit UI for Real-Time Disaster Risk Prediction System
====================================================================
Entry point for the web application.

Run:
    streamlit run app.py

Features:
  • Live weather via PyOWM for any Indian city
  • Three Random Forest risk predictions with confidence %
  • Interactive gauge charts and feature importance bar chart
  • 5-day/3-hour forecast risk table + trend line chart
  • Safety advisories per risk level
  • Training dataset explorer with distribution chart
  • Model accuracy metrics banner
  • Dark theme with colour-coded risk cards
  • 🖼️ Image Disaster Detector (CNN-powered)

FIXES APPLIED:
  1. %{text}% bug fixed → texttemplate now shows actual % values
  2. API key hidden from UI → loaded only from config.py (not shown in sidebar)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import numpy as np

from config import OWM_API_KEY, INDIA_CITIES
from weather_api import WeatherAPIClient
from data_generator import generate_training_data
from model import DisasterRiskModel
from predictor import DisasterPredictor
from utils import (
    risk_color, risk_emoji, style_risk_cell,
    format_forecast_table, build_advisories,
)
from chatbot import chatbot
from image_classifier import classify_image, load_classifier

logging.basicConfig(level=logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌪️ Disaster Risk Prediction",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background: linear-gradient(160deg, #080d18 0%, #0c1624 60%, #0a1220 100%);
}

/* ── Sidebar ───────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Risk Cards ────────────────────────── */
.risk-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 28px 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 8px;
}
.risk-card:hover {
    transform: translateY(-5px);
}
.risk-HIGH   { border-color:#ff4444 !important; box-shadow: 0 0 28px rgba(255,68,68,0.25); }
.risk-MEDIUM { border-color:#ffaa00 !important; box-shadow: 0 0 28px rgba(255,170,0,0.25); }
.risk-LOW    { border-color:#00cc88 !important; box-shadow: 0 0 28px rgba(0,204,136,0.20); }
.risk-YES    { border-color:#ff4444 !important; box-shadow: 0 0 28px rgba(255,68,68,0.25); }
.risk-NO     { border-color:#00cc88 !important; box-shadow: 0 0 28px rgba(0,204,136,0.20); }

/* ── Weather Metric Chips ──────────────── */
.metric-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size:11px; color:#7a8fa8; text-transform:uppercase; letter-spacing:1.2px; }
.metric-value { font-size:28px; font-weight:700; color:#e2ecff;
                margin-top:4px; font-family:'Syne',sans-serif; }
.metric-unit  { font-size:12px; color:#5a7090; margin-top:2px; }

/* ── Section Titles ────────────────────── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #d8e8ff;
    margin: 36px 0 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(80,140,255,0.25);
}

/* ── Alert Banner ──────────────────────── */
.alert-banner {
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 16px;
    font-weight: 500;
    font-size: 15px;
    line-height: 1.5;
}
.alert-HIGH   { background:rgba(255,68,68,0.12);  border:1px solid rgba(255,68,68,0.4); }
.alert-MEDIUM { background:rgba(255,170,0,0.10);  border:1px solid rgba(255,170,0,0.4); }
.alert-LOW    { background:rgba(0,204,136,0.08);  border:1px solid rgba(0,204,136,0.3); }

/* ── Image Detector Card ───────────────── */
.detector-result {
    background: rgba(255,255,255,0.04);
    border: 2px solid rgba(80,140,255,0.35);
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    backdrop-filter: blur(12px);
    margin-top: 16px;
}
.detector-result.result-disaster {
    border-color: rgba(255,68,68,0.6) !important;
    box-shadow: 0 0 36px rgba(255,68,68,0.20);
}
.detector-result.result-safe {
    border-color: rgba(0,204,136,0.6) !important;
    box-shadow: 0 0 36px rgba(0,204,136,0.20);
}

/* ── Streamlit overrides ───────────────── */
.stDataFrame { background: rgba(255,255,255,0.03) !important; border-radius:12px; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("model",        None),
    ("predictor",    None),
    ("metrics",      {}),
    ("train_df",     None),
    ("cnn_model",    None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADER  (cached so training only runs once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🏋️ Training Random Forest models…")
def load_and_train_model(n_samples: int = 8000):
    """Train all three models on synthetic data and cache the result."""
    df      = generate_training_data(n_samples=n_samples)
    model   = DisasterRiskModel()
    metrics = model.train(df)
    pred    = DisasterPredictor(model)
    return model, pred, metrics, df


@st.cache_resource(show_spinner="🧠 Loading CNN disaster image model…")
def load_cnn_model():
    """Load the CNN model once and cache it."""
    return load_classifier()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — API key is loaded silently from config.py (NOT shown in UI)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;font-size:22px;"
        "color:#d8e8ff;margin-bottom:4px'>🌪️ Disaster Risk AI</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Powered by PyOWM + Random Forest ML")
    st.markdown("---")

    # City selector
    city = st.selectbox(
        "📍 Select City",
        INDIA_CITIES,
        index=INDIA_CITIES.index("Mumbai") if "Mumbai" in INDIA_CITIES else 0,
        help="All 50+ major Indian cities supported",
    )

    # Date & time (cosmetic — forecast uses OWM's own timestamps)
    min_dt = datetime.now().date()
    max_dt = min_dt + timedelta(days=4)
    st.date_input("📅 Date", min_value=min_dt, max_value=max_dt, value=min_dt)
    st.time_input("⏰ Time", value=datetime.now().replace(second=0, microsecond=0).time())

    st.markdown("---")

    # Action buttons
    predict_btn = st.button("🔍  Predict Risk Now",  type="primary", use_container_width=True)
    retrain_btn = st.button("🏋️  Retrain Model",     use_container_width=True)

    st.markdown("---")

    # Training data size slider
    n_samples = st.slider(
        "Training samples (synthetic)",
        min_value=2000, max_value=20000, value=8000, step=1000,
        help="More samples → more accurate model, slower training",
    )

    st.markdown("---")
    st.markdown("**📡 Data Sources**")
    st.markdown("---")
    if st.button("🤖 AI Chatbot", use_container_width=True):
        st.session_state.show_chatbot = not st.session_state.get("show_chatbot", False)
    st.caption("• Live weather: [OpenWeatherMap](https://openweathermap.org)")
    st.caption("• ML library: [scikit-learn](https://scikit-learn.org)")
    st.caption("• Risk thresholds: [IMD](https://mausam.imd.gov.in)")
    st.caption("• Emergency: NDMA Helpline **1078**")


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-size:40px;font-weight:800;
color:#e0eeff;margin-bottom:4px;line-height:1.15'>
🌪️ Real-Time Disaster Risk Prediction
</h1>
<p style='color:#7a9abf;font-size:16px;margin-bottom:32px'>
AI-powered flood & cyclone risk assessment for Indian cities
using live weather via <b>PyOWM</b> + <b>Random Forest</b> ML.
</p>
""", unsafe_allow_html=True)
if not predict_btn:

    st.markdown("""
    <div class='section-title'>🚀 Why This App Matters</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="risk-card">
            <div style="font-size:36px">🌦️</div>
            <h4>Live Weather Intelligence</h4>
            <p style="color:#9aaabb;font-size:14px">
            Fetches real-time weather data using PyOWM API.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="risk-card">
            <div style="font-size:36px">🤖</div>
            <h4>AI Risk Prediction</h4>
            <p style="color:#9aaabb;font-size:14px">
            Uses Random Forest ML for disaster prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="risk-card">
            <div style="font-size:36px">📊</div>
            <h4>Smart Insights</h4>
            <p style="color:#9aaabb;font-size:14px">
            Shows probabilities and risk levels visually.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>🌍 Real-World Impact</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-banner alert-MEDIUM">
    ✔ Early warnings before disasters<br><br>
    ✔ Helps decision making<br><br>
    ✔ Improves safety awareness<br><br>
    ✔ Uses real-time + ML data
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# AUTO-TRAIN ON FIRST LOAD
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.model is None or retrain_btn:
    m, p, mx, df = load_and_train_model(n_samples)
    st.session_state.model     = m
    st.session_state.predictor = p
    st.session_state.metrics   = mx
    st.session_state.train_df  = df
    if retrain_btn:
        st.success("✅ Model retrained successfully!")
        load_and_train_model.clear()   # clear cache so next call retrains


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON HANDLER
# ──────────────────────────────────────────────────────────────────────────────
if predict_btn:

    api_key = OWM_API_KEY.strip()

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        st.error(
            "⚠️ API key not found in config.py\n\n"
            "Open **config.py** and set:\n"
            "```\nOWM_API_KEY = os.environ.get('OWM_API_KEY', 'your_actual_key_here')\n```"
        )
        st.stop()

    try:
        client = WeatherAPIClient(api_key)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner(f"🌐 Fetching live weather for **{city}** via PyOWM…"):
        weather_data = client.get_weather(city)

    if weather_data is None:
        st.error(
            f"❌ Could not fetch weather for **{city}**.\n\n"
            "Possible reasons:\n"
            "• API key not yet activated (wait up to 2 hours after signup)\n"
            "• Typo in city name\n"
            "• Network issue"
        )
        st.stop()

    with st.spinner("📅 Fetching 5-day forecast…"):
        forecast_data = client.get_forecast(city)

    predictor: DisasterPredictor = st.session_state.predictor
    result   = predictor.predict(weather_data)
    fc_preds = predictor.predict_forecast(forecast_data) if forecast_data else []

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Current Weather Conditions
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='section-title'>🌦️ Live Weather Conditions</div>",
        unsafe_allow_html=True,
    )

    desc = weather_data.get("description", "").title()
    ts   = str(weather_data.get("timestamp", ""))[:19]
    st.caption(f"📍 **{city}** · {desc} · As of {ts} UTC")

    col1, col2, col3, col4, col5 = st.columns(5)
    chips = [
        (col1, "🌧️ Rainfall",     f"{weather_data.get('rainfall',0):.1f}",    "mm / 3h"),
        (col2, "💧 Humidity",     f"{weather_data.get('humidity',0):.0f}",     "%"),
        (col3, "💨 Wind Speed",   f"{weather_data.get('wind_speed',0):.1f}",   "m/s"),
        (col4, "🌡️ Temperature",  f"{weather_data.get('temperature',0):.1f}",  "°C"),
        (col5, "📊 Pressure",     f"{weather_data.get('pressure',0):.0f}",     "hPa"),
    ]
    for col, label, val, unit in chips:
        with col:
            st.markdown(f"""
            <div class="metric-chip">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Risk Cards
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='section-title'>⚠️ Predicted Risk Levels</div>",
        unsafe_allow_html=True,
    )

    rc1, rc2, rc3 = st.columns(3)
    cards = [
        (rc1, "🌊 Flood Risk",   result["flood_risk"],   result["flood_prob"]),
        (rc2, "🌀 Cyclone Risk", result["cyclone_risk"],  result["cyclone_prob"]),
        (rc3, "⚡ Overall Risk", result["overall_risk"],  result["overall_prob"]),
    ]
    for col, title, level, prob in cards:
        with col:
            emoji  = risk_emoji(level)
            color  = risk_color(level)
            st.markdown(f"""
            <div class="risk-card risk-{level}">
                <div style="font-size:44px;margin-bottom:8px">{emoji}</div>
                <div style="font-family:DM Sans,sans-serif;font-size:13px;
                            color:#8899bb;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:6px">{title}</div>
                <div style="font-family:Syne,sans-serif;font-size:34px;
                            font-weight:800;color:{color}">{level}</div>
                <div style="font-size:14px;color:#9aaabb;margin-top:10px">
                    Confidence: <b style="color:{color}">{prob*100:.1f}%</b>
                </div>
            </div>""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Probability Gauges
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='section-title'>📊 Risk Confidence Gauges</div>",
        unsafe_allow_html=True,
    )

    def make_gauge(value_pct: float, title: str, color: str) -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(value_pct, 1),
            title={"text": title, "font": {"size": 14, "color": "#9aaabb",
                                           "family": "DM Sans"}},
            number={"suffix": "%", "font": {"size": 30, "color": "#e0eeff",
                                            "family": "Syne"}},
            delta={"reference": 50, "increasing": {"color": "#ff6666"},
                   "decreasing": {"color": "#44dd88"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#334455",
                          "tickfont": {"color": "#556677", "size": 10}},
                "bar":  {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "bordercolor": "rgba(255,255,255,0.08)",
                "borderwidth": 1,
                "steps": [
                    {"range": [0,  40], "color": "rgba(0,204,136,0.10)"},
                    {"range": [40, 70], "color": "rgba(255,170,0,0.10)"},
                    {"range": [70,100], "color": "rgba(255,68,68,0.10)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.85,
                    "value": round(value_pct, 1),
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin=dict(l=24, r=24, t=48, b=12),
        )
        return fig

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            make_gauge(result["flood_prob"]*100,   "Flood Risk",   risk_color(result["flood_risk"])),
            use_container_width=True, key="gauge_flood",
        )
    with g2:
        st.plotly_chart(
            make_gauge(result["cyclone_prob"]*100, "Cyclone Risk", risk_color(result["cyclone_risk"])),
            use_container_width=True, key="gauge_cyclone",
        )
    with g3:
        st.plotly_chart(
            make_gauge(result["overall_prob"]*100, "Overall Risk", risk_color(result["overall_risk"])),
            use_container_width=True, key="gauge_overall",
        )

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Class Probability Breakdown
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='section-title'>🎯 Full Probability Breakdown</div>",
        unsafe_allow_html=True,
    )

    pb1, pb2, pb3 = st.columns(3)

    def prob_bar_chart(classes, probas, title, color_map):
        df_prob = pd.DataFrame({
            "Class": classes,
            "Probability": [round(p * 100, 1) for p in probas]
        })
        fig = px.bar(
            df_prob, x="Class", y="Probability",
            color="Class", color_discrete_map=color_map,
            title=title,
            text="Probability",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont_color="#d0e0ff",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#d0e0ff", "family": "DM Sans"},
            title_font={"family": "Syne", "size": 15},
            showlegend=False, height=260,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)",
                       ticksuffix="%"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        return fig

    flood_cmap   = {"LOW": "#00cc88", "MEDIUM": "#ffaa00", "HIGH": "#ff4444"}
    cyclone_cmap = {"NO": "#00cc88", "YES": "#ff4444"}
    overall_cmap = {"LOW": "#00cc88", "MEDIUM": "#ffaa00", "HIGH": "#ff4444"}

    with pb1:
        st.plotly_chart(
            prob_bar_chart(
                result["flood_classes"], result["flood_proba_all"],
                "Flood Risk Classes", flood_cmap,
            ),
            use_container_width=True, key="pb_flood",
        )
    with pb2:
        st.plotly_chart(
            prob_bar_chart(
                result["cyclone_classes"], result["cyclone_proba_all"],
                "Cyclone Risk Classes", cyclone_cmap,
            ),
            use_container_width=True, key="pb_cyclone",
        )
    with pb3:
        st.plotly_chart(
            prob_bar_chart(
                result["overall_classes"], result["overall_proba_all"],
                "Overall Risk Classes", overall_cmap,
            ),
            use_container_width=True, key="pb_overall",
        )

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Feature Importance
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='section-title'>🧠 Feature Importance (Random Forest)</div>",
        unsafe_allow_html=True,
    )

    fi_tab1, fi_tab2, fi_tab3 = st.tabs(["🌊 Flood Model", "🌀 Cyclone Model", "⚡ Overall Model"])

    def fi_chart(model_name: str):
        fi = st.session_state.model.get_feature_importance(model_name)
        if fi is None:
            st.warning("Feature importance unavailable.")
            return
        df_fi = pd.DataFrame(fi, columns=["Feature", "Importance"])
        df_fi = df_fi.sort_values("Importance")
        fig = px.bar(
            df_fi, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#1a3a6a", "#2a7adf", "#60c0ff"],
            text_auto=".3f",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#d0e0ff", "family": "DM Sans"},
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        fig.update_traces(textfont_color="#d0e0ff")
        st.plotly_chart(fig, use_container_width=True)

    with fi_tab1: fi_chart("flood")
    with fi_tab2: fi_chart("cyclone")
    with fi_tab3: fi_chart("overall")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6 — 5-Day Forecast
    # ═════════════════════════════════════════════════════════════════════════
    if fc_preds:
        st.markdown(
            "<div class='section-title'>📅 5-Day Forecast Risk Predictions</div>",
            unsafe_allow_html=True,
        )

        max_rain  = max(r["weather"].get("rainfall", 0)   for r in fc_preds)
        max_wind  = max(r["weather"].get("wind_speed", 0) for r in fc_preds)
        high_days = sum(1 for r in fc_preds if r["overall_risk"] == "HIGH")

        fs1, fs2, fs3 = st.columns(3)
        with fs1: st.metric("🌧️ Max Forecast Rainfall", f"{max_rain:.1f} mm")
        with fs2: st.metric("💨 Max Forecast Wind",     f"{max_wind:.1f} m/s")
        with fs3: st.metric("🔴 HIGH Risk Time Steps",  f"{high_days} / {len(fc_preds)}")

        ft_df = format_forecast_table(fc_preds)
        risk_cols = ["Flood Risk", "Cyclone Risk", "Overall Risk"]
        styled = ft_df.style.applymap(style_risk_cell, subset=risk_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=320)

        st.markdown("**📈 Forecast Weather Trends**")
        times    = [r.get("datetime", "")  for r in fc_preds]
        rainfall = [r["weather"].get("rainfall",    0) for r in fc_preds]
        wind     = [r["weather"].get("wind_speed",  0) for r in fc_preds]
        humidity = [r["weather"].get("humidity",    0) for r in fc_preds]
        temp     = [r["weather"].get("temperature", 0) for r in fc_preds]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=times, y=rainfall, name="Rainfall (mm)",
            line=dict(color="#4488ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(68,136,255,0.08)",
        ))
        fig_fc.add_trace(go.Scatter(
            x=times, y=wind, name="Wind (m/s)",
            line=dict(color="#ff6644", width=2.5),
        ))
        fig_fc.add_trace(go.Scatter(
            x=times, y=humidity, name="Humidity (%)",
            line=dict(color="#44ffbb", width=2, dash="dot"),
            yaxis="y2",
        ))
        fig_fc.add_trace(go.Scatter(
            x=times, y=temp, name="Temp (°C)",
            line=dict(color="#ffdd44", width=2, dash="dash"),
            yaxis="y2",
        ))
        fig_fc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#d0e0ff", "family": "DM Sans"},
            height=360,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                        yanchor="bottom", y=1.01, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(
                title="mm / m/s",
                gridcolor="rgba(255,255,255,0.05)",
                zerolinecolor="rgba(255,255,255,0.1)",
            ),
            yaxis2=dict(
                title="% / °C",
                overlaying="y", side="right",
                showgrid=False,
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, use_container_width=True, key="fc_trend")

        st.markdown("**🎯 Forecast Risk Category Timeline**")
        risk_to_num = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "NO": 1, "YES": 3}
        fig_risk_tl = go.Figure()
        for rtype, rkey, rprob, label in [
            ("flood",   "flood_risk",   "flood_prob",   "🌊 Flood"),
            ("cyclone", "cyclone_risk", "cyclone_prob", "🌀 Cyclone"),
            ("overall", "overall_risk", "overall_prob", "⚡ Overall"),
        ]:
            y_vals  = [risk_to_num.get(r[rkey], 1) for r in fc_preds]
            probs   = [r[rprob]*100 for r in fc_preds]
            colors  = [risk_color(r[rkey]) for r in fc_preds]
            fig_risk_tl.add_trace(go.Scatter(
                x=times, y=y_vals, name=label,
                mode="lines+markers",
                marker=dict(size=10, color=colors, line=dict(width=1, color="#111")),
                line=dict(width=2),
                text=[f"{r[rkey]} ({p:.1f}%)" for r, p in zip(fc_preds, probs)],
                hovertemplate="%{text}<extra></extra>",
            ))
        fig_risk_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#d0e0ff", "family": "DM Sans"},
            height=280,
            yaxis=dict(
                tickvals=[1, 2, 3],
                ticktext=["LOW / NO", "MEDIUM", "HIGH / YES"],
                gridcolor="rgba(255,255,255,0.05)",
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_risk_tl, use_container_width=True, key="fc_risk_tl")

    else:
        st.info("Forecast data unavailable — showing only current weather prediction.")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Safety Advisories
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='section-title'>📋 Safety Advisories</div>",
        unsafe_allow_html=True,
    )

    advisories = build_advisories(result)
    for adv in advisories:
        if adv["type"] == "error":
            st.error(adv["message"])
        elif adv["type"] == "warning":
            st.warning(adv["message"])
        elif adv["type"] == "success":
            st.success(adv["message"])
        else:
            st.info(adv["message"])

    with st.expander("📞 Emergency Contacts — India"):
        st.markdown("""
        | Agency | Contact |
        |--------|---------|
        | National Disaster Helpline (NDMA) | **1078** |
        | NDRF (National Disaster Response Force) | **011-24363260** |
        | IMD Weather Alerts | [mausam.imd.gov.in](https://mausam.imd.gov.in) |
        | NDMA Guidelines | [ndma.gov.in](https://ndma.gov.in) |
        | Coastal Cyclone Warnings | [incois.gov.in](https://incois.gov.in) |
        | Police Emergency | **100** |
        | Ambulance | **108** |
        """)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — 🖼️ IMAGE DISASTER DETECTOR
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-title'>🖼️ Image Disaster Detector</div>",
    unsafe_allow_html=True,
)
st.caption("Upload any disaster image — our CNN model will identify what type of disaster it is.")

# Load CNN model (cached)
cnn_ready = True
try:
    if st.session_state.cnn_model is None:
        st.session_state.cnn_model = load_cnn_model()
except Exception as e:
    cnn_ready = False
    st.warning(
        f"⚠️ CNN model not available: {e}\n\n"
        "Make sure `cnn_model/disaster_cnn.h5` exists. Run `python train_cnn.py` first."
    )

if cnn_ready:
    uploaded_file = st.file_uploader(
        "Upload a disaster image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported: JPG, PNG, BMP, WEBP",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # ── Layout: image left, result right ─────────────────────────────────
        img_col, res_col = st.columns([1, 1], gap="large")

        with img_col:
            st.markdown("**📷 Uploaded Image**")
            st.image(uploaded_file, use_column_width=True)

        with res_col:
            with st.spinner("🔍 Analysing image with CNN…"):
                # Save to temp file for PIL to read
                import tempfile, os
                suffix = os.path.splitext(uploaded_file.name)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result_img = classify_image(tmp_path, st.session_state.cnn_model)
                os.unlink(tmp_path)   # clean up temp file

            label      = result_img["label"]
            display    = result_img["display_label"]
            emoji      = result_img["emoji"]
            confidence = result_img["confidence"]
            all_probs  = result_img["all_probs"]

            is_safe    = label == "Non_Damage"
            card_class = "result-safe" if is_safe else "result-disaster"
            text_color = "#00cc88" if is_safe else "#ff5555"

            st.markdown(f"""
            <div class="detector-result {card_class}">
                <div style="font-size:56px;margin-bottom:10px">{emoji}</div>
                <div style="font-family:'Syne',sans-serif;font-size:26px;
                            font-weight:800;color:{text_color};line-height:1.2">
                    THIS IS<br>{display}
                </div>
                <div style="font-size:18px;color:#aabbd0;margin-top:14px">
                    Confidence: <b style="color:{text_color};font-size:22px">{confidence:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability breakdown bar chart ───────────────────────────────────
        st.markdown("**📊 All Class Probabilities**")

        df_probs = pd.DataFrame({
            "Disaster Type": list(all_probs.keys()),
            "Confidence (%)": list(all_probs.values()),
        }).sort_values("Confidence (%)", ascending=True)

        COLOR_MAP = {
            "Non_Damage":             "#00cc88",
            "Water_Disaster":         "#4488ff",
            "Fire_Disaster":          "#ff6644",
            "Damaged_Infrastructure": "#ffaa00",
            "Land_Disaster":          "#cc88ff",
            "Human_Damage":           "#ff4488",
        }
        bar_colors = [COLOR_MAP.get(c, "#7aaabb") for c in df_probs["Disaster Type"]]

        fig_probs = go.Figure(go.Bar(
            x=df_probs["Confidence (%)"],
            y=df_probs["Disaster Type"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in df_probs["Confidence (%)"]],
            textposition="outside",
            textfont=dict(color="#d0e0ff", size=13),
        ))
        fig_probs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#d0e0ff", "family": "DM Sans"},
            height=300,
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(
                range=[0, 115],
                gridcolor="rgba(255,255,255,0.05)",
                ticksuffix="%",
            ),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_probs, use_container_width=True, key="img_probs")

        # ── Advisory based on result ──────────────────────────────────────────
        ADVISORIES = {
            "Water_Disaster":         ("🌊", "FLOOD DETECTED", "Move to higher ground immediately. Avoid waterlogged roads. Contact NDMA: 1078", "error"),
            "Fire_Disaster":          ("🔥", "FIRE DETECTED",  "Evacuate the area. Call Fire Emergency: 101. Do not use lifts.", "error"),
            "Land_Disaster":          ("⛰️", "LAND DISASTER",  "Stay away from slopes and unstable ground. Evacuate if instructed.", "warning"),
            "Damaged_Infrastructure": ("🏚️", "STRUCTURAL DAMAGE", "Do not enter damaged buildings. Report to local authorities.", "warning"),
            "Human_Damage":           ("🚨", "HUMAN EMERGENCY", "Call Ambulance: 108 immediately. Do not move injured persons unless necessary.", "error"),
            "Non_Damage":             ("✅", "NO DISASTER",    "No disaster detected in this image. Stay alert and monitor official updates.", "success"),
        }

        adv = ADVISORIES.get(label)
        if adv:
            adv_emoji, adv_title, adv_msg, adv_type = adv
            getattr(st, adv_type)(f"{adv_emoji} **{adv_title}** — {adv_msg}")

    else:
        # Placeholder when no image uploaded
        st.markdown("""
        <div style="
            border: 2px dashed rgba(80,140,255,0.30);
            border-radius: 16px;
            padding: 48px 24px;
            text-align: center;
            color: #5a7a9a;
            margin-top: 8px;
        ">
            <div style="font-size:48px;margin-bottom:12px">🖼️</div>
            <div style="font-size:16px;font-family:'Syne',sans-serif;color:#8aaabb">
                Upload an image above to detect the disaster type
            </div>
            <div style="font-size:13px;margin-top:8px;color:#4a6a7a">
                Supports: Flood · Fire · Landslide · Infrastructure Damage · Human Damage
            </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — AI Weather Chatbot (toggled from sidebar)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("show_chatbot", False):
    st.markdown(
        "<div class='section-title'>🤖 AI Weather Safety Chatbot</div>",
        unsafe_allow_html=True,
    )
    st.caption("Ask me anything about weather safety, disasters, or risk tips!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a weather safety question...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = chatbot.ask(user_input)
            st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})