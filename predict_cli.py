"""
predict_cli.py ─ Command-Line Prediction Tool
=============================================
Predict disaster risk directly from the terminal — useful for
testing, batch scripts, or when you don't want to open the Streamlit UI.

Usage
─────
  # Live weather via PyOWM:
  python predict_cli.py --city Mumbai --api-key YOUR_KEY

  # Manual weather values:
  python predict_cli.py --manual \\
      --rainfall 28 --humidity 90 \\
      --wind 18 --temp 29 --pressure 995

  # With saved model (skip retraining):
  python predict_cli.py --city Chennai --api-key KEY --load-model

  # Forecast mode:
  python predict_cli.py --city Kolkata --api-key KEY --forecast
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from data_generator import generate_training_data
from model import DisasterRiskModel
from predictor import DisasterPredictor
from weather_api import WeatherAPIClient
from utils import risk_emoji, risk_color, build_advisories


# ANSI colour codes for terminal output
class C:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def colour_risk(level: str) -> str:
    """Wrap a risk label in ANSI colour."""
    colours = {
        "HIGH":   C.RED,
        "MEDIUM": C.YELLOW,
        "LOW":    C.GREEN,
        "YES":    C.RED,
        "NO":     C.GREEN,
    }
    c = colours.get(level, "")
    return f"{C.BOLD}{c}{level}{C.RESET}"


def print_result(result: dict, city: str = "") -> None:
    """Print a formatted prediction result to the terminal."""
    w = result.get("weather", {})
    print(f"\n{C.BOLD}{C.CYAN}{'═'*56}{C.RESET}")
    print(f"{C.BOLD}  🌪️  DISASTER RISK PREDICTION{C.RESET}")
    if city:
        print(f"  📍 City : {city}")
    print(f"{C.BOLD}{C.CYAN}{'─'*56}{C.RESET}")
    print(f"  🌧️  Rainfall    : {w.get('rainfall',    0):.1f} mm")
    print(f"  💧  Humidity    : {w.get('humidity',    0):.0f} %")
    print(f"  💨  Wind Speed  : {w.get('wind_speed',  0):.1f} m/s")
    print(f"  🌡️  Temperature : {w.get('temperature', 0):.1f} °C")
    print(f"  📊  Pressure    : {w.get('pressure',    0):.0f} hPa")
    print(f"{C.BOLD}{C.CYAN}{'─'*56}{C.RESET}")
    print(f"  {risk_emoji(result['flood_risk'])}  Flood Risk    : "
          f"{colour_risk(result['flood_risk']):<20} "
          f"Confidence: {result['flood_prob']*100:.1f}%")
    print(f"  {risk_emoji(result['cyclone_risk'])}  Cyclone Risk  : "
          f"{colour_risk(result['cyclone_risk']):<20} "
          f"Confidence: {result['cyclone_prob']*100:.1f}%")
    print(f"  {risk_emoji(result['overall_risk'])}  Overall Risk  : "
          f"{colour_risk(result['overall_risk']):<20} "
          f"Confidence: {result['overall_prob']*100:.1f}%")
    print(f"{C.BOLD}{C.CYAN}{'─'*56}{C.RESET}")

    # Advisories
    advisories = build_advisories(result)
    for adv in advisories:
        print(f"\n  {adv['message']}")
    print(f"\n{C.BOLD}{C.CYAN}{'═'*56}{C.RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Disaster Risk CLI Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--city",       type=str,   help="Indian city name (live PyOWM weather)")
    parser.add_argument("--api-key",    type=str,   default="", help="OpenWeatherMap API key")
    parser.add_argument("--manual",     action="store_true", help="Enter weather values manually")
    parser.add_argument("--rainfall",   type=float, default=0.0,    help="Rainfall mm/3h")
    parser.add_argument("--humidity",   type=float, default=60.0,   help="Humidity %%")
    parser.add_argument("--wind",       type=float, default=5.0,    help="Wind speed m/s")
    parser.add_argument("--temp",       type=float, default=28.0,   help="Temperature °C")
    parser.add_argument("--pressure",   type=float, default=1013.0, help="Pressure hPa")
    parser.add_argument("--load-model", action="store_true", help="Load saved model instead of retraining")
    parser.add_argument("--model-path", type=str,   default=None,   help="Path to saved model .pkl")
    parser.add_argument("--forecast",   action="store_true", help="Also print 5-day forecast predictions")
    args = parser.parse_args()

    # ── Load or Train Model ───────────────────────────────────────────────────
    model = DisasterRiskModel()

    if args.load_model or args.model_path:
        try:
            model.load(args.model_path)
        except FileNotFoundError:
            print("[CLI] Saved model not found — training fresh model...")
            df = generate_training_data()
            model.train(df)
    else:
        print("[CLI] Training model on synthetic data...")
        df = generate_training_data()
        model.train(df)

    predictor = DisasterPredictor(model)

    # ── Get Weather Data ──────────────────────────────────────────────────────
    if args.city and args.api_key:
        try:
            client  = WeatherAPIClient(args.api_key)
        except ValueError as e:
            print(f"[CLI] ❌ {e}")
            sys.exit(1)

        print(f"\n[CLI] Fetching live weather for {args.city} via PyOWM...")
        weather = client.get_weather(args.city)
        if weather is None:
            print("[CLI] ❌ Could not fetch weather. "
                  "Check city name and API key.")
            sys.exit(1)

        result = predictor.predict(weather)
        print_result(result, city=args.city)

        # Forecast
        if args.forecast:
            print(f"[CLI] Fetching 5-day forecast for {args.city}...")
            fc_list = client.get_forecast(args.city)
            if fc_list:
                fc_results = predictor.predict_forecast(fc_list)
                print(f"\n{'─'*70}")
                print(f"  5-DAY FORECAST — {args.city}")
                print(f"{'─'*70}")
                print(f"  {'Date/Time':<22} {'Flood':<10} {'Cyclone':<10} {'Overall':<10}")
                print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
                for fr in fc_results:
                    dt = str(fr.get("datetime", ""))[:16]
                    print(
                        f"  {dt:<22} "
                        f"{colour_risk(fr['flood_risk']):<10} "
                        f"{colour_risk(fr['cyclone_risk']):<10} "
                        f"{colour_risk(fr['overall_risk']):<10}"
                    )
                print()

    elif args.manual:
        weather = {
            "rainfall":    args.rainfall,
            "humidity":    args.humidity,
            "wind_speed":  args.wind,
            "temperature": args.temp,
            "pressure":    args.pressure,
        }
        result = predictor.predict(weather)
        print_result(result)

    else:
        print("[CLI] ⚠️  Provide  --city + --api-key  OR use  --manual  mode.\n")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()