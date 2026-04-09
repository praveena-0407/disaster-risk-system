"""
weather_api.py ─ Real-Time Weather Fetching with PyOWM
=======================================================
PyOWM is the official Python library for OpenWeatherMap.
It handles all HTTP calls, authentication, and response parsing
so we only deal with clean Python objects.

Install:   pip install pyowm
Docs:      https://pyowm.readthedocs.io/en/latest/
OWM API:   https://openweathermap.org/api

What this module does
─────────────────────
  WeatherAPIClient.get_weather(city)
      → Calls OWM /data/2.5/weather  (live, right now)
      → Returns dict: rainfall, humidity, wind_speed, temperature, pressure

  WeatherAPIClient.get_forecast(city)
      → Calls OWM /data/2.5/forecast  (5-day, 3-hour intervals = 40 snapshots)
      → Returns list of dicts with datetime + same features

  WeatherAPIClient.test_connection()
      → Validates API key is active and network is reachable
"""

import pyowm
from pyowm.commons import exceptions as owm_exc
from typing import Optional, Dict, List
import logging

from config import OWM_API_KEY

logger = logging.getLogger(__name__)


class WeatherAPIClient:
    """
    Wraps PyOWM to deliver clean weather feature dicts.

    Usage
    ─────
        client  = WeatherAPIClient("your_owm_api_key")
        current = client.get_weather("Mumbai")
        # → {'rainfall': 4.2, 'humidity': 84, 'wind_speed': 6.1,
        #    'temperature': 29.5, 'pressure': 1003.0, ...}

        forecast = client.get_forecast("Chennai")
        # → [ {'datetime': '2025-06-15 06:00:00', 'rainfall': 0.0, ...},
        #      {'datetime': '2025-06-15 09:00:00', 'rainfall': 2.1, ...},
        #      ...40 items... ]
    """

    def __init__(self, api_key: str = OWM_API_KEY):
        """
        Initialise PyOWM with the provided API key.

        Args:
            api_key: OpenWeatherMap API key string.
                     Get a free one at openweathermap.org/api
        """
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "No OWM API key provided.\n"
                "1. Go to https://openweathermap.org/api\n"
                "2. Sign up (free)\n"
                "3. Copy your key from the API Keys tab\n"
                "4. Paste it into config.py  OWM_API_KEY = 'your_key_here'"
            )

        # ── PyOWM initialisation ─────────────────────────────────────────────
        # owm      : root OWM object tied to your account
        # mgr      : WeatherManager — the object that makes API calls
        self.owm = pyowm.OWM(api_key)
        self.mgr = self.owm.weather_manager()

        logger.info("[PyOWM] Client initialised successfully.")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC  :  CURRENT WEATHER
    # ─────────────────────────────────────────────────────────────────────────

    def get_weather(self, city: str) -> Optional[Dict]:
        """
        Fetch LIVE current weather for an Indian city.

        PyOWM call:
            mgr.weather_at_place("Mumbai,IN")
            ↓ internally sends →
            GET https://api.openweathermap.org/data/2.5/weather
                ?q=Mumbai,IN&appid=KEY&units=metric

        Args:
            city: City name string, e.g. "Mumbai", "Chennai"

        Returns:
            Dict with keys: rainfall, humidity, wind_speed,
                            temperature, pressure, city,
                            description, timestamp
            None if the request fails for any reason.
        """
        try:
            # ── Step 1 : Fetch live observation ─────────────────────────────
            # observation holds the Observation object (location + weather)
            observation = self.mgr.weather_at_place(f"{city},IN")

            # ── Step 2 : Extract the Weather object ─────────────────────────
            # weather is the Weather object with all meteorological fields
            weather = observation.weather

            # ── Step 3 : Extract each feature ────────────────────────────────

            # RAINFALL  — dict like {'3h': 5.2} or {} when no rain
            rain_dict = weather.rain                          # PyOWM attribute
            rainfall  = rain_dict.get("3h", 0.0)             # mm in last 3 h
            if rainfall == 0.0:
                rainfall = rain_dict.get("1h", 0.0)          # fallback to 1 h

            # HUMIDITY  — integer 0‥100 %
            humidity = weather.humidity

            # WIND SPEED — specify unit explicitly for clarity
            # PyOWM returns a dict: {'speed': 6.2, 'deg': 210, 'gust': 9.1}
            wind_dict  = weather.wind(unit="meters_sec")
            wind_speed = wind_dict.get("speed", 0.0)         # m/s

            # TEMPERATURE — specify celsius
            # PyOWM returns: {'temp': 29.4, 'feels_like': 33.1,
            #                 'temp_min': 28.0, 'temp_max': 31.0}
            temp_dict   = weather.temperature(unit="celsius")
            temperature = temp_dict.get("temp", 25.0)

            # PRESSURE — dict: {'press': 1003, 'sea_level': 1003}
            pressure_dict = weather.pressure
            pressure      = pressure_dict.get("press", 1013.0)  # hPa

            # METADATA (for display purposes)
            description = weather.detailed_status              # e.g. "moderate rain"
            ref_time    = weather.reference_time("iso")        # ISO 8601 string

            result = {
                "rainfall":    round(float(rainfall),    2),
                "humidity":    round(float(humidity),    1),
                "wind_speed":  round(float(wind_speed),  2),
                "temperature": round(float(temperature), 1),
                "pressure":    round(float(pressure),    1),
                # ── display metadata ──
                "city":        city,
                "description": description,
                "timestamp":   ref_time,
            }

            logger.info(f"[PyOWM] Current weather fetched for {city}: {result}")
            return result

        except owm_exc.NotFoundError:
            logger.error(f"[PyOWM] City not found: '{city}'. "
                         "Ensure the city name is spelled correctly.")
            return None

        except owm_exc.UnauthorizedError:
            logger.error("[PyOWM] Unauthorized. Your API key is invalid or "
                         "not yet activated (takes up to 2 hours after signup).")
            return None

        except owm_exc.APIRequestError as exc:
            logger.error(f"[PyOWM] API request failed: {exc}")
            return None

        except Exception as exc:
            logger.error(f"[PyOWM] Unexpected error fetching weather: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC  :  5-DAY FORECAST  (3-hour steps)
    # ─────────────────────────────────────────────────────────────────────────

    def get_forecast(self, city: str) -> Optional[List[Dict]]:
        """
        Fetch the 5-day / 3-hour forecast for a city.

        PyOWM call:
            mgr.forecast_at_place("Mumbai,IN", "3h")
            ↓ internally sends →
            GET https://api.openweathermap.org/data/2.5/forecast
                ?q=Mumbai,IN&appid=KEY&units=metric

        Returns up to 40 forecast snapshots (5 days × 8 per day).

        Args:
            city: City name string.

        Returns:
            List of dicts, each containing:
                datetime, rainfall, humidity, wind_speed,
                temperature, pressure, description
            None if the request fails.
        """
        try:
            # ── Step 1 : Get Forecaster object ───────────────────────────────
            # "3h" specifies 3-hour interval forecasts
            forecaster = self.mgr.forecast_at_place(f"{city},IN", "3h")

            # ── Step 2 : Access list of Weather objects ──────────────────────
            forecast = forecaster.forecast       # Forecast object
            weathers = forecast.weathers         # list of Weather objects

            parsed_list: List[Dict] = []

            for w in weathers:
                # Same extraction logic as get_weather()
                rain_dict   = w.rain
                rainfall    = rain_dict.get("3h", rain_dict.get("1h", 0.0))

                wind_dict   = w.wind(unit="meters_sec")
                wind_speed  = wind_dict.get("speed", 0.0)

                temp_dict   = w.temperature(unit="celsius")
                temperature = temp_dict.get("temp", 25.0)

                pressure    = w.pressure.get("press", 1013.0)
                humidity    = w.humidity
                dt_str      = w.reference_time("iso")   # "2025-06-15T09:00:00+00:00"

                parsed_list.append({
                    "datetime":    dt_str,
                    "rainfall":    round(float(rainfall),    2),
                    "humidity":    round(float(humidity),    1),
                    "wind_speed":  round(float(wind_speed),  2),
                    "temperature": round(float(temperature), 1),
                    "pressure":    round(float(pressure),    1),
                    "description": w.detailed_status,
                })

            logger.info(f"[PyOWM] Forecast fetched for {city}: "
                        f"{len(parsed_list)} time steps.")
            return parsed_list

        except owm_exc.NotFoundError:
            logger.error(f"[PyOWM] City not found for forecast: '{city}'")
            return None

        except owm_exc.UnauthorizedError:
            logger.error("[PyOWM] Unauthorized — check API key.")
            return None

        except Exception as exc:
            logger.error(f"[PyOWM] Forecast error: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC  :  CONNECTION TEST
    # ─────────────────────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        """
        Validate the API key by fetching weather for a well-known city.

        Returns:
            True  → key valid, network reachable
            False → key invalid OR network issue
        """
        try:
            self.mgr.weather_at_place("Mumbai,IN")
            logger.info("[PyOWM] Connection test PASSED.")
            return True
        except owm_exc.UnauthorizedError:
            logger.error("[PyOWM] Connection test FAILED — invalid API key.")
            return False
        except Exception as exc:
            logger.error(f"[PyOWM] Connection test FAILED — {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK STANDALONE TEST
#  Run:  python weather_api.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    key = input("Enter your OWM API key: ").strip()
    client = WeatherAPIClient(key)

    print("\n─── Testing connection ───────────────────────")
    ok = client.test_connection()
    if not ok:
        sys.exit(1)

    for test_city in ["Mumbai", "Chennai", "Kolkata"]:
        print(f"\n─── Current weather: {test_city} ─────────────")
        data = client.get_weather(test_city)
        if data:
            for k, v in data.items():
                print(f"  {k:<15}: {v}")

    print(f"\n─── 5-Day Forecast: Mumbai (first 5 slots) ─")
    fc = client.get_forecast("Mumbai")
    if fc:
        for slot in fc[:5]:
            print(
                f"  {slot['datetime']} | "
                f"Rain:{slot['rainfall']:>5} mm | "
                f"Wind:{slot['wind_speed']:>5} m/s | "
                f"Humidity:{slot['humidity']:>3}%"
            )