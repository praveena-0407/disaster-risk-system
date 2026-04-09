"""
predictor.py ─ Prediction Pipeline
====================================
Combines live weather data from PyOWM with the trained ML models
to produce disaster risk predictions.

  DisasterPredictor.predict(weather_dict)
      → runs all three models on the current weather snapshot

  DisasterPredictor.predict_forecast(forecast_list)
      → runs all three models on each 3-hour forecast step

The predictor is the single point of contact between:
    weather_api.py  (data)  ←→  predictor.py  ←→  model.py  (inference)
"""

import logging
from typing import Dict, List, Optional

from model import DisasterRiskModel

logger = logging.getLogger(__name__)


class DisasterPredictor:
    """
    Wraps DisasterRiskModel and adds:
      • Input validation before inference
      • Forecast batch prediction
      • Human-readable summary string
      • Alert level helper

    Usage
    ─────
        model     = DisasterRiskModel()
        model.train(df)
        predictor = DisasterPredictor(model)

        # Current weather prediction
        result = predictor.predict(weather_dict)

        # Forecast predictions
        results = predictor.predict_forecast(forecast_list)
    """

    # Minimum required keys in a weather dict
    REQUIRED_KEYS = {"rainfall", "humidity", "wind_speed", "temperature", "pressure"}

    def __init__(self, model: DisasterRiskModel):
        self.model = model

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, weather: Dict) -> Dict:
        """
        Predict risk levels for one weather observation.

        Args:
            weather: Dict from WeatherAPIClient.get_weather()
                     Must contain: rainfall, humidity, wind_speed,
                                   temperature, pressure

        Returns:
            Full prediction dict (all three models) + original weather dict.
        """
        self._validate(weather)

        result = self.model.predict(weather)
        result["weather"] = weather          # attach raw features for display

        logger.info(
            f"[Predictor] {weather.get('city','?')} → "
            f"Flood:{result['flood_risk']} "
            f"Cyclone:{result['cyclone_risk']} "
            f"Overall:{result['overall_risk']}"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST BATCH PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def predict_forecast(self, forecast_list: List[Dict]) -> List[Dict]:
        """
        Run predictions over all forecast time steps.

        Args:
            forecast_list: List of dicts from WeatherAPIClient.get_forecast()
                           Each dict includes 'datetime' + weather features.

        Returns:
            List of prediction dicts, each containing:
                datetime, weather features, + all three risk predictions.
        """
        results = []
        for item in forecast_list:
            try:
                self._validate(item)
                pred = self.model.predict(item)
                pred["datetime"] = item.get("datetime", "")
                pred["weather"]  = item
                results.append(pred)
            except Exception as exc:
                logger.warning(f"[Predictor] Skipping forecast item: {exc}")

        logger.info(f"[Predictor] Forecast predictions: {len(results)} steps.")
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self, result: Dict) -> str:
        """One-line human-readable prediction summary."""
        return (
            f"Flood: {result['flood_risk']} "
            f"({result['flood_prob']*100:.1f}%)  |  "
            f"Cyclone: {result['cyclone_risk']} "
            f"({result['cyclone_prob']*100:.1f}%)  |  "
            f"Overall: {result['overall_risk']} "
            f"({result['overall_prob']*100:.1f}%)"
        )

    def alert_level(self, result: Dict) -> str:
        """
        Return the highest alert level across all three risk dimensions.
        Used for a single top-line status banner.
        """
        if (result["overall_risk"]  == "HIGH"
                or result["flood_risk"]  == "HIGH"
                or result["cyclone_risk"] == "YES"):
            return "HIGH"
        if (result["overall_risk"]  == "MEDIUM"
                or result["flood_risk"]  == "MEDIUM"):
            return "MEDIUM"
        return "LOW"

    def _validate(self, weather: Dict) -> None:
        """Raise ValueError if required weather keys are missing."""
        missing = self.REQUIRED_KEYS - set(weather.keys())
        if missing:
            raise ValueError(
                f"Weather dict missing required keys: {missing}\n"
                f"Present keys: {set(weather.keys())}"
            )