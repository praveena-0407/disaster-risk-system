"""
model.py ─ Machine Learning Models for Disaster Risk Prediction
===============================================================
Three separate Random Forest classifiers are trained — one per risk type:

    flood_model    → LOW / MEDIUM / HIGH
    cyclone_model  → YES / NO
    overall_model  → LOW / MEDIUM / HIGH

Each model is a scikit-learn Pipeline:
    StandardScaler  →  RandomForestClassifier

Why a Pipeline?
    • Scaler is fitted only on training data (no data leakage).
    • At prediction time, new data is automatically scaled with the
      same mean/std learned during training.
    • The whole pipeline is serialised as one object.

Usage
─────
    from data_generator import generate_training_data
    from model import DisasterRiskModel

    df    = generate_training_data(8000)
    model = DisasterRiskModel()
    model.train(df)
    model.save()

    result = model.predict({
        "rainfall": 22, "humidity": 88,
        "wind_speed": 8, "temperature": 30, "pressure": 1005
    })
    # → {'flood_risk': 'HIGH', 'flood_prob': 0.87, ...}
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import RANDOM_FOREST_PARAMS, MODEL_SAVE_PATH, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class DisasterRiskModel:
    """
    Container for all three Random Forest risk classifiers.

    Attributes
    ──────────
    flood_model   : sklearn Pipeline (scaler + RF) for flood risk
    cyclone_model : sklearn Pipeline (scaler + RF) for cyclone risk
    overall_model : sklearn Pipeline (scaler + RF) for overall risk
    feature_columns : list of feature names (must match prediction input)
    """

    def __init__(self):
        self.flood_model:    Optional[Pipeline] = None
        self.cyclone_model:  Optional[Pipeline] = None
        self.overall_model:  Optional[Pipeline] = None
        self.feature_columns: List[str]         = FEATURE_COLUMNS

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        """
        Build a fresh sklearn Pipeline: StandardScaler → RandomForest.

        StandardScaler normalises each feature to mean=0, std=1.
        This is important for RF because it prevents features with large
        numeric ranges (e.g. pressure 940–1040) from dominating
        importance calculations.
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
        ])

    def _train_one(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        label_name: str,
    ) -> Tuple[Pipeline, Dict]:
        """
        Train a single Pipeline and return it + evaluation metrics.

        Uses an 80/20 stratified train/test split.
        Also runs 5-fold cross-validation on training data.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=0.20,
            stratify=y_all,
            random_state=42,
        )

        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred   = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 5-fold CV on training set for more robust accuracy estimate
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5, scoring="accuracy", n_jobs=-1
        )

        logger.info(f"\n[Model] ── {label_name} ─────────────────────────────")
        logger.info(f"  Test Accuracy  : {accuracy*100:.2f}%")
        logger.info(f"  CV Accuracy    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        return pipeline, {
            "accuracy":    accuracy,
            "cv_mean":     float(cv_scores.mean()),
            "cv_std":      float(cv_scores.std()),
            "report":      classification_report(y_test, y_pred, output_dict=True),
            "classes":     list(pipeline.classes_),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC : TRAIN
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train all three models on the provided DataFrame.

        Args:
            df: DataFrame containing FEATURE_COLUMNS + label columns:
                flood_risk, cyclone_risk, overall_risk

        Returns:
            Dict with accuracy metrics for each model.
        """
        logger.info(f"[Model] Training on {len(df):,} samples...")

        X = df[self.feature_columns].values

        metrics = {}

        # ── Flood risk model ─────────────────────────────────────────────────
        y_flood = df["flood_risk"].values
        self.flood_model, flood_metrics = self._train_one(X, y_flood, "Flood Risk")
        metrics["flood"] = flood_metrics

        # ── Cyclone risk model ───────────────────────────────────────────────
        y_cyclone = df["cyclone_risk"].values
        self.cyclone_model, cyclone_metrics = self._train_one(X, y_cyclone, "Cyclone Risk")
        metrics["cyclone"] = cyclone_metrics

        # ── Overall risk model ───────────────────────────────────────────────
        y_overall = df["overall_risk"].values
        self.overall_model, overall_metrics = self._train_one(X, y_overall, "Overall Risk")
        metrics["overall"] = overall_metrics

        # Convenience flat keys for the Streamlit UI
        metrics["flood_accuracy"]   = flood_metrics["accuracy"]
        metrics["cyclone_accuracy"] = cyclone_metrics["accuracy"]
        metrics["overall_accuracy"] = overall_metrics["accuracy"]

        logger.info("\n[Model] ✅ All models trained successfully.")
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC : PREDICT
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, features: Dict) -> Dict:
        """
        Run inference on a single weather observation.

        Args:
            features: Dict with keys matching self.feature_columns.
                      e.g. {'rainfall': 12.5, 'humidity': 78,
                            'wind_speed': 9.2, 'temperature': 30,
                            'pressure': 1003}

        Returns:
            Dict containing:
                flood_risk, flood_prob, flood_classes, flood_proba_all
                cyclone_risk, cyclone_prob, cyclone_classes, cyclone_proba_all
                overall_risk, overall_prob, overall_classes, overall_proba_all
        """
        self._check_trained()

        # Build the feature row in the correct column order
        X = np.array([[features[col] for col in self.feature_columns]])

        def _infer(model: Pipeline) -> Tuple[str, float, list, list]:
            label      = model.predict(X)[0]
            proba_all  = model.predict_proba(X)[0].tolist()
            confidence = max(proba_all)
            classes    = list(model.classes_)
            return label, confidence, classes, proba_all

        flood_label,   flood_prob,   flood_classes,   flood_proba   = _infer(self.flood_model)
        cyclone_label, cyclone_prob, cyclone_classes, cyclone_proba = _infer(self.cyclone_model)
        overall_label, overall_prob, overall_classes, overall_proba = _infer(self.overall_model)

        return {
            "flood_risk":         flood_label,
            "flood_prob":         float(flood_prob),
            "flood_classes":      flood_classes,
            "flood_proba_all":    flood_proba,

            "cyclone_risk":       cyclone_label,
            "cyclone_prob":       float(cyclone_prob),
            "cyclone_classes":    cyclone_classes,
            "cyclone_proba_all":  cyclone_proba,

            "overall_risk":       overall_label,
            "overall_prob":       float(overall_prob),
            "overall_classes":    overall_classes,
            "overall_proba_all":  overall_proba,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC : FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────────

    def get_feature_importance(self, model_name: str = "flood") -> Optional[List[Tuple]]:
        """
        Return feature importances sorted descending.

        Args:
            model_name: "flood" | "cyclone" | "overall"

        Returns:
            List of (feature_name, importance_score) tuples, or None.
        """
        model_map = {
            "flood":   self.flood_model,
            "cyclone": self.cyclone_model,
            "overall": self.overall_model,
        }
        model = model_map.get(model_name)
        if model is None:
            return None

        importances = model.named_steps["rf"].feature_importances_
        pairs = list(zip(self.feature_columns, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC : SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """Serialise all three models + metadata to a pickle file."""
        self._check_trained()
        save_path = path or MODEL_SAVE_PATH
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        payload = {
            "flood_model":      self.flood_model,
            "cyclone_model":    self.cyclone_model,
            "overall_model":    self.overall_model,
            "feature_columns":  self.feature_columns,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"[Model] Saved to {save_path}")
        return save_path

    def load(self, path: Optional[str] = None) -> None:
        """Deserialise models from a pickle file."""
        load_path = path or MODEL_SAVE_PATH
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        self.flood_model     = payload["flood_model"]
        self.cyclone_model   = payload["cyclone_model"]
        self.overall_model   = payload["overall_model"]
        self.feature_columns = payload["feature_columns"]

        logger.info(f"[Model] Loaded from {load_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE : GUARDS
    # ─────────────────────────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if not all([self.flood_model, self.cyclone_model, self.overall_model]):
            raise RuntimeError(
                "Models are not trained. Call model.train(df) first, "
                "or load saved models with model.load()."
            )

    def is_trained(self) -> bool:
        return all([self.flood_model, self.cyclone_model, self.overall_model])