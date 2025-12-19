import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

# Get the project root directory (parent of api directory)
project_root = Path(__file__).parent.parent
model_path = project_root / "models" / "pricing_gd_model.pkl"

# Load model once (best practice)
model_data = joblib.load(str(model_path))

weights = model_data["weights"]
bias = model_data["bias"]
gd_mode = model_data.get("gd_mode", "unknown")
robust_scaler = model_data.get("robust_scaler")
standard_scaler = model_data.get("standard_scaler")
features_meta = model_data.get("features", [])
metrics = model_data.get("metrics", {})


def _build_feature_vector(payload: Dict[str, Any]) -> np.ndarray:
    """Create the feature vector in the exact training order with engineered sine/cosine features."""
    try:
        demand_index = float(payload["demand_index"])
        operational_cost = float(payload["operational_cost"])
        marketing_intensity = float(payload["marketing_intensity"])
        seasonality_index = float(payload["seasonality_index"])
        competition_pressure = float(payload["competition_pressure"])
        time_slot = float(payload["time_slot"])
        day_of_week = float(payload["day_of_week"])
    except KeyError as e:
        raise KeyError(f"Missing field: {e}")

    time_slot_sin = np.sin(2 * np.pi * time_slot / 28)
    time_slot_cos = np.cos(2 * np.pi * time_slot / 28)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Order must match training (see src/preprocessing/pricing_preprocess.py)
    vector = [
        demand_index,
        operational_cost,
        marketing_intensity,
        seasonality_index,
        competition_pressure,
        time_slot_sin,
        time_slot_cos,
        day_of_week_sin,
        day_of_week_cos,
    ]
    return np.array(vector, dtype=float).reshape(1, -1)


def _scale_features(X: np.ndarray) -> np.ndarray:
    """Apply the same scalers used during training."""
    X_scaled = X.copy()
    if robust_scaler is not None:
        X_scaled[:, :2] = robust_scaler.transform(X[:, :2])  # demand_index, operational_cost
    if standard_scaler is not None:
        X_scaled[:, 2:5] = standard_scaler.transform(X[:, 2:5])  # marketing_intensity, seasonality_index, competition_pressure
    return X_scaled


def predict_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: dict with raw feature values
    required keys: demand_index, operational_cost, marketing_intensity,
                   seasonality_index, competition_pressure, time_slot, day_of_week
    """
    X = _build_feature_vector(payload)
    X_scaled = _scale_features(X)

    prediction = float(X_scaled @ weights + bias)

    return {
        "predicted_price": prediction,
        "gd_mode": gd_mode,
        "metrics": metrics,
        "features_order": features_meta,
    }
