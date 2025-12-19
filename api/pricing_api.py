import numpy as np
import joblib
from pathlib import Path

# Get the project root directory (parent of api directory)
project_root = Path(__file__).parent.parent
model_path = project_root / "models" / "pricing_gd_model.pkl"

# Load model once (best practice)
model_data = joblib.load(str(model_path))

weights = model_data["weights"]
bias = model_data["bias"]
scaler = model_data["scaler"]
method = model_data["method"]


def predict_price(features):
    """
    features: list of numerical values
    """
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = float(X_scaled @ weights + bias)

    return {
        "predicted_price": prediction,
        "method": method
    }
