import joblib
import numpy as np
from pathlib import Path

# Get the project root directory (parent of api directory)
project_root = Path(__file__).parent.parent
model_path = project_root / "models" / "credit_risk_tree.pkl"

# Load trained model
data = joblib.load(str(model_path))

model = data["model"]
encoder = data["encoder"]
criterion = data["criterion"]

FEATURE_ORDER = [
    "age",
    "monthly_income",
    "credit_history_years",
    "debt_ratio",
    "job_stability_years",
]


def predict_risk(client_data):
    X = np.array([client_data[f] for f in FEATURE_ORDER]).reshape(1, -1)

    pred_encoded = model.predict(X)[0]
    risk_level = encoder.inverse_transform([pred_encoded])[0]

    return {
        "risk_level": risk_level,
        "model_criterion": criterion,
        "explanation": "Risk estimated using decision rules based on client financial profile."
    }
