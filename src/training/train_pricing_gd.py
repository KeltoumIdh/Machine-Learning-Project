import sys
from pathlib import Path

# -------------------------------------------------
# Project root
# -------------------------------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing.pricing_preprocess import load_and_preprocess_pricing_data

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = project_root / "data" / "raw" / "part1_pricing_gradient_descent_dirty.csv"
MODEL_DIR = project_root / "models"
MODEL_DIR.mkdir(exist_ok=True)

LEARNING_RATE = 0.001
N_ITERATIONS = 2000
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

GD_MODES = ["batch", "stochastic", "mini-batch"]

# -------------------------------------------------
# LOAD DATA (EDA OFF)
# -------------------------------------------------
X, y, meta = load_and_preprocess_pricing_data(
    str(DATA_PATH),
    show_eda=True,
    save_clean=True
)

# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# -------------------------------------------------
# SCALING (same logic as notebook)
# -------------------------------------------------
X_train_scaled = X_train.copy().astype(float)
X_test_scaled = X_test.copy().astype(float)

robust_scaler = RobustScaler()
X_train_scaled[:, [0, 1]] = robust_scaler.fit_transform(X_train[:, [0, 1]])
X_test_scaled[:, [0, 1]] = robust_scaler.transform(X_test[:, [0, 1]])

standard_scaler = StandardScaler()
X_train_scaled[:, [2, 3, 4]] = standard_scaler.fit_transform(X_train[:, [2, 3, 4]])
X_test_scaled[:, [2, 3, 4]] = standard_scaler.transform(X_test[:, [2, 3, 4]])

# Bias term
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# -------------------------------------------------
# GRADIENT DESCENT
# -------------------------------------------------
def gradient_descent(X, y, lr, n_iter, mode="batch", batch_size=32):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []

    for i in range(n_iter):
        if mode == "batch":
            Xb, yb = X, y
        elif mode == "stochastic":
            idx = np.random.randint(0, m)
            Xb, yb = X[idx:idx+1], y[idx:idx+1]
        else:
            idx = np.random.randint(0, m, batch_size)
            Xb, yb = X[idx], y[idx]

        y_pred = Xb @ theta
        error = y_pred - yb
        loss = (1 / (2 * len(yb))) * np.sum(error ** 2)
        losses.append(loss)

        grad = (1 / len(yb)) * (Xb.T @ error)
        theta -= lr * grad

        if i % 200 == 0:
            print(f"{mode.upper():12s} | Iter {i:4d} | Loss {loss:.4f}")

    return theta, losses

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
def evaluate(X, y, theta):
    y_pred = X @ theta
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2, y_pred

# -------------------------------------------------
# TRAINING
# -------------------------------------------------
results = {}

for mode in GD_MODES:
    print(f"\n{'='*50}")
    print(f"Training with {mode.upper()} Gradient Descent")
    print(f"{'='*50}")

    theta, losses = gradient_descent(
        X_train_scaled,
        y_train,
        LEARNING_RATE,
        N_ITERATIONS,
        mode=mode,
        batch_size=BATCH_SIZE
    )

    test_mse, test_r2, _ = evaluate(X_test_scaled, y_test, theta)

    results[mode] = {
        "theta": theta,
        "losses": losses,
        "mse": test_mse,
        "r2": test_r2
    }

# -------------------------------------------------
# SELECT BEST MODEL
# -------------------------------------------------
best_mode = min(results, key=lambda m: results[m]["mse"])
best = results[best_mode]

theta = best["theta"]
bias = theta[0]
weights = theta[1:]

print("\n" + "="*50)
print("FINAL TRAINING SUMMARY")
print("="*50)
print(f"Best Gradient Descent Mode : {best_mode.upper()}")
print(f"Test MSE                  : {best['mse']:.4f}")
print(f"Test R²                   : {best['r2']:.4f}")

print("\nModel Parameters:")
print(f"Bias (θ₀): {bias:.6f}")
print("Weights:")
for name, value in zip(meta["features"], weights):
    print(f"  {name:25s}: {value:.6f}")
print("="*50)

# -------------------------------------------------
# SAVE MODEL FOR API
# -------------------------------------------------
with open(MODEL_DIR / "pricing_gd_model.pkl", "wb") as f:
    pickle.dump(
        {
            "weights": weights,
            "bias": bias,
            "gd_mode": best_mode,
            "robust_scaler": robust_scaler,
            "standard_scaler": standard_scaler,
            "features": meta["features"],
            "metrics": {
                "test_mse": best["mse"],
                "test_r2": best["r2"]
            }
        },
        f
    )

print("\n✅ Model and scalers saved for API consumption")
