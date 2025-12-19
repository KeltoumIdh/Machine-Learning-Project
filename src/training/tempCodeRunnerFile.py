import sys
from pathlib import Path

# Add project root to Python path
# Handle both script execution and Jupyter notebook execution
try:
    # Running as a script
    project_root = Path(__file__).parent.parent.parent
except NameError:
    # Running in Jupyter notebook - find project root by looking for src directory
    current = Path.cwd()
    while current != current.parent:
        if (current / 'src').exists() and (current / 'data').exists():
            project_root = current
            break
        current = current.parent
    else:
        project_root = Path.cwd()  # Fallback to current directory

sys.path.insert(0, str(project_root))

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing.pricing_preprocess import load_and_preprocess_pricing_data

# =====================================================
# Gradient Descent Variants (FROM SCRATCH)
# =====================================================

def batch_gd(X, y, lr=0.01, epochs=500):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(epochs):
        y_pred = X @ w + b
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        w -= lr * dw
        b -= lr * db

    return w, b


def stochastic_gd(X, y, lr=0.01, epochs=10):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(epochs):
        for i in range(m):
            y_pred = X[i] @ w + b
            dw = X[i] * (y_pred - y[i])
            db = y_pred - y[i]

            w -= lr * dw
            b -= lr * db

    return w, b


def mini_batch_gd(X, y, lr=0.01, epochs=100, batch_size=32):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(epochs):
        for i in range(0, m, batch_size):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            y_pred = Xb @ w + b
            dw = (1/len(Xb)) * Xb.T @ (y_pred - yb)
            db = (1/len(Xb)) * np.sum(y_pred - yb)

            w -= lr * dw
            b -= lr * db

    return w, b


# =====================================================
# Training + Evaluation
# =====================================================
if __name__ == "__main__":

    # Use absolute path based on project root
    data_path = project_root / "data" / "raw" / "part1_pricing_gradient_descent_dirty.csv"

    X, y, scaler = load_and_preprocess_pricing_data(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Train models --------
    w_batch, b_batch = batch_gd(X_train, y_train)
    w_sgd, b_sgd = stochastic_gd(X_train, y_train)
    w_mini, b_mini = mini_batch_gd(X_train, y_train)

    # -------- Evaluate models --------
    def evaluate(w, b, name):
        y_pred = X_test @ w + b
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} → MSE: {mse:.4f} | R2: {r2:.4f}")
        return mse

    mse_batch = evaluate(w_batch, b_batch, "Batch GD")
    mse_sgd = evaluate(w_sgd, b_sgd, "Stochastic GD")
    mse_mini = evaluate(w_mini, b_mini, "Mini-Batch GD")

    # -------- Select best model --------
    best_weights, best_bias = w_mini, b_mini
    best_name = "Mini-Batch GD"

    # -------- Save model --------
    model_path = project_root / "models" / "pricing_gd_model.pkl"
    joblib.dump(
        {
            "weights": best_weights,
            "bias": best_bias,
            "scaler": scaler,
            "method": best_name
        },
        str(model_path)
    )

    print(f"\n✅ Best model saved: {best_name}")
