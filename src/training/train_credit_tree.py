import sys
from pathlib import Path

# =====================================================
# Project path handling (script & notebook compatible)
# =====================================================
try:
    # Running as a script
    project_root = Path(__file__).parent.parent.parent
except NameError:
    # Running in Jupyter notebook
    current = Path.cwd()
    while current != current.parent:
        if (current / 'src').exists() and (current / 'data').exists():
            project_root = current
            break
        current = current.parent
    else:
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

# =====================================================
# Imports
# =====================================================
import joblib
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from src.preprocessing.credit_preprocess import load_and_preprocess_credit_data


if __name__ == "__main__":

    # =====================================================
    # Load and preprocess data
    # =====================================================
    data_path = project_root / "data" / "raw" / "part3_credit_risk_dirty.csv"

    X, y, encoder = load_and_preprocess_credit_data(str(data_path))

    # Train / Test split
    # Stratify keeps class distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,           # 20% of data used for testing
        random_state=42,         # Make results reproducible
        stratify=y               # Preserve target class distribution in splits
    )

    # =====================================================
    # Hyperparameter space (controls overfitting)
    # =====================================================
    param_dist = {
        # Maximum depth of the tree (limits how deep splits can go to control overfitting)
        "max_depth": [3, 5, 7, 10],
        # Minimum number of samples required to be at a leaf node (higher values increase regularization)
        "min_samples_leaf": [5, 10, 20, 50],
        # Minimum number of samples required to split an internal node
        "min_samples_split": [2, 5, 10]
    }

    # =====================================================
    # Base models (Gini & Entropy)
    # =====================================================
    base_gini = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )

    base_entropy = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    )

    # =====================================================
    # Randomized Search + Cross-Validation
    # =====================================================
    # CV = 5 folds
    # Scoring = accuracy (classification task)
    search_gini = RandomizedSearchCV(
        estimator=base_gini,            # Decision tree using Gini impurity
        param_distributions=param_dist, # Hyperparameter search space
        n_iter=10,                      # Number of random settings sampled for search
        cv=5,                           # 5-fold cross-validation
        scoring="accuracy",             # Use accuracy to score models
        random_state=42,                # Ensure reproducibility of search
        n_jobs=-1                       # Use all available CPUs for speed
    )

    search_entropy = RandomizedSearchCV(
        estimator=base_entropy,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    # Fit searches (this trains multiple trees internally)
    search_gini.fit(X_train, y_train)
    search_entropy.fit(X_train, y_train)

    # =====================================================
    # Best models from CV
    # =====================================================
    best_gini = search_gini.best_estimator_
    best_entropy = search_entropy.best_estimator_

    # =====================================================
    # Evaluation on test set (FINAL comparison)
    # =====================================================
    def evaluate(model, name):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        return acc

    acc_gini = evaluate(best_gini, "Gini (CV)")
    acc_entropy = evaluate(best_entropy, "Entropy (CV)")

    # =====================================================
    # Select best final model
    # =====================================================
    if acc_gini >= acc_entropy:
        best_model = best_gini
        best_criterion = "Gini"
    else:
        best_model = best_entropy
        best_criterion = "Entropy"

    # =====================================================
    # Save model
    # =====================================================
    model_path = project_root / "models" / "credit_risk_tree.pkl"
    joblib.dump(
        {
            "model": best_model,
            "encoder": encoder,
            "criterion": best_criterion
        },
        str(model_path)
    )

    print(f"\n✅ Credit Risk model saved ({best_criterion})")
