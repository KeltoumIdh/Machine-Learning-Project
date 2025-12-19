import sys
from pathlib import Path

# =====================================================
# Project path handling
# =====================================================
try:
    project_root = Path(__file__).parent.parent.parent
except NameError:
    project_root = Path.cwd()

sys.path.insert(0, str(project_root))

# =====================================================
# Imports
# =====================================================
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.ticket_preprocess import load_and_preprocess_tickets


if __name__ == "__main__":

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    data_path = project_root / "data" / "raw" / "part2_tickets_multiclass_dirty.csv"

    X_text, y = load_and_preprocess_tickets(str(data_path))

    # -------------------------------------------------
    # Train / Test split
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------
    # Vectorization (TF-IDF)
    # -------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -------------------------------------------------
    # One-vs-Rest Logistic Regression
    # -------------------------------------------------
    logreg_ovr = LogisticRegression(
        multi_class="ovr",
        max_iter=1000
    )
    logreg_ovr.fit(X_train_vec, y_train)

    # -------------------------------------------------
    # Softmax (Multinomial) Logistic Regression
    # -------------------------------------------------
    logreg_softmax = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000
    )
    logreg_softmax.fit(X_train_vec, y_train)

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    def evaluate(model, name):
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        return acc

    acc_ovr = evaluate(logreg_ovr, "One-vs-Rest")
    acc_softmax = evaluate(logreg_softmax, "Softmax")

    # -------------------------------------------------
    # Select best model
    # -------------------------------------------------
    if acc_softmax >= acc_ovr:
        best_model = logreg_softmax
        best_type = "Softmax"
    else:
        best_model = logreg_ovr
        best_type = "One-vs-Rest"

    # -------------------------------------------------
    # Save model + vectorizer
    # -------------------------------------------------
    model_path = project_root / "models" / "ticket_classifier.pkl"
    joblib.dump(
        {
            "model": best_model,
            "vectorizer": vectorizer,
            "type": best_type
        },
        str(model_path)
    )

    print(f"\n✅ Ticket classifier saved ({best_type})")
