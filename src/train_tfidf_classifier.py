from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "weak_training_set.csv"
GOLDEN_FILE = PROJECT_ROOT / "data" / "processed" / "golden_set.csv"

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_predictions.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_metrics.json"
)

CONFUSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_confusion_matrix.csv"
)


def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_FILE)

    print("Loading golden set...")
    gold_df = pd.read_csv(GOLDEN_FILE)

    # ---------------------------------------------------------
    # TRAINING DATA
    # ---------------------------------------------------------

    X_train = train_df["text"].fillna("").astype(str)
    y_train = train_df["weak_intent"].astype(str)

    # ---------------------------------------------------------
    # GOLDEN TEST DATA
    # ---------------------------------------------------------

    X_test = gold_df["text"].fillna("").astype(str)
    y_test = gold_df["gold_intent"].astype(str)

    print(f"\nTraining examples: {len(X_train):,}")
    print(f"Golden examples:   {len(X_test):,}")

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Test matrix shape:     {X_test_tfidf.shape}")

    # ---------------------------------------------------------
    # CLASSIFIER
    # ---------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    print("Generating predictions...")

    y_pred = model.predict(X_test_tfidf)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n==============================")
    print("TF-IDF + LOGISTIC REGRESSION")
    print("==============================")

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-intent results:")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    print(report)

    # ---------------------------------------------------------
    # SAVE PREDICTIONS
    # ---------------------------------------------------------

    predictions = gold_df[
        ["example_id", "message_id", "conversation_id", "text", "gold_intent"]
    ].copy()

    predictions["predicted_intent"] = y_pred

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # SAVE METRICS
    # ---------------------------------------------------------

    metrics = {
        "model": "tfidf_logistic_regression",
        "training_examples": len(X_train),
        "golden_examples": len(X_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    labels = sorted(
        set(y_test) | set(y_pred)
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.index.name = "gold_intent"
    cm_df.to_csv(
        CONFUSION_FILE,
        encoding="utf-8-sig",
    )

    print("\nSaved:")
    print(f"- {PREDICTIONS_FILE}")
    print(f"- {METRICS_FILE}")
    print(f"- {CONFUSION_FILE}")


if __name__ == "__main__":
    main()