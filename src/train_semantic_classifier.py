from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weak_training_set.csv"
)

TRAIN_EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_embeddings.npy"
)

GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "golden_set.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_predictions.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_metrics.json"
)

CONFUSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_confusion_matrix.csv"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_FILE)

    print("Loading training embeddings...")
    X_train = np.load(TRAIN_EMBEDDINGS_FILE)

    y_train = train_df["weak_intent"].astype(str).values

    print(f"Training examples: {len(y_train):,}")
    print(f"Embedding shape:   {X_train.shape}")

    # ---------------------------------------------------------
    # LOAD GOLDEN SET
    # ---------------------------------------------------------

    print("\nLoading golden set...")
    gold_df = pd.read_csv(GOLDEN_FILE)

    X_text_test = (
        gold_df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    y_test = gold_df["gold_intent"].astype(str).values

    # ---------------------------------------------------------
    # CREATE GOLDEN EMBEDDINGS
    # ---------------------------------------------------------

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Creating golden-set embeddings...")

    X_test = model.encode(
        X_text_test,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    X_test = np.asarray(X_test)

    print(f"Golden embedding shape: {X_test.shape}")

    # ---------------------------------------------------------
    # TRAIN CLASSIFIER
    # ---------------------------------------------------------

    print("\nTraining semantic classifier...")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    classifier.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    print("Generating predictions...")

    y_pred = classifier.predict(X_test)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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
    print("SEMANTIC CLASSIFIER RESULTS")
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
        [
            "example_id",
            "message_id",
            "conversation_id",
            "text",
            "gold_intent",
        ]
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
        "model": "all-MiniLM-L6-v2 + Logistic Regression",
        "training_examples": len(y_train),
        "golden_examples": len(y_test),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

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