from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_FILE = PROJECT_ROOT / "data" / "processed" / "golden_set.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "majority_baseline_predictions.csv"


def main():
    print(f"Reading golden set: {GOLDEN_FILE}")

    df = pd.read_csv(GOLDEN_FILE)

    # Find the most frequent gold label
    majority_label = df["gold_intent"].value_counts().idxmax()

    print(f"\nMajority class: {majority_label}")
    print(f"Majority count: {(df['gold_intent'] == majority_label).sum()}")

    # Predict the same label for every example
    df["predicted_intent"] = majority_label

    y_true = df["gold_intent"]
    y_pred = df["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n==============================")
    print("MAJORITY BASELINE RESULTS")
    print("==============================")

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-intent results:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # Save predictions for reproducibility
    output_columns = [
        "example_id",
        "text",
        "gold_intent",
        "predicted_intent"
    ]

    df[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Predictions saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()