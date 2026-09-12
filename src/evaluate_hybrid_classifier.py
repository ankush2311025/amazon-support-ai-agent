import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


GOLDEN_FILE = "data/processed/golden_set.csv"
PREDICTIONS_FILE = "data/processed/hybrid_classifier_predictions.csv"

OUTPUT_METRICS = "data/processed/hybrid_classifier_metrics.csv"
OUTPUT_CONFUSION = "data/processed/hybrid_classifier_confusion_matrix.csv"


def main():
    golden = pd.read_csv(
        GOLDEN_FILE,
        dtype={"gold_intent": "string"}
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        dtype={"predicted_intent": "string"}
    )

    # Safety check: predictions must correspond to the same examples
    if len(golden) != len(predictions):
        raise ValueError(
            f"Row count mismatch: golden={len(golden)}, "
            f"predictions={len(predictions)}"
        )

    if not golden["example_id"].equals(predictions["example_id"]):
        raise ValueError(
            "example_id mismatch between golden set and predictions."
        )

    y_true = golden["gold_intent"]
    y_pred = predictions["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\nHybrid Classifier Evaluation")
    print("============================")
    print(f"Examples:    {len(y_true)}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-intent performance")
    print("=======================")

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    print(report)

    # Save overall metrics
    metrics_df = pd.DataFrame([
        {
            "model": "hybrid_keyword_semantic",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "examples": len(y_true),
        }
    ])

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False
    )

    # Save confusion matrix
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.index.name = "true_intent"
    cm_df.to_csv(OUTPUT_CONFUSION)

    print(f"Metrics saved to: {OUTPUT_METRICS}")
    print(f"Confusion matrix saved to: {OUTPUT_CONFUSION}")


if __name__ == "__main__":
    main()