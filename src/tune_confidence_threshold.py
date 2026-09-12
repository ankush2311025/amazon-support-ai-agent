import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


GOLDEN_FILE = "data/processed/golden_set.csv"
TRAINING_FILE = "data/processed/weak_training_set.csv"
TRAINING_EMBEDDINGS_FILE = "data/processed/training_embeddings.npy"

OUTPUT_FILE = "data/processed/confidence_threshold_results.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def main():
    print("Loading golden set...")

    golden = pd.read_csv(
        GOLDEN_FILE,
        dtype={"gold_intent": "string"}
    )

    print("Loading training data...")

    training = pd.read_csv(
        TRAINING_FILE,
        dtype={"weak_intent": "string"}
    )

    training_embeddings = np.load(
        TRAINING_EMBEDDINGS_FILE
    )

    if len(training) != len(training_embeddings):
        raise ValueError(
            f"Training/embedding mismatch: "
            f"{len(training)} vs {len(training_embeddings)}"
        )

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Training semantic classifier...")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    classifier.fit(
        training_embeddings,
        training["weak_intent"].values
    )

    print("Encoding golden set...")

    golden_embeddings = model.encode(
        golden["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    probabilities = classifier.predict_proba(
        golden_embeddings
    )

    classes = classifier.classes_

    # Best semantic prediction and confidence
    best_indices = np.argmax(
        probabilities,
        axis=1
    )

    semantic_predictions = classes[best_indices]

    semantic_confidences = probabilities[
        np.arange(len(golden)),
        best_indices
    ]

    # Keyword predictions
    from hybrid_classifier import keyword_classify

    keyword_predictions = []

    for text in golden["text"]:
        keyword_predictions.append(
            keyword_classify(text)
        )

    results = []

    y_true = golden["gold_intent"].values

    for threshold in THRESHOLDS:

        final_predictions = []
        methods = []

        for i in range(len(golden)):

            keyword_prediction = keyword_predictions[i]

            if keyword_prediction is not None:
                final_predictions.append(
                    keyword_prediction
                )
                methods.append("keyword")

            elif semantic_confidences[i] >= threshold:
                final_predictions.append(
                    semantic_predictions[i]
                )
                methods.append("semantic")

            else:
                final_predictions.append(
                    "other_unclear"
                )
                methods.append(
                    "semantic_low_confidence"
                )

        accuracy = accuracy_score(
            y_true,
            final_predictions
        )

        macro_f1 = f1_score(
            y_true,
            final_predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_true,
            final_predictions,
            average="weighted",
            zero_division=0,
        )

        other_count = sum(
            prediction == "other_unclear"
            for prediction in final_predictions
        )

        keyword_count = methods.count(
            "keyword"
        )

        semantic_count = methods.count(
            "semantic"
        )

        low_confidence_count = methods.count(
            "semantic_low_confidence"
        )

        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "other_unclear_count": other_count,
            "keyword_count": keyword_count,
            "semantic_count": semantic_count,
            "low_confidence_count": low_confidence_count,
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nConfidence Threshold Comparison")
    print("================================")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best = results_df.loc[
        results_df["macro_f1"].idxmax()
    ]

    print("\nBest threshold by Macro F1")
    print("==========================")
    print(
        f"Threshold: {best['threshold']:.2f}"
    )
    print(
        f"Accuracy: {best['accuracy']:.4f}"
    )
    print(
        f"Macro F1: {best['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1: {best['weighted_f1']:.4f}"
    )
    print(
        f"other_unclear: "
        f"{int(best['other_unclear_count'])}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()