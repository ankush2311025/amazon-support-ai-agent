import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from escalation_policy import decide_escalation


GOLDEN_FILE = "data/processed/golden_set.csv"
RETRIEVAL_FILE = "data/processed/reply_baseline_evaluation.csv"

OUTPUT_FILE = "data/processed/escalation_evaluation.csv"

TRAINING_FILE = "data/processed/weak_training_set.csv"
TRAINING_EMBEDDINGS_FILE = "data/processed/training_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.40


def classify_with_keyword(
    text,
    model,
    classifier
):
    """
    Keyword classifier first.
    If no keyword rule matches, use the semantic classifier
    with a confidence threshold of 0.40.
    """

    from hybrid_classifier import keyword_classify
    from confidence_hybrid_classifier import semantic_predict

    # 1. Keyword rules get first priority
    prediction = keyword_classify(text)

    if prediction is not None:
        return prediction, "keyword"

    # 2. Semantic classifier fallback
    semantic_prediction, confidence = semantic_predict(
        text,
        model,
        classifier
    )

    # 3. Confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        return semantic_prediction, "semantic"

    return "other_unclear", "semantic_low_confidence"


def main():

    print("Loading golden set...")

    golden = pd.read_csv(
        GOLDEN_FILE,
        dtype={"gold_intent": "string"}
    )

    print("Loading retrieval evaluation...")

    retrieval = pd.read_csv(
        RETRIEVAL_FILE
    )

    # -----------------------------------------------------
    # Load semantic classifier
    # -----------------------------------------------------

    print("Loading training data...")

    training = pd.read_csv(
        TRAINING_FILE,
        dtype={"weak_intent": "string"}
    )

    print("Loading training embeddings...")

    training_embeddings = np.load(
        TRAINING_EMBEDDINGS_FILE
    )

    if len(training) != len(training_embeddings):
        raise ValueError(
            f"Training/embedding mismatch: "
            f"{len(training)} vs "
            f"{len(training_embeddings)}"
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

    # -----------------------------------------------------
    # Retrieval Top-1
    # -----------------------------------------------------

    top1 = retrieval[
        retrieval["rank"] == 1
    ].copy()

    if len(top1) != len(golden):
        raise ValueError(
            f"Expected {len(golden)} Top-1 results, "
            f"found {len(top1)}."
        )

    rows = []

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    for _, example in golden.iterrows():

        example_id = example["example_id"]
        text = example["text"]
        gold_intent = example["gold_intent"]

        # Final confidence-aware classifier
        predicted_intent, method = classify_with_keyword(
            text,
            model,
            classifier
        )

        # Get Top-1 retrieval similarity
        retrieval_row = top1[
            top1["example_id"] == example_id
        ]

        if len(retrieval_row) != 1:
            raise ValueError(
                f"Missing unique retrieval result "
                f"for example_id={example_id}"
            )

        top_similarity = float(
            retrieval_row.iloc[0]["similarity"]
        )

        # Escalation decision
        decision = decide_escalation(
            text=text,
            predicted_intent=predicted_intent,
            top_similarity=top_similarity,
        )

        rows.append({
            "example_id": example_id,
            "gold_intent": gold_intent,
            "customer_message": text,
            "predicted_intent": predicted_intent,
            "classification_method": method,
            "top_similarity": top_similarity,
            "decision": decision["decision"],
            "reason": decision["reason"],
        })

    output = pd.DataFrame(rows)

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total = len(output)

    auto_handle = (
        output["decision"] == "auto_handle"
    ).sum()

    escalate = (
        output["decision"] == "escalate"
    ).sum()

    print("\nEscalation Evaluation")
    print("=====================")

    print(f"Examples: {total}")

    print(
        f"Auto-handle: {auto_handle} "
        f"({auto_handle / total:.1%})"
    )

    print(
        f"Escalate:    {escalate} "
        f"({escalate / total:.1%})"
    )

    print("\nDecision by predicted intent:")

    print(
        pd.crosstab(
            output["predicted_intent"],
            output["decision"],
        ).to_string()
    )

    print("\nClassification method:")

    print(
        output["classification_method"]
        .value_counts()
        .to_string()
    )

    print("\nEscalation reasons:")

    print(
        output.loc[
            output["decision"] == "escalate",
            "reason"
        ]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()