import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


GOLDEN_FILE = "data/processed/golden_set.csv"
TRAINING_FILE = "data/processed/weak_training_set.csv"
TRAINING_EMBEDDINGS_FILE = "data/processed/training_embeddings.npy"

OUTPUT_FILE = "data/processed/confidence_hybrid_predictions.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

# We will test several thresholds later.
CONFIDENCE_THRESHOLD = 0.60


def keyword_classify(text):
    text = str(text).lower()

    rules = [
        (
            "package_not_received",
            [
                r"\bdelivered\b.*\b(not|never|haven't|didn't|dont|don't)\b.*\b(receive|received|get|got)\b",
                r"\b(not|never|haven't|didn't|dont|don't)\b.*\b(receive|received|get|got)\b.*\bpackage\b",
                r"\bpackage\b.*\bmissing\b",
                r"\bpackage\b.*\blost\b",
            ],
        ),
        (
            "damaged_defective_wrong_item",
            [
                r"\bdamaged\b",
                r"\bbroken\b",
                r"\bdefective\b",
                r"\bwrong item\b",
                r"\bwrong product\b",
                r"\bdoesn't work\b",
                r"\bnot working\b",
                r"\bfaulty\b",
            ],
        ),
        (
            "payment_unexpected_charge",
            [
                r"\bcharged twice\b",
                r"\btwo charges\b",
                r"\bdouble charge\b",
                r"\bunauthori[sz]ed\b.*\bcharge\b",
                r"\bpayment\b.*\bproblem\b",
                r"\bcharge\b.*\bproblem\b",
                r"\bcharged\b.*\bwrong\b",
            ],
        ),
        (
            "order_cancellation_modification",
            [
                r"\bcancel\b.*\border\b",
                r"\bcancel my order\b",
                r"\bcancelled order\b",
                r"\bchange\b.*\border\b",
                r"\bmodify\b.*\border\b",
            ],
        ),
        (
            "return_refund",
            [
                r"\brefund\b",
                r"\breturn\b.*\b(item|product|order)\b",
                r"\breturning\b.*\b(item|product|order)\b",
                r"\breturned\b.*\b(item|product|order)\b",
            ],
        ),
        (
            "prime_membership_benefit",
            [
                r"\bprime membership\b",
                r"\bprime member\b",
                r"\bprime benefits?\b",
                r"\bprime subscription\b",
                r"\bprime charge\b",
                r"\bwhy.*\bprime\b.*\bcharge\b",
            ],
        ),
        (
            "seller_marketplace",
            [
                r"\bthird[- ]party seller\b",
                r"\bmarketplace seller\b",
                r"\bseller\b.*\bamazon\b",
                r"\bseller\b.*\border\b",
                r"\bsold by\b",
            ],
        ),
        (
            "account_access",
            [
                r"\bcan't login\b",
                r"\bcannot login\b",
                r"\bcan't log in\b",
                r"\bcannot log in\b",
                r"\blogin\b.*\bproblem\b",
                r"\bsign in\b.*\bproblem\b",
                r"\baccount\b.*\blocked\b",
                r"\baccount\b.*\baccess\b",
                r"\bforgot\b.*\bpassword\b",
            ],
        ),
        (
            "technical_product_issue",
            [
                r"\bapp\b.*\bnot working\b",
                r"\bwebsite\b.*\bnot working\b",
                r"\btechnical\b.*\bproblem\b",
                r"\btechnical\b.*\bissue\b",
                r"\bfire tv\b",
                r"\bkindle\b.*\bproblem\b",
                r"\bdevice\b.*\bproblem\b",
                r"\bnot working\b",
            ],
        ),
        (
            "delivery_delay",
            [
                r"\blate\b",
                r"\bdelayed\b",
                r"\bdelay\b",
                r"\boverdue\b",
                r"\bwas supposed to\b.*\barrive\b",
                r"\bshould have arrived\b",
                r"\bexpected\b.*\bnot arrived\b",
                r"\bdelivery\b.*\blate\b",
            ],
        ),
        (
            "tracking_delivery_status",
            [
                r"\bwhere is my\b.*\border\b",
                r"\bwhere is my\b.*\bpackage\b",
                r"\btrack\b.*\border\b",
                r"\btracking\b",
                r"\btracking number\b",
                r"\bdelivery status\b",
                r"\bwhen will\b.*\barrive\b",
                r"\bwhen is\b.*\barrive\b",
            ],
        ),
    ]

    matches = []

    for intent, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append(intent)
                break

    # Only trust unambiguous keyword matches.
    if len(set(matches)) == 1:
        return matches[0]

    return None


def train_semantic_classifier():
    training_df = pd.read_csv(TRAINING_FILE)

    embeddings = np.load(
        TRAINING_EMBEDDINGS_FILE
    )

    if len(training_df) != len(embeddings):
        raise ValueError(
            f"Training/embedding mismatch: "
            f"{len(training_df)} vs {len(embeddings)}"
        )

    y = training_df["weak_intent"].values

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    classifier.fit(
        embeddings,
        y
    )

    return classifier


def semantic_predict(
    text,
    model,
    classifier,
):
    embedding = model.encode(
        [text],
        normalize_embeddings=True
    )

    probabilities = classifier.predict_proba(
        embedding
    )[0]

    best_index = np.argmax(probabilities)

    intent = classifier.classes_[best_index]
    confidence = float(probabilities[best_index])

    return intent, confidence


def predict(
    text,
    model,
    classifier,
):
    # First try high-precision keyword rules.
    keyword_prediction = keyword_classify(text)

    if keyword_prediction is not None:
        return (
            keyword_prediction,
            "keyword",
            1.0,
        )

    # Fall back to semantic classifier.
    intent, confidence = semantic_predict(
        text,
        model,
        classifier,
    )

    # Reject uncertain semantic predictions.
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "other_unclear",
            "semantic_low_confidence",
            confidence,
        )

    return (
        intent,
        "semantic",
        confidence,
    )


def main():
    print("Loading golden set...")

    golden = pd.read_csv(
        GOLDEN_FILE,
        dtype={"gold_intent": "string"}
    )

    print("Loading embedding model...")
    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Training semantic classifier...")
    classifier = train_semantic_classifier()

    print("Running confidence-aware classifier...")

    rows = []

    for _, row in golden.iterrows():

        intent, method, confidence = predict(
            row["text"],
            model,
            classifier,
        )

        rows.append({
            "example_id": row["example_id"],
            "gold_intent": row["gold_intent"],
            "text": row["text"],
            "predicted_intent": intent,
            "prediction_method": method,
            "confidence": confidence,
        })

    output = pd.DataFrame(rows)

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nConfidence Hybrid Results")
    print("=========================")

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print("\nPrediction methods:")
    print(
        output["prediction_method"]
        .value_counts()
        .to_string()
    )

    print("\nPredicted intents:")
    print(
        output["predicted_intent"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()