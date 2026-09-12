import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


GOLDEN_FILE = "data/processed/golden_set.csv"
TRAINING_FILE = "data/processed/weak_training_set.csv"
TRAINING_EMBEDDINGS_FILE = "data/processed/training_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


VALID_LABELS = [
    "delivery_delay",
    "package_not_received",
    "tracking_delivery_status",
    "order_cancellation_modification",
    "return_refund",
    "damaged_defective_wrong_item",
    "payment_unexpected_charge",
    "prime_membership_benefit",
    "seller_marketplace",
    "account_access",
    "technical_product_issue",
    "other_unclear",
]


# ---------------------------------------------------------
# 1. High-confidence keyword rules
# ---------------------------------------------------------

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

    # Only use keyword result when exactly one intent matches.
    # Conflicting keyword signals are treated as ambiguous.
    if len(set(matches)) == 1:
        return matches[0]

    return None


# ---------------------------------------------------------
# 2. Semantic classifier
# ---------------------------------------------------------

def train_semantic_classifier():
    training_df = pd.read_csv(TRAINING_FILE)

    X = np.load(TRAINING_EMBEDDINGS_FILE)
    y = training_df["weak_intent"].values

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    classifier.fit(X, y)

    return classifier


# ---------------------------------------------------------
# 3. Hybrid prediction
# ---------------------------------------------------------

def predict(text, model, classifier):
    keyword_prediction = keyword_classify(text)

    # High-confidence keyword rule wins.
    if keyword_prediction is not None:
        return keyword_prediction, "keyword"

    # Otherwise fall back to semantic classifier.
    embedding = model.encode(
        [text],
        normalize_embeddings=True
    )

    prediction = classifier.predict(embedding)[0]

    return prediction, "semantic"


# ---------------------------------------------------------
# 4. Run on golden set
# ---------------------------------------------------------

def main():
    print("Loading data...")

    golden_df = pd.read_csv(
        GOLDEN_FILE,
        dtype={
            "gold_intent": "string"
        }
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Training semantic classifier...")
    classifier = train_semantic_classifier()

    predictions = []
    methods = []

    print("Running hybrid classifier...")

    for text in golden_df["text"]:
        prediction, method = predict(
            text,
            model,
            classifier
        )

        predictions.append(prediction)
        methods.append(method)

    golden_df["predicted_intent"] = predictions
    golden_df["prediction_method"] = methods

    output_file = (
        "data/processed/hybrid_classifier_predictions.csv"
    )

    golden_df.to_csv(
        output_file,
        index=False
    )

    print("\nHybrid classifier results")
    print("=========================")

    print(
        golden_df["prediction_method"]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved predictions to:")
    print(output_file)


if __name__ == "__main__":
    main()