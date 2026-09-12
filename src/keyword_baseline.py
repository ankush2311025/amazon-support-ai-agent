from pathlib import Path
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_FILE = PROJECT_ROOT / "data" / "processed" / "golden_set.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "keyword_baseline_predictions.csv"


# Rules are based on the intent definitions, not on the gold labels.
RULES = [
    (
        "damaged_defective_wrong_item",
        [
            r"\bdamaged\b",
            r"\bdamage\b",
            r"\bbroken\b",
            r"\bdefective\b",
            r"\bfaulty\b",
            r"\bwrong item\b",
            r"\bwrong product\b",
            r"\bwrong order\b",
        ],
    ),
    (
        "payment_unexpected_charge",
        [
            r"\bcharged twice\b",
            r"\bdouble charge\b",
            r"\bcharged twice\b",
            r"\bunexpected charge\b",
            r"\bunauthorized charge\b",
            r"\bpayment\b",
            r"\bcharged\b",
            r"\bcharge\b",
        ],
    ),
    (
        "order_cancellation_modification",
        [
            r"\bcancel\b",
            r"\bcancellation\b",
            r"\bmodify my order\b",
            r"\bchange my order\b",
            r"\bchange the order\b",
        ],
    ),
    (
        "return_refund",
        [
            r"\brefund\b",
            r"\brefunded\b",
            r"\breturn\b",
            r"\breturning\b",
            r"\bmoney back\b",
            r"\brefund status\b",
        ],
    ),
    (
        "package_not_received",
        [
            r"\bnot received\b",
            r"\bnever received\b",
            r"\bdidn't receive\b",
            r"\bdid not receive\b",
            r"\bnot delivered\b",
            r"\bmarked delivered\b",
            r"\bshows delivered\b",
            r"\bdelivered but\b",
        ],
    ),
    (
        "delivery_delay",
        [
            r"\blate\b",
            r"\bdelayed\b",
            r"\bdelay\b",
            r"\boverdue\b",
            r"\bstill waiting\b",
            r"\bhasn't arrived\b",
            r"\bhas not arrived\b",
            r"\bshould have arrived\b",
        ],
    ),
    (
        "tracking_delivery_status",
        [
            r"\btracking\b",
            r"\btrack my\b",
            r"\bwhere is my\b",
            r"\bwhere's my\b",
            r"\bdelivery status\b",
            r"\border status\b",
            r"\bwhen will.*arrive\b",
            r"\bwhen does.*arrive\b",
            r"\beta\b",
        ],
    ),
    (
        "prime_membership_benefit",
        [
            r"\bprime membership\b",
            r"\bamazon prime\b",
            r"\bprime member\b",
            r"\bprime benefits\b",
            r"\bprime benefit\b",
            r"\bprime subscription\b",
            r"\bprime fee\b",
            r"\bprime charge\b",
        ],
    ),
    (
        "seller_marketplace",
        [
            r"\bseller\b",
            r"\bthird party seller\b",
            r"\bthird-party seller\b",
            r"\bmarketplace\b",
            r"\bsold by\b",
        ],
    ),
    (
        "account_access",
        [
            r"\blog in\b",
            r"\blogin\b",
            r"\bsign in\b",
            r"\bsignin\b",
            r"\bpassword\b",
            r"\baccount access\b",
            r"\bcan't access my account\b",
            r"\bcannot access my account\b",
        ],
    ),
    (
        "technical_product_issue",
        [
            r"\bapp\b",
            r"\bwebsite\b",
            r"\bsite\b",
            r"\berror\b",
            r"\bbug\b",
            r"\bnot working\b",
            r"\bdoesn't work\b",
            r"\bdoes not work\b",
            r"\bcrash\b",
            r"\btechnical\b",
        ],
    ),
]


def classify(text):
    text = str(text).lower()

    matches = []

    for intent, patterns in RULES:
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append(intent)
                break

    # No useful signal
    if not matches:
        return "other_unclear"

    # If multiple intents match, use the first rule.
    # This is intentionally simple and will expose ambiguity
    # as a failure mode of the rule-based baseline.
    return matches[0]


def main():
    print(f"Reading golden set: {GOLDEN_FILE}")

    df = pd.read_csv(GOLDEN_FILE)

    df["predicted_intent"] = df["text"].apply(classify)

    y_true = df["gold_intent"]
    y_pred = df["predicted_intent"]

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

    print("\n==============================")
    print("KEYWORD BASELINE RESULTS")
    print("==============================")

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-intent results:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    output_columns = [
        "example_id",
        "text",
        "gold_intent",
        "predicted_intent",
    ]

    df[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nPredictions saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()