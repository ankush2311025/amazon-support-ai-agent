from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "customer_messages.csv"
GOLDEN_FILE = PROJECT_ROOT / "data" / "processed" / "golden_set.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "weak_training_set.csv"

MAX_PER_INTENT = 700


RULES = {
    "damaged_defective_wrong_item": [
        r"\bdamaged\b",
        r"\bdamage(d)?\b",
        r"\bbroken\b",
        r"\bdefective\b",
        r"\bfaulty\b",
        r"\bwrong item\b",
        r"\bwrong product\b",
        r"\bwrong order\b",
    ],

    "payment_unexpected_charge": [
        r"\bcharged twice\b",
        r"\bdouble charged\b",
        r"\bdouble charge\b",
        r"\bunauthorized charge\b",
        r"\bunexpected charge\b",
        r"\bcharged extra\b",
        r"\bcharged more\b",
    ],

    "order_cancellation_modification": [
        r"\bcancel my order\b",
        r"\bcancel the order\b",
        r"\bcancel an order\b",
        r"\bplease cancel\b",
        r"\bwanted to cancel\b",
        r"\bwant to cancel\b",
        r"\bchange my order\b",
        r"\bmodify my order\b",
    ],

    "return_refund": [
        r"\bwant a refund\b",
        r"\bwanted a refund\b",
        r"\bneed a refund\b",
        r"\brequest a refund\b",
        r"\brefund my order\b",
        r"\bwhere is my refund\b",
        r"\bwhen will i get my refund\b",
        r"\brefund status\b",
        r"\bwant to return\b",
        r"\bwanted to return\b",
        r"\breturn my order\b",
    ],

    "package_not_received": [
        r"\bmarked delivered but not received\b",
        r"\bmarked as delivered but not received\b",
        r"\bshows delivered but i didn't receive\b",
        r"\bshows delivered but i did not receive\b",
        r"\bdelivered but i haven't received\b",
        r"\bdelivered but i have not received\b",
        r"\bnever received my package\b",
        r"\bnever received my order\b",
        r"\bnever got my package\b",
        r"\bnever got my order\b",
        r"\bdidn't receive my package\b",
        r"\bdid not receive my package\b",
        r"\bdidn't receive my order\b",
        r"\bdid not receive my order\b",
        r"\bpackage is missing\b",
        r"\border is missing\b",
        r"\bpackage missing\b",
    ],

    "delivery_delay": [
        r"\border is delayed\b",
        r"\bdelivery is delayed\b",
        r"\bpackage is delayed\b",
        r"\border was delayed\b",
        r"\bdelivery was delayed\b",
        r"\bdelivery is late\b",
        r"\border is late\b",
        r"\bpackage is late\b",
        r"\border hasn't arrived\b",
        r"\border has not arrived\b",
        r"\bpackage hasn't arrived\b",
        r"\bpackage has not arrived\b",
        r"\bstill waiting for my order\b",
        r"\bstill waiting for my package\b",
        r"\bdelivery overdue\b",
        r"\border overdue\b",
    ],

    "tracking_delivery_status": [
        r"\bwhere is my package\b",
        r"\bwhere is my order\b",
        r"\bwhere's my package\b",
        r"\bwhere's my order\b",
        r"\btrack my package\b",
        r"\btrack my order\b",
        r"\btracking number\b",
        r"\btracking information\b",
        r"\btracking status\b",
        r"\bdelivery status\b",
        r"\border status\b",
        r"\bdelivery update\b",
        r"\border update\b",
        r"\bwhen will my order arrive\b",
        r"\bwhen will my package arrive\b",
    ],

    "prime_membership_benefit": [
        r"\bprime membership\b",
        r"\bprime member\b",
        r"\bprime benefits\b",
        r"\bprime benefit\b",
        r"\bamazon prime membership\b",
        r"\bprime subscription\b",
        r"\bcancel prime membership\b",
        r"\bprime membership charge\b",
        r"\bprime membership fee\b",
    ],

    "seller_marketplace": [
        r"\bthird party seller\b",
        r"\bthird-party seller\b",
        r"\bmarketplace seller\b",
        r"\bseller on amazon\b",
        r"\bsold by a seller\b",
        r"\bcontact the seller\b",
        r"\bamazon seller\b",
    ],

    "account_access": [
        r"\bcan't log in\b",
        r"\bcannot log in\b",
        r"\bcan't login\b",
        r"\bcannot login\b",
        r"\bcan't sign in\b",
        r"\bcannot sign in\b",
        r"\bforgot my password\b",
        r"\breset my password\b",
        r"\bcan't access my account\b",
        r"\bcannot access my account\b",
        r"\baccount locked\b",
        r"\blocked out of my account\b",
    ],

    "technical_product_issue": [
        r"\bapp keeps crashing\b",
        r"\bapp is crashing\b",
        r"\bwebsite keeps crashing\b",
        r"\bwebsite is crashing\b",
        r"\bwebsite is not working\b",
        r"\bapp is not working\b",
        r"\bapp doesn't work\b",
        r"\bapp does not work\b",
        r"\bwebsite doesn't work\b",
        r"\bwebsite does not work\b",
        r"\bwon't open\b",
        r"\bwill not open\b",
        r"\bcan't open\b",
        r"\bcannot open\b",
        r"\berror message\b",
        r"\btechnical problem\b",
        r"\btechnical issue\b",
    ],
}


def matched_intents(text):
    text = str(text).lower()

    matches = []

    for intent, patterns in RULES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append(intent)
                break

    return matches


def main():
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    df = df[
        df["text"].notna()
        & df["text"].astype(str).str.strip().ne("")
    ].copy()

    # ---------------------------------------------------------
    # GOLDEN SET PROTECTION
    # ---------------------------------------------------------
    golden = pd.read_csv(GOLDEN_FILE)

    golden_ids = set(
        golden["message_id"].astype(str)
    )

    df["message_id"] = df["message_id"].astype(str)

    df = df[
        ~df["message_id"].isin(golden_ids)
    ].copy()

    print(
        f"Messages available after golden-set exclusion: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------
    # WEAK LABELING
    # ---------------------------------------------------------
    rows = []

    for _, row in df.iterrows():

        matches = matched_intents(row["text"])

        # Only accept unambiguous examples.
        if len(matches) != 1:
            continue

        rows.append(
            {
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "timestamp": row["timestamp"],
                "text": row["text"],
                "weak_intent": matches[0],
            }
        )

    training = pd.DataFrame(rows)

    if training.empty:
        raise RuntimeError(
            "No weakly labelled examples were found."
        )

    training = training.drop_duplicates(
        subset=["message_id"]
    )

    # ---------------------------------------------------------
    # BALANCE THE TRAINING SET
    # ---------------------------------------------------------
    balanced_parts = []

    for intent, group in training.groupby("weak_intent"):

        n = min(len(group), MAX_PER_INTENT)

        sampled = group.sample(
            n=n,
            random_state=42
        )

        balanced_parts.append(sampled)

    training = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    training = training.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    training.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print("\n==============================")
    print("WEAK TRAINING SET V2")
    print("==============================")

    print(
        f"Total examples: {len(training):,}"
    )

    print("\nIntent distribution:")

    print(
        training["weak_intent"]
        .value_counts()
        .sort_values(ascending=False)
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()