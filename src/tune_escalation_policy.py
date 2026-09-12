import pandas as pd

from escalation_policy import decide_escalation


INPUT_FILE = "data/processed/escalation_evaluation.csv"
OUTPUT_FILE = "data/processed/escalation_policy_results.csv"


def apply_policy(row, policy):
    """
    Apply one escalation policy variant to a single example.
    """

    predicted_intent = row["predicted_intent"]
    top_similarity = float(row["top_similarity"])
    text = str(row["customer_message"])

    # -----------------------------------------------------
    # Policy 1: Current baseline
    # -----------------------------------------------------

    if policy == "baseline":
        return decide_escalation(
            text=text,
            predicted_intent=predicted_intent,
            top_similarity=top_similarity,
        )

    # -----------------------------------------------------
    # Policy 2:
    # Escalate all semantic predictions for ambiguous
    # delivery-related intents.
    # -----------------------------------------------------

    if policy == "delivery_safety":

        delivery_intents = {
            "delivery_delay",
            "package_not_received",
            "tracking_delivery_status",
            "prime_membership_benefit",
        }

        if predicted_intent in delivery_intents:

            delivery_words = [
                "deliver",
                "delivery",
                "package",
                "parcel",
                "shipment",
                "arrived",
                "late",
                "delay",
                "delayed",
                "tracking",
                "track",
                "prime",
            ]

            text_lower = text.lower()

            # If the message contains delivery-state language,
            # require stronger evidence.
            if any(
                word in text_lower
                for word in delivery_words
            ):
                if top_similarity < 0.80:
                    return {
                        "decision": "escalate",
                        "reason": (
                            "Ambiguous delivery-related "
                            "intent with moderate evidence."
                        ),
                    }

        return decide_escalation(
            text=text,
            predicted_intent=predicted_intent,
            top_similarity=top_similarity,
        )

    # -----------------------------------------------------
    # Policy 3:
    # Semantic predictions require stronger retrieval
    # evidence before auto-handling.
    # -----------------------------------------------------

    if policy == "semantic_evidence":

        # Classification method is stored in evaluation output.
        method = row["classification_method"]

        if method == "semantic":
            if top_similarity < 0.80:
                return {
                    "decision": "escalate",
                    "reason": (
                        "Semantic prediction without "
                        "strong retrieval evidence."
                    ),
                }

        return decide_escalation(
            text=text,
            predicted_intent=predicted_intent,
            top_similarity=top_similarity,
        )

    # -----------------------------------------------------
    # Policy 4:
    # Combine semantic evidence + delivery safety.
    # -----------------------------------------------------

    if policy == "combined":

        method = row["classification_method"]

        delivery_intents = {
            "delivery_delay",
            "package_not_received",
            "tracking_delivery_status",
            "prime_membership_benefit",
        }

        # Semantic predictions need strong evidence
        if method == "semantic" and top_similarity < 0.80:
            return {
                "decision": "escalate",
                "reason": (
                    "Semantic prediction without "
                    "strong retrieval evidence."
                ),
            }

        # Delivery-related semantic predictions need
        # even stronger evidence.
        if (
            method == "semantic"
            and predicted_intent in delivery_intents
            and top_similarity < 0.85
        ):
            return {
                "decision": "escalate",
                "reason": (
                    "Ambiguous delivery-related "
                    "semantic prediction."
                ),
            }

        return decide_escalation(
            text=text,
            predicted_intent=predicted_intent,
            top_similarity=top_similarity,
        )

    raise ValueError(
        f"Unknown policy: {policy}"
    )


def evaluate_policy(df, policy):

    decisions = []

    for _, row in df.iterrows():

        result = apply_policy(
            row,
            policy
        )

        decisions.append(
            result["decision"]
        )

    evaluated = df.copy()

    evaluated["policy_decision"] = decisions

    # Intent correctness
    evaluated["intent_correct"] = (
        evaluated["gold_intent"]
        == evaluated["predicted_intent"]
    )

    auto = evaluated[
        evaluated["policy_decision"]
        == "auto_handle"
    ]

    safe_auto = auto[
        auto["intent_correct"]
    ]

    unsafe_auto = auto[
        ~auto["intent_correct"]
    ]

    total = len(evaluated)

    auto_count = len(auto)
    safe_count = len(safe_auto)
    unsafe_count = len(unsafe_auto)

    precision = (
        safe_count / auto_count
        if auto_count > 0
        else 0
    )

    return {
        "policy": policy,
        "total": total,
        "auto_handle": auto_count,
        "escalate": total - auto_count,
        "auto_handle_rate": auto_count / total,
        "safe_auto_handle": safe_count,
        "unsafe_auto_handle": unsafe_count,
        "safe_auto_handle_rate": safe_count / total,
        "auto_handle_precision": precision,
    }


def main():

    print("Loading escalation evaluation...")

    df = pd.read_csv(
        INPUT_FILE
    )

    policies = [
        "baseline",
        "delivery_safety",
        "semantic_evidence",
        "combined",
    ]

    results = []

    for policy in policies:

        print(
            f"Evaluating policy: {policy}"
        )

        result = evaluate_policy(
            df,
            policy
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nEscalation Policy Comparison")
    print("============================")

    print(
        results_df[
            [
                "policy",
                "auto_handle",
                "escalate",
                "auto_handle_rate",
                "safe_auto_handle",
                "unsafe_auto_handle",
                "safe_auto_handle_rate",
                "auto_handle_precision",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()