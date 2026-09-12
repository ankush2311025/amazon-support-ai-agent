import re


def has_human_request(text):
    text = str(text).lower()

    patterns = [
        r"\bspeak to (a )?human\b",
        r"\bspeak to (an )?agent\b",
        r"\btalk to (a )?human\b",
        r"\btalk to (an )?agent\b",
        r"\breal person\b",
        r"\bcustomer service\b",
        r"\bcustomer support\b",
        r"\brepresentative\b",
        r"\bmanager\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def has_multiple_issues(text):
    text = str(text).lower()

    issue_groups = {
        "delivery": [
            "late", "delay", "delayed", "delivery",
            "package", "parcel", "tracking"
        ],
        "payment": [
            "charge", "charged", "payment", "billing",
            "paid", "payment"
        ],
        "return_refund": [
            "refund", "return", "returned"
        ],
        "product": [
            "broken", "damaged", "defective",
            "wrong item", "not working"
        ],
        "order": [
            "cancel", "cancellation", "change my order"
        ],
        "account": [
            "account", "login", "log in", "password"
        ],
    }

    matched_groups = 0

    for keywords in issue_groups.values():
        if any(keyword in text for keyword in keywords):
            matched_groups += 1

    return matched_groups >= 2


def decide_escalation(
    text,
    predicted_intent,
    top_similarity,
):
    """
    Explainable escalation policy.

    Returns:
        {
            "decision": "auto_handle" or "escalate",
            "reason": "..."
        }
    """

    text = str(text)

    # 1. No clear intent
    if predicted_intent == "other_unclear":
        return {
            "decision": "escalate",
            "reason": "Intent is unclear or does not match a supported category."
        }

    # 2. Customer explicitly asks for a human
    if has_human_request(text):
        return {
            "decision": "escalate",
            "reason": "Customer explicitly requested human/customer support assistance."
        }

    # 3. Multiple issues in one message
    if has_multiple_issues(text):
        return {
            "decision": "escalate",
            "reason": "Message contains multiple potentially conflicting issues."
        }

    # 4. Sensitive account-specific issues
    if predicted_intent == "account_access":
        return {
            "decision": "escalate",
            "reason": "Account-specific access issues may require account verification."
        }

    # 5. Payment issues
    if predicted_intent == "payment_unexpected_charge":
        return {
            "decision": "escalate",
            "reason": "Payment or charge issues may require account/order verification."
        }

    # 6. Weak retrieval evidence
    if top_similarity < 0.65:
        return {
            "decision": "escalate",
            "reason": (
                f"Retrieved historical evidence is weak "
                f"(top similarity={top_similarity:.3f})."
            )
        }

    # Otherwise auto-handle
    return {
        "decision": "auto_handle",
        "reason": (
            f"Intent is supported and relevant historical evidence "
            f"is available (top similarity={top_similarity:.3f})."
        )
    }


def main():
    test_cases = [
        (
            "My package says delivered but I never received it",
            "package_not_received",
            0.7913,
        ),
        (
            "I was charged twice for the same order",
            "payment_unexpected_charge",
            0.8098,
        ),
        (
            "My order is late and I also want a refund",
            "delivery_delay",
            0.72,
        ),
        (
            "I need to speak to a human agent",
            "other_unclear",
            0.80,
        ),
        (
            "What is my order status?",
            "tracking_delivery_status",
            0.73,
        ),
    ]

    print("\nEscalation Policy Tests")
    print("=======================")

    for text, intent, similarity in test_cases:
        result = decide_escalation(
            text,
            intent,
            similarity,
        )

        print("\nCustomer:", text)
        print("Intent:", intent)
        print("Decision:", result["decision"])
        print("Reason:", result["reason"])


if __name__ == "__main__":
    main()