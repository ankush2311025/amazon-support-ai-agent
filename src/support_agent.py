import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from hybrid_classifier import keyword_classify
from confidence_hybrid_classifier import semantic_predict
from escalation_policy import decide_escalation



# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TRAINING_FILE = "data/processed/weak_training_set.csv"
TRAINING_EMBEDDINGS_FILE = "data/processed/training_embeddings.npy"

RETRIEVAL_FILE = "data/processed/retrieval_corpus.csv"
RETRIEVAL_EMBEDDINGS_FILE = "data/processed/retrieval_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.40


# ---------------------------------------------------------
# Load models and data
# ---------------------------------------------------------

print("Loading training data...")

training = pd.read_csv(
    TRAINING_FILE,
    dtype={"weak_intent": "string"}
)

training_embeddings = np.load(
    TRAINING_EMBEDDINGS_FILE
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

print("Loading retrieval corpus...")

retrieval_corpus = pd.read_csv(
    RETRIEVAL_FILE
)

print("Loading retrieval embeddings...")

retrieval_embeddings = np.load(
    RETRIEVAL_EMBEDDINGS_FILE
)


# ---------------------------------------------------------
# Intent classification
# ---------------------------------------------------------

def classify_intent(text):

    # Keyword rules first
    keyword_prediction = keyword_classify(
        text
    )

    if keyword_prediction is not None:

        return {
            "intent": keyword_prediction,
            "method": "keyword",
            "confidence": 1.0,
        }

    # Semantic fallback
    semantic_prediction, confidence = semantic_predict(
        text,
        model,
        classifier
    )

    if confidence >= CONFIDENCE_THRESHOLD:

        return {
            "intent": semantic_prediction,
            "method": "semantic",
            "confidence": float(confidence),
        }

    return {
        "intent": "other_unclear",
        "method": "semantic_low_confidence",
        "confidence": float(confidence),
    }


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_evidence(
    text,
    top_k=3
):

    query_embedding = model.encode(
        [text],
        normalize_embeddings=True
    )[0]

    similarities = (
        retrieval_embeddings
        @ query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        row = retrieval_corpus.iloc[index]

        results.append({
            "case_id": row["case_id"],
            "customer_message": row[
                "customer_message"
            ],
            "historical_response": row[
                "amazon_response"
            ],
            "similarity": float(
                similarities[index]
            ),
        })

    return results


# ---------------------------------------------------------
# Response generation
# ---------------------------------------------------------

# ---------------------------------------------------------
# Response generation
# ---------------------------------------------------------

def sanitize_historical_response(response):
    """
    Remove Twitter-specific artifacts from historical responses.
    """

    if not response:
        return ""

    # Remove Twitter usernames
    response = re.sub(r'@\w+', '', response)

    # Remove old t.co links
    response = re.sub(r'https?://t\.co/\S+', '', response)

    # Remove agent initials such as ^DD, ^KM, ^GG
    response = re.sub(r'\^[A-Z]{1,4}\b', '', response)

    # Normalize whitespace
    response = re.sub(r'\s+', ' ', response).strip()

    # Remove spaces before punctuation
    response = re.sub(r'\s+([,.!?])', r'\1', response)

    return response


def generate_reply(
    intent,
    evidence,
    decision
):
    """
    Generate a grounded support reply.

    For common operational intents, use conservative templates.
    Historical responses are used as evidence and fallback wording,
    rather than copied blindly.
    """

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    if decision["decision"] == "escalate":

        return (
            "Thanks for reaching out. "
            "This issue needs additional assistance, "
            "so I’m escalating it to our support team "
            "for further help."
        )

    # -----------------------------------------------------
    # No evidence
    # -----------------------------------------------------

    if not evidence:

        return (
            "Thanks for reaching out. "
            "We need a little more information "
            "to help with this issue."
        )

    # -----------------------------------------------------
    # Conservative intent-specific responses
    # -----------------------------------------------------

    if intent == "return_refund":

        return (
            "I can help with the return and refund. "
            "Please check the available return options for the item. "
            "If you have already returned it, let us know so we can "
            "help look into the refund status."
        )

    if intent == "delivery_delay":

        return (
            "I'm sorry your order is delayed. "
            "Please check the latest delivery status for the order. "
            "If the expected delivery date has already passed, "
            "we can help look into the delay."
        )

    if intent == "package_not_received":

        return (
            "I'm sorry you haven't received your package. "
            "Please check the latest delivery status and the delivery "
            "location information. If the package is marked as delivered "
            "but you still cannot find it, we can help look into it."
        )

    if intent == "tracking_delivery_status":

        return (
            "I can help with the delivery status. "
            "Please check the latest tracking information for the order "
            "to see its current status and expected delivery date."
        )

    if intent == "order_cancellation_modification":

        return (
            "I can help with the order change. "
            "Please check whether the order is still eligible "
            "for cancellation or modification."
        )

    if intent == "damaged_defective_wrong_item":

        return (
            "I'm sorry there is an issue with the item. "
            "Could you tell us whether the item is damaged, defective, "
            "or different from what you ordered? That will help us "
            "determine the appropriate next step."
        )

    if intent == "prime_membership_benefit":

        return (
            "I'd be happy to help with your Prime-related issue. "
            "Could you provide a little more detail about the Prime "
            "benefit or membership issue you're experiencing?"
        )

    if intent == "seller_marketplace":

        return (
            "I'd be happy to help with the marketplace order. "
            "Could you provide a little more detail about the issue "
            "with the seller or order?"
        )

    if intent == "technical_product_issue":

        return (
            "We'd like to help with the technical issue. "
            "Could you tell us a little more about what is not working "
            "and what you have tried so far?"
        )

    # -----------------------------------------------------
    # Fallback: use historical response as evidence
    # -----------------------------------------------------

    historical_response = evidence[0]["historical_response"]

    cleaned_response = sanitize_historical_response(
        historical_response
    )

    if cleaned_response:
        return cleaned_response

    return (
        "Thanks for reaching out. "
        "We need a little more information "
        "to help with this issue."
    )


# ---------------------------------------------------------
# Main agent
# ---------------------------------------------------------

def run_agent(text):

    # 1. Classify intent
    classification = classify_intent(
        text
    )

    intent = classification["intent"]

    # 2. Retrieve historical evidence
    evidence = retrieve_evidence(
        text,
        top_k=3
    )

    top_similarity = (
        evidence[0]["similarity"]
        if evidence
        else 0.0
    )

    # 3. Decide auto-handle vs escalation
    decision = decide_escalation(
        text=text,
        predicted_intent=intent,
        top_similarity=top_similarity,
    )

    # 4. Generate grounded reply
    reply = generate_reply(
        intent=intent,
        evidence=evidence,
        decision=decision,
    )

    return {
        "customer_message": text,
        "intent": intent,
        "classification_method": classification[
            "method"
        ],
        "classification_confidence": classification[
            "confidence"
        ],
        "decision": decision["decision"],
        "escalation_reason": decision["reason"],
        "reply": reply,
        "evidence": evidence,
    }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nAmazon Support Agent")
    print("====================")

    message = input(
        "\nCustomer message: "
    ).strip()

    if not message:

        print("No message provided.")
        raise SystemExit(1)

    result = run_agent(
        message
    )

    print("\nIntent")
    print("------")
    print(
        result["intent"]
    )

    print(
        f"Method: "
        f"{result['classification_method']}"
    )

    print(
        f"Confidence: "
        f"{result['classification_confidence']:.3f}"
    )

    print("\nDecision")
    print("--------")
    print(
        result["decision"]
    )

    print(
        f"Reason: "
        f"{result['escalation_reason']}"
    )

    print("\nDraft Reply")
    print("-----------")
    print(
        result["reply"]
    )

    print("\nEvidence")
    print("--------")

    for i, item in enumerate(
        result["evidence"],
        start=1
    ):

        print(
            f"\n[{i}] "
            f"Similarity: "
            f"{item['similarity']:.3f}"
        )

        print(
            f"Historical customer: "
            f"{item['customer_message']}"
        )

        print(
            f"Historical response: "
            f"{item['historical_response']}"
        )