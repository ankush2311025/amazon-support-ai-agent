from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon_cases.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_corpus.csv"
)


def main():
    print(f"Reading: {INPUT_FILE}")

    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:
            case = json.loads(line)

            messages = case.get("messages", [])

            if not messages:
                continue

            # Find first customer message
            customer_message = None
            customer_index = None

            for i, message in enumerate(messages):
                if message.get("inbound") is True:
                    text = str(message.get("text", "")).strip()

                    if text:
                        customer_message = text
                        customer_index = i
                        break

            if not customer_message:
                continue

            # Find the first Amazon/support response after it
            amazon_response = None

            for message in messages[customer_index + 1:]:
                if message.get("inbound") is False:
                    text = str(message.get("text", "")).strip()

                    if text:
                        amazon_response = text
                        break

            # We need an actual support response for grounding
            if not amazon_response:
                continue

            rows.append(
                {
                    "case_id": case.get("conversation_id"),
                    "customer_message": customer_message,
                    "amazon_response": amazon_response,
                }
            )

    corpus = pd.DataFrame(rows)

    if corpus.empty:
        raise RuntimeError(
            "No usable historical cases found."
        )

    # Remove exact duplicate customer/response pairs
    corpus = corpus.drop_duplicates(
        subset=[
            "customer_message",
            "amazon_response",
        ]
    )

    corpus = corpus.reset_index(drop=True)

    corpus.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n==============================")
    print("RETRIEVAL CORPUS")
    print("==============================")

    print(f"Usable cases: {len(corpus):,}")

    print("\nExample:")

    example = corpus.iloc[0]

    print("\nCUSTOMER:")
    print(example["customer_message"])

    print("\nAMAZON RESPONSE:")
    print(example["amazon_response"])

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()