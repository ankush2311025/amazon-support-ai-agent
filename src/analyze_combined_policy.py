import pandas as pd

from tune_escalation_policy import apply_policy


INPUT_FILE = "data/processed/escalation_evaluation.csv"

OUTPUT_FILE = (
    "data/processed/"
    "combined_policy_unsafe_cases.csv"
)


def main():

    print("Loading escalation evaluation...")

    df = pd.read_csv(INPUT_FILE)

    unsafe_cases = []

    for _, row in df.iterrows():

        result = apply_policy(
            row,
            "combined"
        )

        # Only inspect cases that the combined policy
        # decided to auto-handle.
        if result["decision"] != "auto_handle":
            continue

        # Unsafe = predicted intent != human gold intent.
        if row["gold_intent"] != row["predicted_intent"]:

            unsafe_cases.append({
                "example_id": row["example_id"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": row["predicted_intent"],
                "classification_method": row[
                    "classification_method"
                ],
                "top_similarity": row[
                    "top_similarity"
                ],
                "customer_message": row[
                    "customer_message"
                ],
            })

    unsafe = pd.DataFrame(
        unsafe_cases
    )

    unsafe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCombined Policy Safety Analysis")
    print("===============================")

    print(
        f"Unsafe auto-handled cases: "
        f"{len(unsafe)}"
    )

    if len(unsafe) == 0:
        print("\nNo unsafe auto-handled cases.")
        return

    print("\nUnsafe cases by gold intent:")
    print(
        unsafe["gold_intent"]
        .value_counts()
        .to_string()
    )

    print("\nIntent confusions:")
    print(
        unsafe
        .groupby(
            [
                "gold_intent",
                "predicted_intent"
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
        .to_string(index=False)
    )

    print("\nUnsafe auto-handled examples:")
    print(
        unsafe[
            [
                "example_id",
                "gold_intent",
                "predicted_intent",
                "classification_method",
                "top_similarity",
                "customer_message",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()