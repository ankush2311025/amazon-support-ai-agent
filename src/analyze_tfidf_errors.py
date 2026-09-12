from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_error_analysis.csv"
)


def main():
    df = pd.read_csv(PREDICTIONS_FILE)

    # Keep only incorrect predictions
    errors = df[
        df["gold_intent"] != df["predicted_intent"]
    ].copy()

    print("\n==============================")
    print("TF-IDF ERROR ANALYSIS")
    print("==============================")

    print(f"Total examples: {len(df)}")
    print(f"Total errors:   {len(errors)}")
    print(
        f"Error rate:     "
        f"{len(errors) / len(df):.2%}"
    )

    # ---------------------------------------------------------
    # TOP CONFUSION PAIRS
    # ---------------------------------------------------------

    confusion_pairs = (
        errors
        .groupby(
            ["gold_intent", "predicted_intent"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
    )

    print("\nTop confusion pairs:")

    print(
        confusion_pairs
        .head(15)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # SAVE ALL ERRORS
    # ---------------------------------------------------------

    errors = errors.merge(
        confusion_pairs,
        on=["gold_intent", "predicted_intent"],
        how="left",
    )

    errors = errors.sort_values(
        "count",
        ascending=False
    )

    errors.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # SHOW EXAMPLES FROM TOP CONFUSIONS
    # ---------------------------------------------------------

    print("\n==============================")
    print("REAL MISCLASSIFIED EXAMPLES")
    print("==============================")

    top_pairs = confusion_pairs.head(5)

    for _, pair in top_pairs.iterrows():

        gold = pair["gold_intent"]
        predicted = pair["predicted_intent"]

        print("\n--------------------------------")
        print(f"GOLD:      {gold}")
        print(f"PREDICTED: {predicted}")
        print(f"COUNT:     {pair['count']}")
        print("--------------------------------")

        examples = errors[
            (errors["gold_intent"] == gold)
            & (errors["predicted_intent"] == predicted)
        ].head(5)

        for _, row in examples.iterrows():
            print(
                f"- [{row['example_id']}] "
                f"{row['text']}"
            )

    print(
        f"\nFull error analysis saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()