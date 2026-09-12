import pandas as pd


INPUT_FILE = "data/processed/escalation_evaluation.csv"
OUTPUT_FILE = "data/processed/escalation_safety_analysis.csv"


def main():

    print("Loading escalation evaluation...")

    df = pd.read_csv(INPUT_FILE)

    # -----------------------------------------------------
    # Identify whether the predicted intent is correct
    # -----------------------------------------------------

    df["intent_correct"] = (
        df["gold_intent"] == df["predicted_intent"]
    )

    # -----------------------------------------------------
    # Auto-handle safety
    # -----------------------------------------------------

    auto = df[
        df["decision"] == "auto_handle"
    ].copy()

    safe_auto = auto[
        auto["intent_correct"]
    ]

    unsafe_auto = auto[
        ~auto["intent_correct"]
    ]

    total = len(df)
    auto_count = len(auto)
    escalate_count = (
        df["decision"] == "escalate"
    ).sum()

    safe_count = len(safe_auto)
    unsafe_count = len(unsafe_auto)

    print("\nEscalation Safety Analysis")
    print("==========================")

    print(f"Total examples:       {total}")
    print(
        f"Auto-handle:          {auto_count} "
        f"({auto_count / total:.1%})"
    )
    print(
        f"Escalate:             {escalate_count} "
        f"({escalate_count / total:.1%})"
    )

    print("\nAuto-handle quality")
    print("-------------------")

    print(
        f"Safe auto-handle:     {safe_count} "
        f"({safe_count / total:.1%} of all)"
    )

    print(
        f"Unsafe auto-handle:   {unsafe_count} "
        f"({unsafe_count / total:.1%} of all)"
    )

    print(
        f"Auto-handle precision:"
        f" {safe_count / auto_count:.1%}"
    )

    # -----------------------------------------------------
    # Unsafe auto-handle by intent
    # -----------------------------------------------------

    print("\nUnsafe auto-handle by gold intent")
    print("---------------------------------")

    if len(unsafe_auto) > 0:
        print(
            unsafe_auto["gold_intent"]
            .value_counts()
            .to_string()
        )
    else:
        print("None")

    # -----------------------------------------------------
    # Most common intent confusions
    # -----------------------------------------------------

    print("\nUnsafe intent confusions")
    print("------------------------")

    if len(unsafe_auto) > 0:

        confusion = (
            unsafe_auto
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

        print(
            confusion.to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # Save full safety analysis
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Save unsafe examples separately
    # -----------------------------------------------------

    unsafe_file = (
        "data/processed/"
        "unsafe_auto_handle_examples.csv"
    )

    unsafe_auto.to_csv(
        unsafe_file,
        index=False
    )

    print(
        f"\nSaved analysis to: {OUTPUT_FILE}"
    )

    print(
        f"Saved unsafe examples to: {unsafe_file}"
    )


if __name__ == "__main__":
    main()