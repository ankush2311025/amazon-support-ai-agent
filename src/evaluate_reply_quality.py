import pandas as pd


INPUT_FILE = "data/processed/reply_review_sheet_labeled.csv"
OUTPUT_FILE = "data/processed/reply_quality_metrics.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Convert evaluation columns to numeric
    score_columns = [
        "reply_relevance",
        "reply_groundedness",
        "reply_helpfulness",
        "reply_safety",
        "reply_overall",
    ]

    for column in score_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Evidence metrics
    evidence_1_hit = (
        df["evidence_1_relevant"] == 1
    ).mean()

    evidence_2_hit = (
        df["evidence_2_relevant"] == 1
    ).mean()

    evidence_3_hit = (
        df["evidence_3_relevant"] == 1
    ).mean()

    # At least one relevant result in Top-3
    hit_at_3 = (
        (
            (df["evidence_1_relevant"] == 1)
            | (df["evidence_2_relevant"] == 1)
            | (df["evidence_3_relevant"] == 1)
        )
    ).mean()

    # Reply quality averages
    metrics = {
        "examples": len(df),
        "evidence_hit_at_1": evidence_1_hit,
        "evidence_rank2_relevance": evidence_2_hit,
        "evidence_rank3_relevance": evidence_3_hit,
        "evidence_hit_at_3": hit_at_3,
        "reply_relevance_mean": df["reply_relevance"].mean(),
        "reply_groundedness_mean": df["reply_groundedness"].mean(),
        "reply_helpfulness_mean": df["reply_helpfulness"].mean(),
        "reply_safety_mean": df["reply_safety"].mean(),
        "reply_overall_mean": df["reply_overall"].mean(),
    }

    output = pd.DataFrame([metrics])

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nReply Quality Evaluation")
    print("========================")
    print(f"Examples: {len(df)}")

    print(
        f"Evidence Hit@1: "
        f"{evidence_1_hit:.4f}"
    )

    print(
        f"Evidence Hit@3: "
        f"{hit_at_3:.4f}"
    )

    print(
        f"Reply relevance: "
        f"{metrics['reply_relevance_mean']:.2f}/5"
    )

    print(
        f"Reply groundedness: "
        f"{metrics['reply_groundedness_mean']:.2f}/5"
    )

    print(
        f"Reply helpfulness: "
        f"{metrics['reply_helpfulness_mean']:.2f}/5"
    )

    print(
        f"Reply safety: "
        f"{metrics['reply_safety_mean']:.2f}/5"
    )

    print(
        f"Overall reply quality: "
        f"{metrics['reply_overall_mean']:.2f}/5"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()