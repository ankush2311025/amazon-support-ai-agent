import pandas as pd


INPUT_FILE = "data/processed/retrieval_eval_sample_labeled.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Group retrieval results by query/example
    grouped = df.groupby("example_id")

    hit_at_1 = []
    hit_at_3 = []
    reciprocal_ranks = []

    for example_id, group in grouped:
        group = group.sort_values("rank")

        # Relevant results: retrieval_relevant == 1
        relevant = group["retrieval_relevant"].tolist()

        # Hit@1
        hit_at_1.append(1 if relevant[0] == 1 else 0)

        # Hit@3
        hit_at_3.append(1 if any(relevant) else 0)

        # MRR
        rr = 0.0
        for position, is_relevant in enumerate(relevant, start=1):
            if is_relevant == 1:
                rr = 1.0 / position
                break

        reciprocal_ranks.append(rr)

    print("\nRetrieval Evaluation")
    print("====================")
    print(f"Queries: {len(hit_at_1)}")
    print(f"Hit@1:   {sum(hit_at_1) / len(hit_at_1):.4f}")
    print(f"Hit@3:   {sum(hit_at_3) / len(hit_at_3):.4f}")
    print(f"MRR:     {sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}")


if __name__ == "__main__":
    main()