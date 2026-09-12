import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


GOLDEN_FILE = "data/processed/golden_set.csv"
CORPUS_FILE = "data/processed/retrieval_corpus.csv"
EMBEDDINGS_FILE = "data/processed/retrieval_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

OUTPUT_FILE = "data/processed/reply_baseline_evaluation.csv"


def load_resources():
    golden = pd.read_csv(
        GOLDEN_FILE,
        dtype={"gold_intent": "string"}
    )

    corpus = pd.read_csv(CORPUS_FILE)

    embeddings = np.load(EMBEDDINGS_FILE)

    if len(corpus) != len(embeddings):
        raise ValueError(
            f"Corpus/embedding mismatch: "
            f"{len(corpus)} rows vs {len(embeddings)} embeddings"
        )

    model = SentenceTransformer(MODEL_NAME)

    return golden, corpus, embeddings, model


def retrieve_without_leakage(
    query,
    excluded_case_id,
    corpus,
    embeddings,
    model,
    top_k=TOP_K,
):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding

    # Prevent the golden example's own historical case
    # from being retrieved.
    mask = corpus["case_id"].astype(str) == str(excluded_case_id)
    scores[mask.to_numpy()] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = corpus.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results


def main():
    print("Loading resources...")

    golden, corpus, embeddings, model = load_resources()

    rows = []

    print("Evaluating leakage-free reply baseline...")

    for _, example in golden.iterrows():

        query = example["text"]
        gold_intent = example["gold_intent"]
        example_id = example["example_id"]
        conversation_id = example["conversation_id"]

        results = retrieve_without_leakage(
            query=query,
            excluded_case_id=conversation_id,
            corpus=corpus,
            embeddings=embeddings,
            model=model,
            top_k=TOP_K,
        )

        for rank, (_, result) in enumerate(
            results.iterrows(),
            start=1
        ):
            rows.append({
                "example_id": example_id,
                "gold_intent": gold_intent,
                "customer_message": query,
                "rank": rank,
                "similarity": float(result["similarity"]),
                "historical_case_id": result["case_id"],
                "historical_customer_message": (
                    result["customer_message"]
                ),
                "historical_response": (
                    result["amazon_response"]
                ),
            })

    output = pd.DataFrame(rows)

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Only rank-1 results for summary statistics.
    top1 = output[output["rank"] == 1]

    print("\nLeakage-Free Reply Baseline Evaluation")
    print("======================================")
    print(f"Golden examples: {len(golden)}")
    print(f"Retrieved results: {len(output)}")

    print(
        f"Average Top-1 similarity: "
        f"{top1['similarity'].mean():.4f}"
    )

    print(
        f"Median Top-1 similarity: "
        f"{top1['similarity'].median():.4f}"
    )

    print(
        f"Minimum Top-1 similarity: "
        f"{top1['similarity'].min():.4f}"
    )

    print(
        f"Maximum Top-1 similarity: "
        f"{top1['similarity'].max():.4f}"
    )

    # Check that no golden case was retrieved as its own result.
    leakage = output[
        output["historical_case_id"].astype(str)
        == output["example_id"].astype(str)
    ]

    print(f"\nPotential leakage rows: {len(leakage)}")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()