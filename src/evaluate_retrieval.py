from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_FILE = PROJECT_ROOT / "data" / "processed" / "golden_set.csv"
CORPUS_FILE = PROJECT_ROOT / "data" / "processed" / "retrieval_corpus.csv"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "processed" / "retrieval_embeddings.npy"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def main():
    print("Loading golden set...")
    golden = pd.read_csv(GOLDEN_FILE)

    print("Loading retrieval corpus...")
    corpus = pd.read_csv(CORPUS_FILE)

    print("Loading retrieval embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    print(f"\nGolden queries:     {len(golden):,}")
    print(f"Historical cases:   {len(corpus):,}")
    print(f"Embedding shape:    {embeddings.shape}")

    
   

    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    queries = (
        golden["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("\nCreating golden query embeddings...")

    query_embeddings = model.encode(
        queries,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    print("\nRunning leakage-free retrieval...")

    rows = []

    for query_index, query_embedding in enumerate(query_embeddings):
        golden_row = golden.iloc[query_index]

        # Normalized vectors -> dot product = cosine similarity.
        scores = embeddings @ query_embedding

        
        same_text = (
    corpus["customer_message"].fillna("").astype(str).to_numpy()
    == str(golden_row["text"]))

        scores[same_text] = -np.inf

        top_indices = np.argsort(scores)[::-1][:TOP_K]

        for rank, corpus_index in enumerate(top_indices, start=1):
            historical = corpus.iloc[corpus_index]

            rows.append(
                {
                    "example_id": golden_row["example_id"],
                    "gold_intent": golden_row["gold_intent"],
                    "query": golden_row["text"],
                    "rank": rank,
                    "similarity": float(scores[corpus_index]),
                    "historical_case_id": historical["case_id"],
                    "historical_customer_message": historical[
                        "customer_message"
                    ],
                    "historical_amazon_response": historical[
                        "amazon_response"
                    ],
                }
            )

        if (query_index + 1) % 25 == 0:
            print(
                f"Processed {query_index + 1}/{len(golden)} queries"
            )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    top1 = results[results["rank"] == 1]["similarity"]
    top5 = results["similarity"]

    exact_matches = (
        results["similarity"] >= 0.999999
    ).sum()

    print("\n==============================")
    print("LEAKAGE-FREE RETRIEVAL EVALUATION")
    print("==============================")

    print(f"Queries evaluated: {len(golden):,}")
    print(f"Retrieved results: {len(results):,}")
    print(f"Top-1 mean similarity: {top1.mean():.4f}")
    print(f"Top-1 median similarity: {top1.median():.4f}")
    print(f"Top-5 mean similarity: {top5.mean():.4f}")
    print(f"Exact/self matches remaining: {exact_matches}")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()