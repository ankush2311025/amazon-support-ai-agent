from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_corpus.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_embeddings.npy"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    texts = (
        df["customer_message"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"Historical cases: {len(texts):,}")

    print(f"\nLoading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("\nCreating retrieval embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()