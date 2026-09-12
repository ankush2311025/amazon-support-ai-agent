from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weak_training_set.csv"
)

OUTPUT_EMBEDDINGS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_embeddings.npy"
)


MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print(f"Reading training data: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"Training examples: {len(texts):,}")

    print(f"\nLoading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("\nCreating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings)

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings
    )

    print(
        f"Saved to: {OUTPUT_EMBEDDINGS}"
    )


if __name__ == "__main__":
    main()