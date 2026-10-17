"""
embed.py — turn every benchmark prompt into a vector (the router's input).

Run once, after the benchmark has finished:
    pip install sentence-transformers
    python embed.py

Outputs:
    results/embeddings.npy     shape (n_questions, 384), one row per question
    results/prompt_index.csv   qid + task for each row, in the same order
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("results/benchmark_quick.csv")

# One row per question (the benchmark CSV has one row per question x model).
prompts = df.drop_duplicates("qid")[["qid", "task", "prompt"]].reset_index(drop=True)

# Small, fast sentence encoder (~90 MB download the first time). Uses the Mac GPU automatically.
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X = encoder.encode(prompts["prompt"].tolist(), batch_size=64,
                   show_progress_bar=True, normalize_embeddings=True)

np.save("results/embeddings.npy", X)
prompts[["qid", "task"]].to_csv("results/prompt_index.csv", index=False)
print(f"Saved {X.shape[0]} embeddings of size {X.shape[1]}")