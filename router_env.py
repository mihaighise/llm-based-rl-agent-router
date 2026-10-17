"""
router_env.py — offline environment for the LLM router.

Every (question, model) result is already in benchmark.csv, so "asking a model"
is just a table lookup: training is instant and costs nothing.

    env = RouterEnv(PRICES, lam_lat=0.1, lam_cost=0.2)
    env.X[i]           context for question i (embedding + bias term)
    env.rewards[i, a]  reward if model a answers question i
"""
import numpy as np
import pandas as pd


class RouterEnv:
    def __init__(self, prices: dict, lam_lat: float = 0.0, lam_cost: float = 0.0,
                 results_csv="results/benchmark.csv",
                 emb_path="results/embeddings.npy",
                 index_csv="results/prompt_index.csv"):
        df = pd.read_csv(results_csv)
        index = pd.read_csv(index_csv)

        self.models = list(prices)               # arm order = order of `prices` (small -> large)
        self.qids = index["qid"].to_numpy()
        self.tasks = index["task"].to_numpy()

        # Context = prompt embedding + a constant 1, so each linear model gets an intercept.
        # X.shape = (no_questions, embedding_size)
        X = np.load(emb_path)
        self.X = np.hstack([X, np.ones((len(X), 1))])
        # X.shape = (no_questions, embedding_size + 1)

        # Calculate cost of tokens used
        df["cost"] = (df["prompt_tokens"] + df["output_tokens"]) * df["model"].map(prices) / 1e6

        def table(col):  # shape (n_questions, n_models), rows aligned with the embeddings
            return (df.pivot(index="qid", columns="model", values=col)
                      .loc[self.qids, self.models].to_numpy(dtype=float))

        # Extract values from resulted df (using table)
        self.correct = table("correct")
        self.latency = table("latency_s")
        self.cost = table("cost")

        # Normalise so the slowest / most expensive model averages 1.0.
        # Then lam_lat = 0.1 means: "the slowest model's typical latency is worth 0.1 accuracy".
        lat_n = self.latency / self.latency.mean(axis=0).max()
        cost_n = self.cost / self.cost.mean(axis=0).max()
        self.rewards = self.correct - lam_lat * lat_n - lam_cost * cost_n

    @property
    def n_arms(self) -> int:
        return len(self.models)

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def split(self, test_frac=0.2, seed=0):
        """Train/test split of question indices, stratified by task."""
        rng = np.random.default_rng(seed)
        train, test = [], []
        for t in np.unique(self.tasks):
            idx = rng.permutation(np.where(self.tasks == t)[0])
            k = int(len(idx) * test_frac)
            test += list(idx[:k])
            train += list(idx[k:])
        return np.array(train), np.array(test)

    def evaluate(self, choices, idx) -> dict:
        """choices[j] = arm picked for question idx[j]. Returns summary metrics."""
        arms = np.asarray(choices)
        return {
            "reward": self.rewards[idx, arms].mean(),           # average reward for all questions based on chosen model
            "accuracy": self.correct[idx, arms].mean(),         # average accuracy for all questions based on chosen model
            "avg_cost": self.cost[idx, arms].mean(),            # average cost for all questions based on chosen model
            "avg_latency_s": self.latency[idx, arms].mean(),    # average latency for all questions based on chosen model
            # dict which shows much how much in % each model was used
            "share": {m: round(float(np.mean(arms == a)), 2) for a, m in enumerate(self.models)},   
        }