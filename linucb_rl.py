"""
train_linucb.py — baselines + a LinUCB contextual bandit router.

    python train_linucb.py

Fill in the three TODOs in the LinUCB class first.
"""
import numpy as np
import pandas as pd

from router_env import RouterEnv

# Edit to your 4 models, ordered small -> large ($ per 1M tokens, simulated).
PRICES = {
    "qwen3:1.7b": 0.05,
    "llama3.2:3b": 0.10,
    "qwen3:4b": 0.15,
    "qwen3:8b": 0.50,
}
LAM_LAT, LAM_COST = 0.1, 0.2      # how much speed and price matter vs. accuracy


class LinUCB:
    """One ridge-regression reward model per arm, plus an optimism bonus for uncertainty."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, reg: float = 1.0):
        self.alpha = alpha
        self.A = np.stack([reg * np.eye(dim) for _ in range(n_arms)])   # (arms, d, d)
        self.b = np.zeros((n_arms, dim))                                 # (arms, d)

    def select(self, x: np.ndarray, explore: bool = True) -> int:
        scores = []
        for a in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[a])        # invert this model's A matrix
            theta = A_inv @ self.b[a]               # its learned weights (ridge regression)
            mean = theta @ x                        # predicted reward for THIS prompt
            if explore:
                bonus = self.alpha * np.sqrt(x @ A_inv @ x)   # how unsure we are
            else:
                bonus = 0.0                         # at test time: no exploring, just the best guess
            score = mean + bonus
            scores.append(score)
        return int(np.argmax(scores))

    def update(self, arm: int, x: np.ndarray, r: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += r * x


def main():
    env = RouterEnv(PRICES, lam_lat=LAM_LAT, lam_cost=LAM_COST)
    train, test = env.split()
    rng = np.random.default_rng(0)
    results = {}

    # ---- baselines, evaluated on the held-out test questions
    for a, m in enumerate(env.models):
        results[f"always {m}"] = env.evaluate(np.full(len(test), a), test)
    results["random"] = env.evaluate(rng.integers(env.n_arms, size=len(test)), test)
    # Oracle = best reward per question in hindsight. It sees the answers, so it's an upper bound.
    results["oracle"] = env.evaluate(env.rewards[test].argmax(axis=1), test)

    # ---- train LinUCB: it only ever sees the reward of the model it picked
    agent = LinUCB(env.n_arms, env.dim, alpha=1.0)
    for epoch in range(5):
        for i in rng.permutation(train):
            x = env.X[i]
            a = agent.select(x)
            agent.update(a, x, env.rewards[i, a])

    # ---- evaluate greedily (no exploration) on questions it never trained on
    choices = [agent.select(env.X[i], explore=False) for i in test]
    results["LinUCB"] = env.evaluate(choices, test)

    table = pd.DataFrame(results).T
    print(table.drop(columns="share").astype(float).round(4))
    print("\nLinUCB routing share:", results["LinUCB"]["share"])
    print("Oracle routing share:", results["oracle"]["share"])


if __name__ == "__main__":
    main()