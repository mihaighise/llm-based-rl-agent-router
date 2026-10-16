"""
benchmark.py — build the offline results table for the LLM router.

Runs every question through every model and saves one row per (question, model):
    qid, task, model, correct, latency_s, prompt_tokens, output_tokens, answer, gold, raw_output

Usage:
    pip install ollama datasets pandas tqdm
    python benchmark.py            # full run (N_PER_TASK questions per task)
    python benchmark.py --quick    # 5 questions per task, to test the pipeline

The script is resumable: if it stops, run it again and it skips rows already saved.
"""

import argparse
import os
import random
import re
import time

import ollama
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


MODELS = ["qwen3:1.7b", "qwen3:4b", "qwen3:8b"]

N_PER_TASK = 30          # 100 GSM8K + 100 MMLU = 200 questions per model
SEED = 42
OUT_PATH = "results/benchmark.csv"

SYSTEM_PROMPTS = {
    "gsm8k": (
        "Solve the math problem. Think step by step but keep it brief. "
        "End your reply with a final line of the form 'Answer: <number>'."
    ),
    "mmlu": "Answer the multiple-choice question with only the letter A, B, C, or D. No explanation.",
}
MAX_TOKENS = {"gsm8k": 1024, "mmlu": 512}   # caps output length (and runaway latency)


# ---------------------------------------------------------------- data
def load_questions(n_per_task: int) -> list[dict]:
    rng = random.Random(SEED)
    questions = []

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for i in rng.sample(range(len(gsm)), n_per_task):
        row = gsm[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        questions.append({"qid": f"gsm8k-{i}", "task": "gsm8k",
                          "prompt": row["question"], "gold": gold})

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    for i in rng.sample(range(len(mmlu)), n_per_task):
        row = mmlu[i]
        options = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", row["choices"]))
        prompt = f"{row['question']}\n\n{options}"
        questions.append({"qid": f"mmlu-{i}", "task": "mmlu", "prompt": prompt,
                          "gold": "ABCD"[row["answer"]], "subject": row["subject"]})

    return questions


# ---------------------------------------------------------------- grading
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(text: str) -> str | None:
    text = text.replace(",", "").replace("$", "")
    m = re.search(r"Answer:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = NUM_RE.findall(text)          # fallback: last number in the reply
    return nums[-1] if nums else None


def extract_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def grade(task: str, output: str, gold: str) -> tuple[str | None, bool]:
    if task == "gsm8k":
        ans = extract_number(output)
        try:
            return ans, ans is not None and abs(float(ans) - float(gold)) < 1e-6
        except ValueError:
            return ans, False
    ans = extract_letter(output)
    return ans, ans == gold


# ---------------------------------------------------------------- model calls
def ask(model: str, task: str, prompt: str) -> dict:
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPTS[task]},
                     {"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": MAX_TOKENS[task]},
    }
    if model.startswith("qwen3"):
        kwargs["think"] = False          # disable Qwen3's reasoning mode

    start = time.perf_counter()
    resp = ollama.chat(**kwargs)
    latency = time.perf_counter() - start

    return {"raw_output": resp["message"]["content"],
            "latency_s": round(latency, 4),
            "prompt_tokens": resp.get("prompt_eval_count"),
            "output_tokens": resp.get("eval_count")}


def warm_up(model: str) -> None:
    ollama.chat(model=model, messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1})


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5 questions per task")
    args = parser.parse_args()

    n = 5 if args.quick else N_PER_TASK
    out_path = OUT_PATH.replace(".csv", "_quick.csv") if args.quick else OUT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    questions = load_questions(n)
    print(f"Loaded {len(questions)} questions.")

    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(zip(prev["qid"], prev["model"]))
        print(f"Resuming: {len(done)} rows already saved.")

    # Model-major loop: only one model in memory at a time.
    for model in MODELS:
        todo = [q for q in questions if (q["qid"], model) not in done]
        if not todo:
            print(f"{model}: already done.")
            continue

        print(f"\n{model}: warming up...")
        warm_up(model)

        for q in tqdm(todo, desc=model):
            result = ask(model, q["task"], q["prompt"])
            answer, correct = grade(q["task"], result["raw_output"], q["gold"])
            row = {"qid": q["qid"], "task": q["task"], "subject": q.get("subject", ""),
                   "model": model, "correct": int(correct),
                   "latency_s": result["latency_s"],
                   "prompt_tokens": result["prompt_tokens"],
                   "output_tokens": result["output_tokens"],
                   "answer": answer, "gold": q["gold"],
                   "prompt": q["prompt"], "raw_output": result["raw_output"]}
            # Append each row immediately so a crash loses nothing.
            pd.DataFrame([row]).to_csv(out_path, mode="a", index=False,
                                       header=not os.path.exists(out_path))

    # Quick summary
    df = pd.read_csv(out_path)
    summary = df.groupby(["model", "task"]).agg(
        accuracy=("correct", "mean"),
        avg_latency_s=("latency_s", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        n=("correct", "size"),
    ).round(3)
    print("\n", summary)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
