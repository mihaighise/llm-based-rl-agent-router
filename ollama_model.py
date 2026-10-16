import ollama, time

MODEL = "qwen3:1.7b"

# warm-up call (not timed)
ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}], think=False)

start = time.time()
resp = ollama.chat(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Answer with only the final number. No explanation."},
        {"role": "user", "content": "What is 9 + 10?"},
    ],
    think=False,
    options={"temperature": 0},
)

print(resp["message"]["content"])
print(f"{time.time() - start:.2f}s")