# LLM Router Using RL Trained Agent
LLM Router using RL trained agent

# Setup

## Ollama models installation
1. Install Ollama using `brew install ollama` (for MacOS)

2. Run Ollama server using `ollama serve`

3. Download the three models used in this project:
   * Qwen3:1.7b : `ollama pull qwen3:1.7b`  -> 1.7B params | 1.5GB
   * Qwen3:4b : `ollama pull qwen3:4b`      -> 4B params | 2.5GB
   * Qwen3:8b : `ollama pull qwen3:8b`      -> 8B params | 4.5GB
  

4. Run a model using `ollama run qwen3:1.7b`. This will run the 1.7B params qwen3 model.

**Note**: Make sure the ollama server is running at all times. To list all ollama models use `ollama list`.


### Benchmark base models

Before creating the RL Agent to route between the downloaded models, I created a benchmark script to evaluate each model based on:
  * `accuracy` (how good the model is)
  * `latency`  (how fast the model is)
  * `output tokens` (how costly the model is)

The evaluation is done using 2 LLM evaluation data sets:
> 1. GSM8K (Grade School Math 8K) -> benchmark dataset of 8.5K high-quality human-written grade-school math word problems designed to evaluate multi-step mathematical reasoning in large language models.
> 2. MMLU (Massive Multitask Language Understanding) -> standard benchmark dataset used to evaluate the knowledge and reasoning capabilities of large language models.

The benchmark can be seen in `results/benchmark.csv`

## Jupyter notebook setup
1. Install jupyter notebook & matplotlib: `pip install jupyter matplotlib`
2. Open jupyter notebook: `jupyter notebook`
