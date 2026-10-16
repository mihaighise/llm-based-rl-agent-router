# LLM Router Using RL Trained Agent
LLM Router using RL trained agent

# Setup

### Ollama models installation
1. Install Ollama using `brew install ollama` (for MacOS)

2. Run Ollama server using `ollama serve`

3. Download the three models used in this project:
   * Qwen3:1.7b : `ollama pull qwen3:1.7b`  -> 1.7B params | 1.5GB
   * Qwen3:4b : `ollama pull qwen3:4b`      -> 4B params | 2.5GB
   * Qwen3:8b : `ollama pull qwen3:8b`      -> 8B params | 4.5GB
  

4. Run a model using `ollama run qwen3:1.7b`. This will run the 1.7B params qwen3 model.

**Note**: Make sure the ollama server is running at all times.
