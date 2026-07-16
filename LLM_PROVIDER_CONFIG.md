# LLM Provider Configuration

This system uses a **tiered fallback** for LLM inference:

1. **Primary**: [Ollama Load Balancer](https://github.com/jmorganca/ollama-load-balancer)
   - Default: `http://127.0.0.1:5050`
   - Set via `OLLAMA_LB_URL` (env var).

2. **Fallback**: [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI-compatible)
   - Default: `http://127.0.0.1:4000/v1`
   - Automatically attempted when Ollama LB fails.
   - Set API key via `LITELLM_API_KEY` (if required).

3. **Final Fallback**: Direct Ollama
   - Default: `http://127.0.0.1:11434`
   - Set via `OLLAMA_URL` (env var).

---

## Environment Variables

```bash
# Ollama LB (Primary)
export OLLAMA_LB_URL="http://your-lb-server:5050"
export OLLAMA_STREAM_PORT=11434  # Port for streaming from GPU nodes

# LiteLLM (Fallback)
export LITELLM_BASE_URL="http://your-litellm-server:4000/v1"
export LITELLM_API_KEY="your-api-key"  # Optional

# Direct Ollama (Final Fallback)
export OLLAMA_URL="http://localhost:11434"
```

---

## Behavior

- The system **automatically fails over** if a provider is unreachable.
- Logs warnings when falling back (e.g., `[OLLAMA-LB] Failed, falling back to LiteLLM`).
- Priority order: **Ollama LB → LiteLLM → Direct Ollama**.

---

## Setup Instructions

### 1. Ollama Load Balancer (Primary)

Run the Ollama LB server:
```bash
# Clone and run the load balancer (see: https://github.com/jmorganca/ollama-load-balancer)
git clone https://github.com/jmorganca/ollama-load-balancer.git
cd ollama-load-balancer
python olb.py --port 5050
```

Ensure GPU nodes are registered and running Ollama.

---

### 2. LiteLLM (Fallback)

Install and run LiteLLM:
```bash
pip install litellm
litellm --model ollama/mistral:7b-instruct --port 4000
```

Set the environment variables if using a custom URL or API key.

---

### 3. Direct Ollama (Final Fallback)

Run Ollama locally:
```bash
ollama serve
```

Pull the model:
```bash
ollama pull mistral:7b-instruct
```

---

## Disabling Fallbacks

To **disable LiteLLM fallback** and only use Ollama LB + Direct Ollama, set LiteLLM to an unreachable URL:
```bash
export LITELLM_BASE_URL="http://127.0.0.1:1/v1"
```

To **disable all fallbacks** and only use Ollama LB:
```bash
export LITELLM_BASE_URL="http://127.0.0.1:1/v1"
export OLLAMA_URL=""  # Disable direct Ollama
```
