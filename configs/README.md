# Configuration profiles

All configuration is read from environment variables (prefix `MRAG_`) via
`src/config.py`. The files here are ready-made profiles you can copy to `.env`:

| Profile | Use when |
| --- | --- |
| `default.env` | You have a GPU (or patience) and can download Whisper / Qwen-VL / MiniLM weights from Hugging Face. |
| `openai_compatible.env` | Qwen-VL is served elsewhere (vLLM, Ollama, DashScope) through an OpenAI-style `/chat/completions` endpoint. Whisper + embeddings still run locally. |
| `offline.env` | No model weights can be downloaded (air-gapped / CI). Uses the hashing embedder, metadata-only vision and the extractive answer fallback. Everything runs, but **no neural model is involved** and the UI says so. |

```bash
cp configs/default.env .env   # then edit as needed
```
