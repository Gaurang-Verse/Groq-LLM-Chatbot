# Groq LLM Chatbot

A multi-model streaming chat application built with Streamlit and the Groq API. Switch between models at runtime and get responses rendered token-by-token via Groq's LPU inference engine.

## Features

- **Streaming responses** — tokens render progressively as they're generated
- **Multi-model support** — switch models from a sidebar dropdown without restarting:
- GPT-OSS 120B (`openai/gpt-oss-120b`)
- GPT-OSS 20B (`openai/gpt-oss-20b`)
- **Persistent conversation state** — full multi-turn history is sent with each request
- **Secure key handling** — API key loaded from environment variables, never hardcoded; app fails with a clear error if the key is missing
- **Basic error handling** — Groq API errors (auth, rate limits, connection issues) are caught and shown in the UI instead of crashing the app

## Architecture

The app is organized into three classes with separated responsibilities:

| Class | Responsibility |
|---|---|
| `GroqAPI` | Owns the Groq client, reads `GROQ_API_KEY` from environment, exposes a generator that streams response chunks |
| `Message` | Manages `st.session_state.messages`, renders chat history, streams assistant output via `st.write_stream` |
| `ModelSelector` | Renders the sidebar dropdown for choosing the active model |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Gaurang-Verse/Groq-LLM-Chatbot.git
cd Groq-LLM-Chatbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Copy `.env.example` to `.env` and add your key (get a free one at [console.groq.com/keys](https://console.groq.com/keys)):
```bash
cp .env.example .env
```

**4. Run the app**
```bash
streamlit run main.py
```

## Tech Stack

- **Frontend:** Streamlit
- **LLM Inference:** Groq API (LPU inference engine)
- **Config:** python-dotenv

## Benchmark Results

Measured locally with `benchmark.py` (3 prompts × 3 runs each, streamed):

| Model | Tokens/sec | Time to First Token | Notes |
|---|---|---|---|
| `openai/gpt-oss-120b` | 95.1 | 0.337s | 0 empty responses across 9 runs |
| `openai/gpt-oss-20b` | 49.5 | 0.323s | 3/9 runs returned no content — likely reasoning tokens consuming the `max_tokens` budget before visible output; increasing `max_tokens` reduces this |

## Known Limitations

- Not currently deployed to a public URL — [update once your Streamlit Cloud link is live]
- Model list should be checked against [console.groq.com/docs/models](https://console.groq.com/docs/models) periodically — Groq deprecates models on a rolling basis


## License

Apache License 2.0 — see [LICENSE](LICENSE).