# WON backend (AI chat)

FastAPI service for the WON UI's AI assist. One endpoint: `POST /api/ai/chat`.
It gathers the inputs a WON needs, reports what is still missing, and returns the draft.
Rules for "missing" are in `ai_chat.py` and mirror `src/lib/wonDraft.js` in the UI project.

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # pick ONE provider inside (Ollama is the free local default)
uvicorn main:app --reload --port 8000 --env-file .env
```

Check http://localhost:8000/docs. The UI (separate project) proxies `/api` to port 8000 in dev;
set `VITE_AI_MOCK=false` in the UI's `.env`.

## Providers

Set in `.env`: Ollama (free, local), Groq or Gemini free tiers, any OpenAI-compatible server,
or Anthropic (`WON_AI_PROVIDER=anthropic`, `pip install anthropic`).

## Before production

- Add authentication and per-user rate limiting (the AI route costs money or compute).
- Restrict CORS/origins if the UI is served from another domain.

## Deploy (demo)

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips="*"`
- Environment variables (set in the host dashboard, never commit keys):
  `WON_AI_PROVIDER=openai`, `WON_AI_BASE_URL`, `WON_AI_API_KEY`, `WON_AI_MODEL`
  (Groq or Gemini values are in `.env.example`), and `WON_ALLOWED_ORIGINS=https://<your-ui-url>`.
- Health check path: `/health`
- `guard.py` limits each IP per minute and caps total requests per day to protect the free quota.
