# Voice-Driven Customer Support Agent

A FastAPI WebSocket gateway runs a typed LangGraph pipeline: intent classification,
rolling sentiment, RAG answer drafting, a hard grounding gate, and deterministic
human escalation. The Next.js client shows live transcript, sentiment, and handoff
events.

## Setup

1. Copy `.env.example` to `.env`, then fill `OLLAMA_API_KEY` and set a long
   `API_AUTH_SECRET`. Do not commit `.env`.
2. Create a Python 3.12 virtual environment and run `pip install -r requirements.txt`.
3. Run `python -m app.rag.ingest` to build the persistent Chroma index, then run
   `uvicorn app.main:app --reload`.
4. In `frontend`, copy `.env.local.example` to `.env.local`, run `npm install`, then
   `npm run dev`.

Alternatively, after creating `.env`, run `docker-compose up --build`.

The browser starts a short-lived signed session token before connecting to the
authenticated `ws://localhost:8000/ws/audio/{session_id}` gateway. Production
deployments must use `wss://`/TLS. Raw audio is processed in memory and is never
persisted; only redacted state snapshots are stored.

## Validation

Backend: `ruff check app/ tests/`, `black --check app/ tests/`, `mypy app/`,
`pytest --cov=app --cov-report=term-missing`, `bandit -r app/ -ll`, and
`pip-audit -r requirements.txt`.

Frontend: `npm run lint`, `npm run typecheck`, `npm run test`, `npm run build`, and
`npm audit --omit=dev`.
