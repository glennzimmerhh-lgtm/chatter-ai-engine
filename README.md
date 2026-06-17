# Chatter-AI Engine (standalone)

A separate service that learns from your CRM chat history and proposes replies.
It runs **next to** the CRM (own Railway service), reads the same Postgres, and
exposes a small API. Later the CRM just calls `/draft` — no risk to the live app.

## What it does
- **`/ingest`** — embeds your whole message history into `pgvector` ("reads & understands" all chats).
- **`/draft`** — for a given fan: pulls recent history + your gold-standard examples + similar past replies + profile → **gpt-4o-mini** → one suggested reply.
- **`/teach`** — you give it an ideal answer to a situation (you train it).
- **`/feedback`** — you approve/correct a draft; "good" ones become gold examples it follows.
- **`/examples`** — see how much it has been trained.

The more you `/teach` and `/feedback`, the closer it gets to your best chatters.
Those gold examples are also your future **fine-tuning dataset**.

## Deploy (Railway — new service in the same project)
1. New service → deploy this folder.
2. Set env vars:
   - `DATABASE_URL` → the **same** Postgres URL the CRM uses (Variables → reference it).
   - `OPENAI_API_KEY` → your OpenAI key.
   - `API_TOKEN` → pick any secret string (protects the endpoints). Optional but recommended.
   - `AI_MODEL` → `gpt-4o-mini` (default).
   - `PERSONA` → optional; otherwise edit the default in `main.py`.
3. Needs `pgvector`. Railway's Postgres supports it; the service runs
   `CREATE EXTENSION IF NOT EXISTS vector` on startup. If that fails, enable it once manually.

## First run
```
# health
GET  /health
# embed history (call repeatedly until "remaining": 0)
POST /ingest?limit=2000      (Header: Authorization: Bearer <API_TOKEN>)
# get a suggestion for a fan
GET  /draft?tg_id=123456789
# teach it an ideal reply
POST /teach   {"incoming":"how much for a custom video?","ideal_reply":"hey babe 🙈 customs start at 50€ ..."}
# approve/correct a draft
POST /feedback {"incoming":"...","final_reply":"the right answer","rating":"good"}
```

## Wire into the CRM (later, tiny change)
In the chat UI, point the "AI Suggestions" button at `GET <engine-url>/draft?tg_id=<currentChatTgId>`
and show the `suggestion`. Add an "approve / edit & send" flow that POSTs to `/feedback`.
That gives you the human-in-the-loop trainer. Autonomous sending is a later flag.

## Roadmap on top of this
1. **Now:** ingest + draft + teach/feedback (this service).
2. **Autonomous send:** a worker that drafts on each incoming message → guardrails →
   human-like delay → sends via the CRM userbot. Kill-switch + per-fan on/off.
3. **Voice calls:** see below.
4. **OnlyFans:** an importer that pulls CreatorHero/Infloww chats into the same tables → same pipeline.

## Voice calls (AI calling subs) — options
Your CRM already plays **pre-recorded** audio into Telegram calls (pytgcalls). Three levels:
- **Easy/cheap:** AI decides *when* to trigger a pre-recorded call, and picks which clip.
- **Medium:** AI writes a script → **TTS** (text-to-speech, optionally a cloned voice) → play that audio into the call. Dynamic but still one-way.
- **Hard/expensive:** real-time voice agent (speech-to-text → LLM → TTS, live two-way). Needs a streaming voice stack and careful latency/cost handling; highest ban/quality risk.

Recommended path: start with **trigger + TTS clips**, then evaluate real-time later.
