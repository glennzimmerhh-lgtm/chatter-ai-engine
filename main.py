"""
Chatter-AI Engine — standalone service
======================================
Reads your existing CRM Postgres (messages / conversations / sales), embeds the
full chat history into pgvector, and produces context-aware reply suggestions
with gpt-4o-mini (RAG: recent history + similar past messages + fan profile).

This runs SEPARATELY from the CRM. Nothing here writes to your data except the
`message_embeddings` table it creates. Wire it into the CRM later by calling
GET /draft?tg_id=... from the chat UI.

ENV VARS (set these in Railway / your host — never in code):
  DATABASE_URL     Postgres connection string (same DB as the CRM)   [required]
  OPENAI_API_KEY   OpenAI key                                         [required]
  AI_MODEL         chat model        (default: gpt-4o-mini)
  EMBED_MODEL      embedding model   (default: text-embedding-3-small)
  API_TOKEN        optional bearer token to protect the endpoints
  PERSONA          system prompt / persona (edit to taste)
"""
from __future__ import annotations
import os
import json
import time
from typing import Optional, List

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # text-embedding-3-small
API_TOKEN = os.environ.get("API_TOKEN", "")

# ── Persona / system prompt — EDIT THIS to match your top chatter's style ──────
PERSONA = os.environ.get("PERSONA", (
    "You are an expert OnlyFans/Telegram chatter writing on behalf of the creator. "
    "Goals: build rapport, keep the fan engaged, and naturally move them toward a "
    "purchase (PPV content, calls, custom content) using the price list. "
    "Match the tone and language of the conversation (mirror the fan's language). "
    "Keep replies short, personal and human — never robotic. Use the fan's history "
    "and what they bought before. Do NOT invent facts. If the fan asks for a refund, "
    "is upset, or anything risky/sensitive, reply with [[HANDOFF]] so a human takes over."
))

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Chatter-AI Engine")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "*"],
)


def db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _auth(authorization: Optional[str]):
    if not API_TOKEN:
        return
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Unauthorized")


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _get_persona() -> str:
    """Editable persona/system prompt (DB), falling back to the PERSONA env/default."""
    try:
        with db() as conn, conn.cursor() as c:
            c.execute("SELECT value FROM ai_config WHERE key='persona'")
            row = c.fetchone()
            if row and (row["value"] or "").strip():
                return row["value"]
    except Exception:
        pass
    return PERSONA


def _get_knowledge() -> list:
    try:
        with db() as conn, conn.cursor() as c:
            c.execute("SELECT id, content FROM knowledge ORDER BY id")
            return c.fetchall()
    except Exception:
        return []


def _get_price_list() -> str:
    """The authoritative price list (DB, ai_config key='price_list'). '' if unset."""
    try:
        with db() as conn, conn.cursor() as c:
            c.execute("SELECT value FROM ai_config WHERE key='price_list'")
            row = c.fetchone()
            if row and (row["value"] or "").strip():
                return row["value"]
    except Exception:
        pass
    return ""


def _get_payment_terms() -> str:
    """Payment conditions/terms (DB, ai_config key='payment_terms'). '' if unset."""
    try:
        with db() as conn, conn.cursor() as c:
            c.execute("SELECT value FROM ai_config WHERE key='payment_terms'")
            row = c.fetchone()
            if row and (row["value"] or "").strip():
                return row["value"]
    except Exception:
        pass
    return ""


# Hard anti-hallucination rule — prices were being invented; this stops it.
PRICE_RULE = (
    "PRICE RULE (ABSOLUTE, NON-NEGOTIABLE): State prices ONLY exactly as written in the "
    "PRICE LIST below — verbatim, same number, same currency. NEVER invent, estimate, "
    "round, convert, or guess a price. If a fan asks about something not in the PRICE LIST, "
    "do NOT make up a number — say you'll quickly check. The PRICE LIST is the single source "
    "of truth and overrides anything you might assume.\n"
    "SENDING A PRICE LIST: When a fan asks for prices or a price list for a category "
    "(calls/cam, sexchat, or PPV content), output that category's ENTIRE block EXACTLY as "
    "written in the PRICE LIST — every single line, in full, INCLUDING the payment methods "
    "(Zahlungsmethoden) and the fake-check note underneath it. Do NOT shorten it, do NOT "
    "summarise, do NOT drop the payment methods or the fake-check, and do NOT replace it with "
    "a generic line like 'sag einfach Bescheid'. Copy the whole category block 1:1. You may "
    "add one short, natural sentence before or after, but the block itself stays complete and "
    "unchanged."
)


def _grounding_block(know_txt: str = None, include_price: bool = True) -> str:
    """Consistent FACTS + PRICE LIST + price rule block, injected into every generation path."""
    if know_txt is None:
        know = _get_knowledge()
        know_txt = "\n".join(f"- {k['content']}" for k in know)
    parts = ["FACTS & RULES you must always follow:\n" + (know_txt or "(none yet)")]
    if include_price:
        pl = _get_price_list()
        parts.append("PRICE LIST (the ONLY valid prices — quote verbatim):\n" + (pl or "(none set yet)"))
        pt = _get_payment_terms()
        if pt:
            parts.append("PAYMENT TERMS (Zahlungsbedingungen — always follow & communicate these):\n" + pt)
        parts.append(PRICE_RULE)
    return "\n\n".join(parts)


# ── SETUP ─────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def setup():
    try:
        with db() as conn, conn.cursor() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    msg_id    BIGINT PRIMARY KEY,
                    tg_id     TEXT,
                    direction TEXT,
                    chatter   TEXT,
                    content   TEXT,
                    embedding vector({EMBED_DIM})
                )
            """)
            # Your coaching: ideal replies you provide / approve. These are the
            # gold-standard the AI is trained on (and your future fine-tune dataset).
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS training_examples (
                    id          SERIAL PRIMARY KEY,
                    incoming    TEXT,
                    ideal_reply TEXT,
                    tags        TEXT DEFAULT '',
                    rating      TEXT DEFAULT 'good',
                    source      TEXT DEFAULT 'manual',
                    created_at  TIMESTAMPTZ DEFAULT now(),
                    embedding   vector({EMBED_DIM})
                )
            """)
            # Editable persona/system-prompt + general knowledge/rules (not tied to chats)
            c.execute("CREATE TABLE IF NOT EXISTS ai_config (key TEXT PRIMARY KEY, value TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS knowledge (id SERIAL PRIMARY KEY, content TEXT, created_at TIMESTAMPTZ DEFAULT now())")
            conn.commit()
        print("✅ message_embeddings ready (pgvector)")
    except Exception as e:
        print(f"⚠️ setup warning: {e} — make sure pgvector is available on your Postgres")


@app.get("/health")
def health():
    try:
        with db() as conn, conn.cursor() as c:
            c.execute("SELECT COUNT(*) AS n FROM messages")
            total = c.fetchone()["n"]
            c.execute("SELECT COUNT(*) AS n FROM message_embeddings")
            embedded = c.fetchone()["n"]
        return {"status": "ok", "model": AI_MODEL, "messages": total, "embedded": embedded}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── INGEST: embed all messages into pgvector ──────────────────────────────────
@app.post("/ingest")
def ingest(limit: int = Query(2000, description="max messages to embed this call"),
           authorization: Optional[str] = Header(None)):
    _auth(authorization)
    embedded = 0
    with db() as conn, conn.cursor() as c:
        # messages not yet embedded (skip empty / media placeholders)
        c.execute("""
            SELECT m.id, m.tg_id, m.direction, m.chatter, m.text
            FROM messages m
            LEFT JOIN message_embeddings e ON e.msg_id = m.id
            WHERE e.msg_id IS NULL
              AND m.text IS NOT NULL AND length(trim(m.text)) > 1
              AND m.text NOT LIKE '[%%]'
            ORDER BY m.id
            LIMIT %s
        """, (limit,))
        rows = c.fetchall()
        # embed in batches of 100
        for i in range(0, len(rows), 100):
            batch = rows[i:i + 100]
            vecs = _embed([r["text"][:2000] for r in batch])
            args = []
            for r, v in zip(batch, vecs):
                args.append((r["id"], r["tg_id"], r["direction"], r["chatter"] or "",
                             r["text"][:4000], _vec_literal(v)))
            psycopg2.extras.execute_values(
                c,
                "INSERT INTO message_embeddings (msg_id,tg_id,direction,chatter,content,embedding) "
                "VALUES %s ON CONFLICT (msg_id) DO NOTHING",
                args, template="(%s,%s,%s,%s,%s,%s::vector)"
            )
            conn.commit()
            embedded += len(batch)
    remaining = _remaining()
    return {"embedded_now": embedded, "remaining": remaining,
            "done": remaining == 0, "hint": "call /ingest again until remaining = 0"}


def _remaining() -> int:
    with db() as conn, conn.cursor() as c:
        c.execute("""
            SELECT COUNT(*) AS n FROM messages m
            LEFT JOIN message_embeddings e ON e.msg_id = m.id
            WHERE e.msg_id IS NULL AND m.text IS NOT NULL
              AND length(trim(m.text)) > 1 AND m.text NOT LIKE '[%%]'
        """)
        return c.fetchone()["n"]


# ── DRAFT: context-aware reply suggestion ─────────────────────────────────────
class DraftOut(BaseModel):
    suggestion: str
    handoff: bool
    used_gold: int
    used_examples: int


@app.get("/draft", response_model=DraftOut)
def draft(tg_id: str,
          incoming: Optional[str] = Query(None, description="override: the fan's latest message"),
          authorization: Optional[str] = Header(None)):
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        # fan profile
        c.execute("""SELECT internal_name, anon_id, notes, funnel_stage, tg_username
                     FROM conversations WHERE tg_id=%s""", (tg_id,))
        prof = c.fetchone() or {}
        # recent history (last 25, chronological)
        c.execute("""SELECT direction, chatter, text, timestamp FROM messages
                     WHERE tg_id=%s ORDER BY id DESC LIMIT 25""", (tg_id,))
        hist = list(reversed(c.fetchall()))
        # what the fan bought
        c.execute("""SELECT product, amount FROM sales WHERE tg_id=%s ORDER BY id DESC LIMIT 10""", (tg_id,))
        sales = c.fetchall()

    latest_in = incoming or next((m["text"] for m in reversed(hist) if m["direction"] == "in"), "")
    if not latest_in:
        raise HTTPException(400, "no incoming message to respond to")

    # RAG retrieval — your GOLD-STANDARD coaching first, then general style refs
    examples = []
    gold = []
    try:
        qvec = _vec_literal(_embed([latest_in[:2000]])[0])
        with db() as conn, conn.cursor() as c:
            c.execute("""SELECT incoming, ideal_reply FROM training_examples
                         WHERE rating='good'
                         ORDER BY embedding <=> %s::vector LIMIT 5""", (qvec,))
            gold = c.fetchall()
            c.execute("""SELECT content FROM message_embeddings
                         WHERE direction='out' AND length(content) > 8
                         ORDER BY embedding <=> %s::vector LIMIT 6""", (qvec,))
            examples = [r["content"] for r in c.fetchall()]
    except Exception as e:
        print(f"RAG retrieve skipped: {e}")

    # Build prompt
    bought = ", ".join(f"{s['product']} ({s['amount']}€)" for s in sales) or "nothing yet"
    prof_txt = (
        f"Fan: {prof.get('internal_name') or prof.get('anon_id') or tg_id}\n"
        f"Funnel stage: {prof.get('funnel_stage') or 'unknown'}\n"
        f"Notes: {prof.get('notes') or '-'}\n"
        f"Already bought: {bought}\n"
    )
    convo_txt = "\n".join(
        f"{'FAN' if m['direction']=='in' else 'YOU'}: {m['text']}" for m in hist
    )
    style_txt = "\n".join(f"- {ex}" for ex in examples)
    gold_txt = "\n".join(f"FAN: {g['incoming']}\nIDEAL REPLY: {g['ideal_reply']}" for g in gold)

    know = _get_knowledge()
    know_txt = "\n".join(f"- {k['content']}" for k in know)
    sys = (
        _get_persona() + "\n\n"
        + _grounding_block(know_txt) + "\n\n"
        "GOLD-STANDARD examples the operator trained you on — follow this approach and tone closely:\n"
        + (gold_txt or "(none yet — operator is still training)") + "\n\n"
        "Additional style references from past chats (do not copy verbatim):\n"
        + (style_txt or "(none)") + "\n\n"
        "Fan profile:\n" + prof_txt
    )
    user = (
        "Recent conversation (YOU = the chatter, FAN = the subscriber):\n"
        + convo_txt + "\n\n"
        "Write the single best next reply as YOU. Keep it short and natural, "
        "in the fan's language. Output ONLY the reply text."
    )

    chat = client.chat.completions.create(
        model=AI_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.8, max_tokens=700,
    )
    out = (chat.choices[0].message.content or "").strip()
    handoff = "[[HANDOFF]]" in out
    out = out.replace("[[HANDOFF]]", "").strip()
    return DraftOut(suggestion=out, handoff=handoff, used_gold=len(gold), used_examples=len(examples))


# ── ACT: autonomous reply + actions (function-calling brain) ──────────────────
# The engine DECIDES (reply text + structured actions). The worker EXECUTES them
# against its existing endpoints, behind the master switch + guardrails.
class ActIn(BaseModel):
    tg_id: str
    incoming: Optional[str] = None
    available_ppv: Optional[List[str]] = None     # vault filenames the AI may send
    available_calls: Optional[List[str]] = None   # "folder/filename" recordings it may play
    price_list: Optional[str] = ""
    payment_confirmed: bool = False               # worker tells us if a payment is on file

class ActOut(BaseModel):
    reply: str
    actions: list
    handoff: bool

ACT_TOOLS = [
    {"type": "function", "function": {
        "name": "send_ppv",
        "description": "Send a media/PPV file from the vault to the fan. ONLY use a filename from the provided available list. Paid content must NOT be sent without a confirmed payment.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "exact filename from the available list"},
            "caption": {"type": "string", "description": "short caption, optional"}},
            "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "set_funnel_stage",
        "description": "Update the fan's sales funnel stage based on the conversation.",
        "parameters": {"type": "object", "properties": {
            "stage": {"type": "string", "enum": ["kalt", "warm", "hot", "angebot", "gebucht", "done"]}},
            "required": ["stage"]}}},
    {"type": "function", "function": {
        "name": "start_call",
        "description": "Start a pre-recorded call to the fan. 'fake_checks' = free warm-up check; 'paid_calls' ONLY after a confirmed payment. Use a filename from the provided available list.",
        "parameters": {"type": "object", "properties": {
            "folder": {"type": "string", "enum": ["fake_checks", "paid_calls"]},
            "filename": {"type": "string", "description": "exact filename from the available list"}},
            "required": ["folder", "filename"]}}},
]

@app.post("/act", response_model=ActOut)
def act(body: ActIn, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    import json
    tg_id = body.tg_id
    with db() as conn, conn.cursor() as c:
        c.execute("""SELECT internal_name, anon_id, notes, funnel_stage, tg_username
                     FROM conversations WHERE tg_id=%s""", (tg_id,))
        prof = c.fetchone() or {}
        c.execute("""SELECT direction, chatter, text, timestamp FROM messages
                     WHERE tg_id=%s ORDER BY id DESC LIMIT 25""", (tg_id,))
        hist = list(reversed(c.fetchall()))
        c.execute("""SELECT product, amount FROM sales WHERE tg_id=%s ORDER BY id DESC LIMIT 10""", (tg_id,))
        sales = c.fetchall()

    latest_in = body.incoming or next((m["text"] for m in reversed(hist) if m["direction"] == "in"), "")
    if not latest_in:
        raise HTTPException(400, "no incoming message to respond to")

    examples, gold = [], []
    try:
        qvec = _vec_literal(_embed([latest_in[:2000]])[0])
        with db() as conn, conn.cursor() as c:
            c.execute("""SELECT incoming, ideal_reply FROM training_examples
                         WHERE rating='good' ORDER BY embedding <=> %s::vector LIMIT 5""", (qvec,))
            gold = c.fetchall()
            c.execute("""SELECT content FROM message_embeddings
                         WHERE direction='out' AND length(content) > 8
                         ORDER BY embedding <=> %s::vector LIMIT 6""", (qvec,))
            examples = [r["content"] for r in c.fetchall()]
    except Exception as e:
        print(f"RAG retrieve skipped: {e}")

    bought = ", ".join(f"{s['product']} ({s['amount']}€)" for s in sales) or "nothing yet"
    prof_txt = (f"Fan: {prof.get('internal_name') or prof.get('anon_id') or tg_id}\n"
                f"Funnel stage: {prof.get('funnel_stage') or 'unknown'}\n"
                f"Notes: {prof.get('notes') or '-'}\nAlready bought: {bought}\n")
    convo_txt = "\n".join(f"{'FAN' if m['direction']=='in' else 'YOU'}: {m['text']}" for m in hist)
    style_txt = "\n".join(f"- {ex}" for ex in examples)
    gold_txt = "\n".join(f"FAN: {g['incoming']}\nIDEAL REPLY: {g['ideal_reply']}" for g in gold)
    know = _get_knowledge()
    know_txt = "\n".join(f"- {k['content']}" for k in know)

    ppv_txt = "\n".join(f"- {p}" for p in (body.available_ppv or [])) or "(none provided)"
    calls_txt = "\n".join(f"- {c}" for c in (body.available_calls or [])) or "(none provided)"
    pay_txt = "A payment from this fan IS confirmed." if body.payment_confirmed else \
              "NO confirmed payment from this fan."

    sys = (
        _get_persona() + "\n\n"
        "FACTS & RULES you must always follow:\n" + (know_txt or "(none yet)") + "\n\n"
        "PRICE LIST (the ONLY valid prices — quote verbatim):\n"
        + (body.price_list or _get_price_list() or "(none set)") + "\n\n"
        "PAYMENT TERMS (Zahlungsbedingungen — always follow):\n"
        + (_get_payment_terms() or "(none set)") + "\n\n"
        + PRICE_RULE + "\n\n"
        "GOLD-STANDARD examples — follow this approach and tone closely:\n"
        + (gold_txt or "(none yet)") + "\n\n"
        "Style references from past chats (do not copy verbatim):\n" + (style_txt or "(none)") + "\n\n"
        "Fan profile:\n" + prof_txt + "\n"
        "Available PPV files you may send (use EXACT names):\n" + ppv_txt + "\n\n"
        "Available call recordings (folder/filename) you may play:\n" + calls_txt + "\n\n"
        "PAYMENT STATUS: " + pay_txt + "\n\n"
        "HARD RULES:\n"
        "- NEVER send paid PPV content or start a 'paid_calls' recording unless PAYMENT STATUS confirms a payment.\n"
        "- Only reference files that appear in the available lists above.\n"
        "- For refunds, upset fans, or anything risky/sensitive, put [[HANDOFF]] in your reply and take no action.\n"
    )
    user = ("Recent conversation (YOU = the chatter, FAN = the subscriber):\n" + convo_txt + "\n\n"
            "Decide the single best next move as YOU. Write the reply text (short, natural, the fan's language). "
            "If an action fits (send PPV, change funnel stage, start a call), call the matching tool. "
            "If nothing beyond replying is needed, just write the reply.")

    resp = client.chat.completions.create(
        model=AI_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        tools=ACT_TOOLS, tool_choice="auto",
        temperature=0.7, max_tokens=700,
    )
    msg = resp.choices[0].message
    reply = (msg.content or "").strip()
    handoff = "[[HANDOFF]]" in reply
    reply = reply.replace("[[HANDOFF]]", "").strip()
    actions = []
    if not handoff:
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            actions.append({"tool": tc.function.name, "args": args})
    return ActOut(reply=reply, actions=actions, handoff=handoff)


# ── TRAINING: you teach / coach the AI ────────────────────────────────────────
class TeachIn(BaseModel):
    incoming: str            # an example fan message / situation
    ideal_reply: str         # how the AI SHOULD respond
    tags: str = ""

class FeedbackIn(BaseModel):
    incoming: str            # the fan message that was answered
    final_reply: str         # the corrected/approved reply (the right one)
    tg_id: str = ""
    ai_suggestion: str = ""  # what the AI had proposed (for the record)
    rating: str = "good"     # 'good' = use as gold example, 'bad' = avoid this style
    note: str = ""

def _store_example(incoming: str, ideal_reply: str, tags: str, rating: str, source: str):
    vec = _vec_literal(_embed([incoming[:2000]])[0])
    with db() as conn, conn.cursor() as c:
        c.execute(
            "INSERT INTO training_examples (incoming,ideal_reply,tags,rating,source,embedding) "
            "VALUES (%s,%s,%s,%s,%s,%s::vector) RETURNING id",
            (incoming[:4000], ideal_reply[:4000], tags, rating, source, vec),
        )
        new_id = c.fetchone()["id"]
        conn.commit()
    return new_id

@app.post("/teach")
def teach(body: TeachIn, authorization: Optional[str] = Header(None)):
    """You give the AI an ideal answer to a situation. It learns from it."""
    _auth(authorization)
    new_id = _store_example(body.incoming, body.ideal_reply, body.tags, "good", "manual")
    return {"ok": True, "id": new_id}

@app.post("/feedback")
def feedback(body: FeedbackIn, authorization: Optional[str] = Header(None)):
    """Approve/correct an AI draft. 'good' replies become gold examples it follows."""
    _auth(authorization)
    new_id = _store_example(body.incoming, body.final_reply, body.note, body.rating, "feedback")
    return {"ok": True, "id": new_id, "learned": body.rating == "good"}

@app.get("/examples")
def examples(authorization: Optional[str] = Header(None)):
    """How much the AI has been trained so far."""
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        c.execute("SELECT rating, COUNT(*) AS n FROM training_examples GROUP BY rating")
        by_rating = {r["rating"]: r["n"] for r in c.fetchall()}
        c.execute("SELECT id, incoming, ideal_reply, tags, created_at FROM training_examples ORDER BY id DESC LIMIT 20")
        recent = c.fetchall()
    return {"by_rating": by_rating, "recent": recent}


# ── PERSONA / PROMPT (you write the AI's instructions) ────────────────────────
class ConfigIn(BaseModel):
    persona: str

@app.get("/config")
def get_config(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    return {"persona": _get_persona(), "price_list": _get_price_list(),
            "payment_terms": _get_payment_terms()}

@app.post("/config")
def set_config(body: ConfigIn, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        c.execute("INSERT INTO ai_config (key,value) VALUES ('persona',%s) "
                  "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (body.persona,))
        conn.commit()
    return {"ok": True}


# ── PRICE LIST (the authoritative prices — grounds the AI so it never invents) ─
class PriceListIn(BaseModel):
    price_list: str

@app.get("/price-list")
def get_price_list(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    return {"price_list": _get_price_list()}

@app.post("/price-list")
def set_price_list(body: PriceListIn, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        c.execute("INSERT INTO ai_config (key,value) VALUES ('price_list',%s) "
                  "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (body.price_list,))
        conn.commit()
    return {"ok": True}


# ── PAYMENT TERMS (Zahlungsbedingungen — the AI always knows & honors them) ────
class PaymentTermsIn(BaseModel):
    payment_terms: str

@app.get("/payment-terms")
def get_payment_terms_ep(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    return {"payment_terms": _get_payment_terms()}

@app.post("/payment-terms")
def set_payment_terms_ep(body: PaymentTermsIn, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        c.execute("INSERT INTO ai_config (key,value) VALUES ('payment_terms',%s) "
                  "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (body.payment_terms,))
        conn.commit()
    return {"ok": True}


# ── KNOWLEDGE / RULES (teach facts apart from chats) ──────────────────────────
class KnowledgeIn(BaseModel):
    content: str

@app.get("/knowledge")
def list_knowledge(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    return {"items": _get_knowledge()}

@app.post("/knowledge")
def add_knowledge(body: KnowledgeIn, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        c.execute("INSERT INTO knowledge (content) VALUES (%s) RETURNING id", (body.content[:2000],))
        nid = c.fetchone()["id"]
        conn.commit()
    return {"ok": True, "id": nid}

@app.delete("/knowledge/{kid}")
def del_knowledge(kid: int, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        c.execute("DELETE FROM knowledge WHERE id=%s", (kid,))
        conn.commit()
    return {"ok": True}


# ── PLAYGROUND (chat freely with the AI to test prompt + knowledge) ───────────
class PlaygroundIn(BaseModel):
    message: str

@app.post("/playground")
def playground(body: PlaygroundIn, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    know = _get_knowledge()
    know_txt = "\n".join(f"- {k['content']}" for k in know)
    gold = []
    try:
        qvec = _vec_literal(_embed([body.message[:2000]])[0])
        with db() as conn, conn.cursor() as c:
            c.execute("""SELECT incoming, ideal_reply FROM training_examples
                         WHERE rating='good' ORDER BY embedding <=> %s::vector LIMIT 4""", (qvec,))
            gold = c.fetchall()
    except Exception:
        pass
    gold_txt = "\n".join(f"FAN: {g['incoming']}\nIDEAL: {g['ideal_reply']}" for g in gold)
    sys = (
        _get_persona() + "\n\n"
        + _grounding_block(know_txt) + "\n\n"
        "GOLD examples:\n" + (gold_txt or "(none)")
    )
    chat = client.chat.completions.create(
        model=AI_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": body.message}],
        temperature=0.8, max_tokens=700,
    )
    out = (chat.choices[0].message.content or "").strip()
    return {"reply": out.replace("[[HANDOFF]]", "").strip(), "handoff": "[[HANDOFF]]" in out}


# ── TRAINER CHAT (talk to the AI in plain language to instruct/configure it) ──
class ChatMsg(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    messages: List[ChatMsg]

TRAINER_SYS = (
    "Du bist der Trainings-Assistent fuer eine KI-Chatterin (Adult-Content-Verkauf). "
    "Der Betreiber (Admin) spricht mit dir in normaler Sprache und sagt dir, WIE die "
    "Chatterin sich verhalten soll: Charakter, Ton, Regeln, Preise, Verbote, Ablaeufe. "
    "Deine Aufgabe ist es, gemeinsam mit ihm den PERSONA-PROMPT der Chatterin Schritt fuer "
    "Schritt zu verbessern und sein Feedback dauerhaft einzuarbeiten.\n\n"
    "Antworte freundlich und kurz auf Deutsch wie ein Kollege. Wenn eine Anweisung klar ist, "
    "bestaetige sie knapp ('Verstanden — ...') und fasse in einem Satz zusammen, was du "
    "eingearbeitet hast. Wenn etwas unklar oder mehrdeutig ist, stelle GENAU EINE gezielte "
    "Rueckfrage und aendere noch nichts. Erfinde nichts, was der Betreiber nicht gesagt hat.\n\n"
    "WICHTIG — wenn der Betreiber etwas sagt, das die Persona betrifft (Charakter, Ton, Stil, "
    "Verhalten, Verbote), dann gib AM ENDE deiner Antwort den KOMPLETTEN, verbesserten Persona-"
    "Prompt aus — die bisherige Persona plus die neue Anweisung sauber eingearbeitet, nicht nur "
    "den Zusatz. Format, exakt:\n"
    "[[PERSONA]]\n<vollstaendiger neuer Persona-Text>\n[[/PERSONA]]\n"
    "Wenn die Nachricht NUR eine reine Faktregel ist (Name, Alter, Preis, Zahlungsweg), gib "
    "stattdessen am Ende eine Zeile aus: [[RULE]] <die Regel in einem kurzen Satz>. "
    "Pro Antwort entweder ein [[PERSONA]]-Block ODER eine [[RULE]]-Zeile — und nur, wenn sich "
    "wirklich etwas Bleibendes geaendert hat. Bei reinen Rueckfragen oder Smalltalk: keins von beidem."
)

@app.post("/chat")
def trainer_chat(body: ChatIn, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    persona = _get_persona()
    know = _get_knowledge()
    know_txt = "\n".join(f"- {k['content']}" for k in know)
    sys = (
        TRAINER_SYS + "\n\n"
        "AKTUELLE PERSONA DER CHATTERIN:\n" + (persona or "(noch keine gesetzt)") + "\n\n"
        "BEREITS GESPEICHERTE REGELN:\n" + (know_txt or "(noch keine)") + "\n\n"
        "AKTUELLE PREISLISTE (verbindlich):\n" + (_get_price_list() or "(noch keine)") + "\n\n"
        "AKTUELLE ZAHLUNGSBEDINGUNGEN:\n" + (_get_payment_terms() or "(noch keine)")
    )
    msgs = [{"role": "system", "content": sys}]
    for m in body.messages[-20:]:
        role = m.role if m.role in ("user", "assistant") else "user"
        msgs.append({"role": role, "content": m.content[:4000]})
    chat = client.chat.completions.create(
        model=AI_MODEL, messages=msgs, temperature=0.4, max_tokens=900,
    )
    out = (chat.choices[0].message.content or "").strip()
    persona_update = None
    rule = None
    if "[[PERSONA]]" in out and "[[/PERSONA]]" in out:
        head, rest = out.split("[[PERSONA]]", 1)
        body_txt, _ = rest.split("[[/PERSONA]]", 1)
        persona_update = body_txt.strip()
        out = head.strip()
    if "[[RULE]]" in out:
        parts = out.split("[[RULE]]", 1)
        out = parts[0].strip()
        rule = parts[1].strip().lstrip(":-").strip() or None
    return {"reply": out, "rule": rule, "persona_update": persona_update}


# Apply a persona that was proposed in the trainer chat (one-click "remember")
class PersonaApply(BaseModel):
    persona: str

@app.post("/chat/apply-persona")
def apply_persona(body: PersonaApply, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    with db() as conn, conn.cursor() as c:
        c.execute("INSERT INTO ai_config (key,value) VALUES ('persona',%s) "
                  "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (body.persona,))
        conn.commit()
    return {"ok": True}


# ── WINNING-CHAT MINER (learn from conversations that actually closed money) ──
# For every real sale, grab the fan's last message before the sale and the
# chatter's closing reply, and store it as a GOLD example. The AI then learns
# the exact wording/timing that converted — not theory. This is the biggest
# single lever to make it outperform average chatters.
@app.post("/mine-winning")
def mine_winning(limit: int = Query(400, description="how many recent sales to scan"),
                 authorization: Optional[str] = Header(None)):
    _auth(authorization)
    mined, skipped, scanned = 0, 0, 0
    try:
        with db() as conn, conn.cursor() as c:
            c.execute("SELECT id, tg_id, product, amount, timestamp FROM sales "
                      "ORDER BY id DESC LIMIT %s", (limit,))
            sales = c.fetchall()
    except Exception as e:
        raise HTTPException(500, f"sales read failed: {e}")

    scanned = len(sales)
    for s in sales:
        sale_tag = f"sale{s['id']}"
        # dedup — never mine the same sale twice
        try:
            with db() as conn, conn.cursor() as c:
                c.execute("SELECT 1 FROM training_examples WHERE source='winning' "
                          "AND tags LIKE %s LIMIT 1", (f"%{sale_tag}%",))
                if c.fetchone():
                    skipped += 1
                    continue
        except Exception:
            pass
        # the exchange leading up to the sale (chronological)
        try:
            with db() as conn, conn.cursor() as c:
                c.execute("""SELECT direction, text FROM messages
                             WHERE tg_id=%s AND timestamp <= %s
                             ORDER BY timestamp DESC LIMIT 10""",
                          (s["tg_id"], str(s["timestamp"])))
                rows = list(reversed(c.fetchall()))
        except Exception:
            continue
        if not rows:
            continue
        # incoming = fan's last real message; ideal_reply = chatter's closing line(s) after it
        last_in = None
        for i in range(len(rows) - 1, -1, -1):
            t = (rows[i]["text"] or "").strip()
            if rows[i]["direction"] == "in" and t and not t.startswith("["):
                last_in = i
                break
        if last_in is None:
            continue
        incoming = (rows[last_in]["text"] or "").strip()
        closing = " ".join((r["text"] or "").strip() for r in rows[last_in + 1:]
                           if r["direction"] == "out" and (r["text"] or "").strip()
                           and not (r["text"] or "").strip().startswith("["))
        if not closing:
            # fallback: the chatter's pitch right before the fan's message
            for j in range(last_in - 1, -1, -1):
                t = (rows[j]["text"] or "").strip()
                if rows[j]["direction"] == "out" and t and not t.startswith("["):
                    closing = t
                    break
        if not incoming or not closing:
            continue
        try:
            _store_example(incoming, closing,
                           f"winning,{sale_tag},{(s['product'] or '').strip()}",
                           "good", "winning")
            mined += 1
        except Exception as e:
            print(f"mine store error: {e}")
    return {"mined_now": mined, "skipped_existing": skipped, "sales_scanned": scanned}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8090)))
