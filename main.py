"""
Chatter CRM – Subscriber Chat Backend
Telethon Userbot + FastAPI REST API — PostgreSQL + SQLite fallback (production-ready)
"""
from __future__ import annotations

import asyncio
import os
import csv
import io
import re
import uuid
import shutil
import threading
import urllib.request
import urllib.parse
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
from datetime import datetime, timedelta
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

# Vault storage directory (Railway volume or local)
VAULT_DIR = os.environ.get('VAULT_PATH', '/data/vault')
os.makedirs(VAULT_DIR, exist_ok=True)

# Sale proof screenshots directory
PROOFS_DIR = os.path.join(VAULT_DIR, '_proofs')
os.makedirs(PROOFS_DIR, exist_ok=True)

# Subscriber media cache directory (downloaded Telegram photos/docs)
MEDIA_CACHE_DIR = os.path.join(VAULT_DIR, '_media')
os.makedirs(MEDIA_CACHE_DIR, exist_ok=True)

# Fake call recordings directory
CALLS_DIR = os.path.join(VAULT_DIR, '_calls')
CALL_FOLDERS = ['fake_checks', 'paid_calls']  # fixed folder names
os.makedirs(CALLS_DIR, exist_ok=True)
for _f in CALL_FOLDERS:
    os.makedirs(os.path.join(CALLS_DIR, _f), exist_ok=True)

# ── pytgcalls (optional — fake call feature) ──────────────────────────────────
_PYTGCALLS_ERR = None
try:
    from pytgcalls import PyTgCalls
    from pytgcalls.types import MediaStream, AudioQuality
    _PYTGCALLS_OK = True
    print('✅ pytgcalls available')
except Exception as _e:
    _PYTGCALLS_OK = False
    _PYTGCALLS_ERR = str(_e)
    print(f'⚠️  pytgcalls not available: {_e}')

calls_client: Optional[object] = None
active_calls: dict = {}   # tg_id → {file, chatter, started_at}
_tg_priority: int = 0    # >0 = broadcast pauses (reply/call in progress)

import sqlite3 as _sqlite3

import uvicorn
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_OK = True
except ImportError:
    _PSYCOPG2_OK = False
from fastapi import FastAPI, HTTPException, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from telethon import TelegramClient, events, functions, types
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession
from telethon.tl.types import (User, InputPeerUser, UpdateReadHistoryOutbox, PeerUser,
                               UpdateUserStatus, UserStatusOnline, UserStatusOffline,
                               UserStatusRecently, UserStatusLastWeek, UserStatusLastMonth)
# Safe optional import — not present in all Telethon builds
try:
    from telethon.tl.types import UpdatePhoneCall as _UpdatePhoneCall
    from telethon.tl.types import PhoneCallDiscarded as _PhoneCallDiscarded
    _PHONE_CALL_TYPES_OK = True
except ImportError:
    _UpdatePhoneCall = None
    _PhoneCallDiscarded = None
    _PHONE_CALL_TYPES_OK = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
TG_API_ID   = os.environ.get('TG_API_ID', '')
TG_API_HASH = os.environ.get('TG_API_HASH', '')
TG_SESSION  = os.environ.get('TG_SESSION', '')
PORT        = int(os.environ.get('PORT', 8000))
DATABASE_URL = os.environ.get('DATABASE_URL', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
SUBSCRIBER_BACKUP_WEBHOOK = os.environ.get('SUBSCRIBER_BACKUP_WEBHOOK', '')

# ── PostgreSQL health check at startup ───────────────────────────────────────
# If DATABASE_URL is set but PostgreSQL is down/red, fall back to SQLite.
_PG_AVAILABLE = False
if DATABASE_URL and _PSYCOPG2_OK:
    try:
        import psycopg2 as _pg_test
        _test_conn = _pg_test.connect(DATABASE_URL)
        _test_conn.close()
        _PG_AVAILABLE = True
        print('✅ PostgreSQL reachable — using PostgreSQL')
    except Exception as _pg_err:
        print(f'⚠️ PostgreSQL unreachable ({_pg_err}) — falling back to SQLite')
        _PG_AVAILABLE = False

# ── WEBSOCKET MANAGER ────────────────────────────────────────────────────────
class WSManager:
    def __init__(self):
        self._connections: set = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.add(ws)
        print(f'🔌 WS connected — total: {len(self._connections)}')

    def disconnect(self, ws: WebSocket):
        self._connections.discard(ws)
        print(f'🔌 WS disconnected — total: {len(self._connections)}')

    async def broadcast(self, data: dict):
        dead = set()
        for ws in list(self._connections):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._connections -= dead

ws_manager = WSManager()

# ── DEBUG: zeige ob DATABASE_URL gesetzt ist ─────────────────────────────────
_db_preview = DATABASE_URL[:40] + '...' if len(DATABASE_URL) > 40 else DATABASE_URL
print(f'🔍 DATABASE_URL (erste 40 Zeichen): [{_db_preview}]  (leer={not DATABASE_URL})')

# ── SETUP STATE ───────────────────────────────────────────────────────────────
setup_client: Optional[TelegramClient] = None
setup_phone: str = ''

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = os.environ.get('DB_PATH', '/data/chatter_crm.db')
# Use SQLite if: no DATABASE_URL, psycopg2 not installed, OR PostgreSQL is down
USE_SQLITE = not _PG_AVAILABLE

if USE_SQLITE:
    print(f'📂 Using SQLite: {DB_PATH}')
else:
    print(f'🐘 Using PostgreSQL')

# ─ SQLite adapter: makes sqlite3 behave like psycopg2+RealDictCursor ─────────
class _SLCursor:
    """Wraps sqlite3 cursor to behave like psycopg2 RealDictCursor (dict rows, %s params, RETURNING)."""
    def __init__(self, cur, conn_ref):
        self._c = cur
        self._conn = conn_ref  # for UPDATE RETURNING SELECT
        self._ret_rows = None

    def execute(self, sql, params=()):
        self._ret_rows = None
        returning = None
        upper = sql.upper()
        ret_pos = upper.rfind(' RETURNING ')
        if ret_pos >= 0:
            returning = sql[ret_pos + 11:].strip()
            sql = sql[:ret_pos]
        # adapt syntax
        sql = sql.replace('%s', '?')
        sql = sql.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY')
        sql = sql.replace('BOOLEAN DEFAULT FALSE', 'INTEGER DEFAULT 0')
        sql = sql.replace('BOOLEAN DEFAULT TRUE', 'INTEGER DEFAULT 1')
        sql = sql.replace('BOOLEAN DEFAULT NULL', 'INTEGER DEFAULT NULL')
        sql = sql.replace(' BOOLEAN ', ' INTEGER ')
        sql = sql.replace('ADD COLUMN IF NOT EXISTS', 'ADD COLUMN')
        # PostgreSQL SUBSTRING(col FROM n) → SQLite SUBSTR(col, n)
        sql = _re.sub(r"SUBSTRING\((\w+)\s+FROM\s+(\d+)\)", r"SUBSTR(\1,\2)", sql, flags=_re.IGNORECASE)
        self._c.execute(sql, params or ())
        if returning:
            self._handle_returning(returning, sql, params)

    def _handle_returning(self, returning, sql, params):
        upper = sql.upper().strip()
        if upper.startswith('INSERT') or upper.startswith('UPDATE'):
            last_id = self._c.lastrowid
            cols = [c.strip() for c in returning.split(',')]
            if cols == ['id'] or cols == ['*']:
                self._ret_rows = [{'id': last_id}]
            else:
                # Need a SELECT — figure out table name and row id
                tbl_match = _re.search(r'(?:INTO|UPDATE)\s+(\w+)', sql, _re.IGNORECASE)
                if tbl_match and last_id:
                    tbl = tbl_match.group(1)
                    sel = f"SELECT {', '.join(cols)} FROM {tbl} WHERE rowid = ?"
                    try:
                        cur2 = self._conn.cursor()
                        cur2.execute(sel, (last_id,))
                        row = cur2.fetchone()
                        if row and cur2.description:
                            keys = [d[0] for d in cur2.description]
                            self._ret_rows = [dict(zip(keys, row))]
                        else:
                            self._ret_rows = [{'id': last_id}]
                    except Exception:
                        self._ret_rows = [{'id': last_id}]
                else:
                    self._ret_rows = [{'id': last_id}]

    def _row_to_dict(self, row):
        if row is None:
            return None
        if isinstance(row, dict):
            return row
        cols = [d[0] for d in self._c.description]
        return dict(zip(cols, row))

    def fetchone(self):
        if self._ret_rows is not None:
            return self._ret_rows[0] if self._ret_rows else None
        return self._row_to_dict(self._c.fetchone())

    def fetchall(self):
        if self._ret_rows is not None:
            return self._ret_rows
        rows = self._c.fetchall()
        if not rows or not self._c.description:
            return []
        cols = [d[0] for d in self._c.description]
        return [dict(zip(cols, r)) for r in rows]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

class _SLConn:
    """Wraps sqlite3 connection to behave like psycopg2 connection."""
    def __init__(self, path):
        self._c = _sqlite3.connect(path, check_same_thread=False)
        self._c.execute('PRAGMA journal_mode=WAL')
        self._c.execute('PRAGMA foreign_keys=ON')

    def cursor(self):
        return _SLCursor(self._c.cursor(), self._c)

    def commit(self):
        self._c.commit()

    def rollback(self):
        try: self._c.rollback()
        except Exception: pass

    def close(self):
        pass  # keep SQLite connection open

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._c.commit()
        else:
            self.rollback()

# Single shared SQLite connection (WAL mode allows concurrent reads)
_sqlite_conn: Optional[_SLConn] = None
_sqlite_lock = threading.Lock()

def _get_sqlite():
    global _sqlite_conn
    with _sqlite_lock:
        if _sqlite_conn is None:
            _sqlite_conn = _SLConn(DB_PATH)
    return _sqlite_conn

@contextmanager
def db():
    """Open a fresh connection per request — commit on success, rollback+close on error."""
    if USE_SQLITE:
        conn = _get_sqlite()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

def init_db():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS conversations (
                tg_id         TEXT PRIMARY KEY,
                anon_id       TEXT NOT NULL,
                internal_name TEXT DEFAULT '',
                notes         TEXT DEFAULT '',
                last_msg      TEXT DEFAULT '',
                last_time     TEXT DEFAULT '',
                first_time    TEXT DEFAULT '',
                unread        INTEGER DEFAULT 0,
                msg_count     INTEGER DEFAULT 0,
                time_waster   BOOLEAN DEFAULT FALSE,
                tg_username   TEXT DEFAULT '',
                tg_phone      TEXT DEFAULT ''
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY,
                tg_id     TEXT NOT NULL,
                text      TEXT NOT NULL,
                direction TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                chatter   TEXT DEFAULT ''
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS sales (
                id        INTEGER PRIMARY KEY,
                tg_id     TEXT NOT NULL,
                anon_id   TEXT NOT NULL,
                amount    REAL NOT NULL,
                product   TEXT DEFAULT '',
                notes     TEXT DEFAULT '',
                chatter   TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS pledges (
                id              INTEGER PRIMARY KEY,
                tg_id           TEXT NOT NULL,
                anon_id         TEXT NOT NULL,
                amount          REAL NOT NULL,
                payment_method  TEXT DEFAULT '',
                notes           TEXT DEFAULT '',
                chatter         TEXT DEFAULT '',
                deadline        TEXT NOT NULL,
                status          TEXT DEFAULT 'open',
                created_at      TEXT NOT NULL,
                paid_at         TEXT DEFAULT ''
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS scheduled_broadcasts (
                id              INTEGER PRIMARY KEY,
                text            TEXT NOT NULL,
                scheduled_at    TEXT NOT NULL,
                filter_stage    TEXT DEFAULT '',
                exclude_stages  TEXT DEFAULT '',
                chatter         TEXT DEFAULT 'Broadcast',
                status          TEXT DEFAULT 'pending',
                created_at      TEXT NOT NULL,
                sent_at         TEXT DEFAULT '',
                sent_count      INTEGER DEFAULT 0,
                fail_count      INTEGER DEFAULT 0
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS lists (
                id    INTEGER PRIMARY KEY,
                name  TEXT NOT NULL,
                color TEXT DEFAULT '#00d4aa'
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS list_members (
                list_id INTEGER NOT NULL,
                tg_id   TEXT NOT NULL,
                PRIMARY KEY (list_id, tg_id)
            )''')
            # PayPal payment-received notifications (parsed from email)
            c.execute('''CREATE TABLE IF NOT EXISTS paypal_notifications (
                id        SERIAL PRIMARY KEY,
                mail_uid  TEXT UNIQUE,
                amount    TEXT DEFAULT '',
                currency  TEXT DEFAULT '',
                sender    TEXT DEFAULT '',
                subject   TEXT DEFAULT '',
                ts        TEXT NOT NULL,
                seen      INTEGER DEFAULT 0
            )''')
            # Users table
            c.execute('''CREATE TABLE IF NOT EXISTS crm_users (
                id            INTEGER PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE,
                email         TEXT DEFAULT '',
                password_hash TEXT NOT NULL,
                role          TEXT DEFAULT 'chatter'
            )''')
            # Add display_name column if missing (safe — ignore if already exists)
            try:
                c.execute("ALTER TABLE crm_users ADD COLUMN IF NOT EXISTS display_name TEXT DEFAULT ''")
            except Exception:
                pass  # column already exists from previous deployment
            # Seed default admin if no users exist
            c.execute('SELECT COUNT(*) as n FROM crm_users')
            if c.fetchone()['n'] == 0:
                import hashlib
                c.execute(
                    "INSERT INTO crm_users (username,email,password_hash,role,display_name) VALUES (%s,%s,%s,%s,%s)",
                    ('Glenn', 'glennzimmerhh@gmail.com', hashlib.sha256('Smartviral1!'.encode()).hexdigest(), 'admin', 'Glenn')
                )

            # Settings table
            c.execute('''CREATE TABLE IF NOT EXISTS crm_settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            )''')
            # safe migrations + indexes for performance
            for stmt in [
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS time_waster BOOLEAN DEFAULT FALSE",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS is_muted BOOLEAN DEFAULT FALSE",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tg_username TEXT DEFAULT ''",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tg_phone TEXT DEFAULT ''",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tg_access_hash TEXT DEFAULT ''",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS followup_at TEXT DEFAULT NULL",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS is_online BOOLEAN DEFAULT FALSE",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS last_seen TEXT DEFAULT NULL",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS funnel_stage TEXT DEFAULT 'kalt'",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS call_followup_at TEXT DEFAULT NULL",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS call_followup_note TEXT DEFAULT ''",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS last_auto_msg_at TEXT DEFAULT NULL",
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS tg_msg_id INTEGER DEFAULT 0",
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS is_read BOOLEAN DEFAULT FALSE",
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS read_at TEXT DEFAULT ''",
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS translation TEXT DEFAULT ''",
                # Sale proof system
                "ALTER TABLE sales ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'approved'",
                "ALTER TABLE sales ADD COLUMN IF NOT EXISTS screenshot TEXT DEFAULT ''",
                "ALTER TABLE sales ADD COLUMN IF NOT EXISTS rejection_reason TEXT DEFAULT ''",
                "ALTER TABLE sales ADD COLUMN IF NOT EXISTS reviewed_by TEXT DEFAULT ''",
                "ALTER TABLE sales ADD COLUMN IF NOT EXISTS reviewed_at TEXT DEFAULT ''",
                "ALTER TABLE sales ADD COLUMN IF NOT EXISTS payment_method TEXT DEFAULT ''",
                "ALTER TABLE sales ADD COLUMN IF NOT EXISTS payment_code TEXT DEFAULT ''",
                # YouSafe reference codes
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS payment_ref TEXT DEFAULT ''",
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS source TEXT DEFAULT ''",
                "CREATE INDEX IF NOT EXISTS idx_messages_tg_id ON messages(tg_id)",
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_conv_last_time ON conversations(last_time DESC)",

            ]:
                try:
                    c.execute(stmt)
                except Exception as e:
                    print(f'Migration skip: {e}')
                    conn.rollback()
    # Backfill payment_ref for existing subscribers
    with db() as conn:
        with conn.cursor() as c:
            if USE_SQLITE:
                c.execute("UPDATE conversations SET payment_ref='ZF-'||SUBSTR(anon_id,7) WHERE (payment_ref IS NULL OR payment_ref='') AND anon_id LIKE 'User #%'")
            else:
                c.execute("UPDATE conversations SET payment_ref='ZF-'||SUBSTRING(anon_id FROM 7) WHERE (payment_ref IS NULL OR payment_ref='') AND anon_id LIKE 'User #%'")
    print('✅ DB initialized')

def _fire_webhook_sync(url: str, payload: dict):
    """Non-blocking fire-and-forget webhook via background thread."""
    import threading, urllib.request, json as _j
    def _send():
        try:
            data = _j.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=8)
        except Exception as e:
            print(f'⚠️  Webhook failed: {e}')
    threading.Thread(target=_send, daemon=True).start()

def ensure_conv(tg_id: str, username: str = '', phone: str = '', access_hash: str = '') -> str:
    is_new = False
    anon_id = ''
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT anon_id FROM conversations WHERE tg_id=%s', (tg_id,))
            row = c.fetchone()
            if row:
                c.execute(
                    "UPDATE conversations SET tg_username=COALESCE(NULLIF(tg_username,''),%s), tg_phone=COALESCE(NULLIF(tg_phone,''),%s), tg_access_hash=COALESCE(NULLIF(tg_access_hash,''),%s) WHERE tg_id=%s",
                    (username, phone, access_hash, tg_id)
                )
                return row['anon_id']
            c.execute('SELECT COUNT(*) as n FROM conversations')
            n = c.fetchone()['n']
            anon_id = f'User #{1001 + n}'
            payment_ref = f'ZF-{1001 + n}'
            now = datetime.now().isoformat()
            c.execute(
                'INSERT INTO conversations (tg_id,anon_id,last_msg,last_time,first_time,unread,msg_count,tg_username,tg_phone,payment_ref) VALUES (%s,%s,%s,%s,%s,0,0,%s,%s,%s)',
                (tg_id, anon_id, '', now, now, username, phone, payment_ref)
            )
            # Backfill payment_ref for any existing subscribers that don't have one
            c.execute("UPDATE conversations SET payment_ref='ZF-'||SUBSTRING(anon_id FROM 7) WHERE payment_ref='' AND anon_id LIKE 'User #%'")
            is_new = True
    if is_new and SUBSCRIBER_BACKUP_WEBHOOK:
        _fire_webhook_sync(SUBSCRIBER_BACKUP_WEBHOOK, {
            'tg_id': tg_id, 'anon_id': anon_id,
            'username': username, 'phone': phone,
            'first_seen': datetime.now().isoformat()
        })
    return anon_id

def save_msg(tg_id: str, text: str, direction: str, chatter: str = '', tg_msg_id: int = 0):
    import threading
    ts = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                'INSERT INTO messages (tg_id,text,direction,timestamp,chatter,tg_msg_id) VALUES (%s,%s,%s,%s,%s,%s)',
                (tg_id, text, direction, ts, chatter, tg_msg_id)
            )
            if direction == 'in':
                # Incoming message clears follow-up timer
                c.execute(
                    'UPDATE conversations SET last_msg=%s, last_time=%s, unread=unread+1, msg_count=msg_count+1, followup_at=NULL WHERE tg_id=%s',
                    (text[:100], ts, tg_id)
                )
                # Auto-translate in background thread
                threading.Thread(target=_auto_translate_message, args=(tg_id, text), daemon=True).start()
            else:
                c.execute(
                    'UPDATE conversations SET last_msg=%s, last_time=%s, msg_count=msg_count+1 WHERE tg_id=%s',
                    (text[:100], ts, tg_id)
                )

_auto_tx_sem = threading.Semaphore(3)
def _auto_translate_message(tg_id: str, text: str):
    """Translate incoming message to English and store in DB (background thread)."""
    if not OPENAI_API_KEY or not text.strip():
        return
    if not _auto_tx_sem.acquire(blocking=False):
        return  # cap concurrent auto-translations so the worker never gets flooded
    try:
        import urllib.request as _r, json as _j
        payload = _j.dumps({
            'model': 'gpt-4o-mini',
            'messages': [
                {'role': 'system', 'content': 'Translate the following message to English. Return ONLY the translation.'},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': 200, 'temperature': 0.3
        }).encode()
        req = _r.Request('https://api.openai.com/v1/chat/completions', data=payload,
                         headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_API_KEY}'})
        with _r.urlopen(req, timeout=12) as resp:
            data = _j.loads(resp.read())
        translation = data['choices'][0]['message']['content'].strip()
        with db() as conn:
            with conn.cursor() as c:
                c.execute(
                    "UPDATE messages SET translation=%s WHERE id=(SELECT id FROM messages WHERE tg_id=%s AND text=%s AND direction='in' AND translation='' ORDER BY id DESC LIMIT 1)",
                    (translation, tg_id, text)
                )
    except Exception as e:
        print(f'Auto-translate error: {e}')
    finally:
        _auto_tx_sem.release()

def _send_auto_online_msg(tg_id: str):
    """Fire auto-message to subscriber who just came online (run in thread)."""
    import threading
    def _do():
        try:
            enabled = get_setting('auto_online_enabled', '0')
            if enabled != '1':
                return
            text = get_setting('auto_online_text', '').strip()
            if not text:
                return
            cooldown_h = int(get_setting('auto_online_cooldown_h', '24') or 24)
            allowed_stages = get_setting('auto_online_stages', '')  # comma-sep or empty=all

            with db() as conn:
                with conn.cursor() as c:
                    c.execute('SELECT tg_access_hash, funnel_stage, last_auto_msg_at FROM conversations WHERE tg_id=%s', (tg_id,))
                    row = c.fetchone()
            if not row:
                return
            # Stage filter
            if allowed_stages:
                stages = [s.strip() for s in allowed_stages.split(',')]
                if row['funnel_stage'] not in stages:
                    return
            # Cooldown check
            if row['last_auto_msg_at']:
                from datetime import timezone
                last = datetime.fromisoformat(row['last_auto_msg_at'])
                diff_h = (datetime.now() - last).total_seconds() / 3600
                if diff_h < cooldown_h:
                    return
            # Send message via Telethon (must run in event loop)
            import asyncio
            async def _send():
                if not tg_client or not tg_client.is_connected():
                    return
                ah = int(row['tg_access_hash']) if row['tg_access_hash'] else 0
                peer = InputPeerUser(int(tg_id), ah) if ah else int(tg_id)
                sent = await tg_client.send_message(peer, text)
                now = datetime.now().isoformat()
                save_msg(tg_id, text, 'out', 'Auto', tg_msg_id=sent.id)
                with db() as conn:
                    with conn.cursor() as c:
                        c.execute('UPDATE conversations SET last_auto_msg_at=%s WHERE tg_id=%s', (now, tg_id))
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(_send(), loop).result(timeout=15)
        except Exception as e:
            print(f'Auto-online-msg error: {e}')
    threading.Thread(target=_do, daemon=True).start()

def update_online_status(tg_id: str, is_online: bool):
    """Update subscriber online status — only for known conversations."""
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT 1 FROM conversations WHERE tg_id=%s', (tg_id,))
            if c.fetchone():
                now = datetime.now().isoformat()
                c.execute(
                    'UPDATE conversations SET is_online=%s, last_seen=%s WHERE tg_id=%s',
                    (is_online, now if is_online else None, tg_id)
                )
                if is_online:
                    _send_auto_online_msg(tg_id)

def mark_read(tg_id: str, max_msg_id: int):
    """Mark outgoing messages as read and start follow-up timer."""
    now = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                "UPDATE messages SET is_read=TRUE, read_at=%s WHERE tg_id=%s AND direction='out' AND tg_msg_id>0 AND tg_msg_id<=%s AND is_read=FALSE",
                (now, tg_id, max_msg_id)
            )
            updated = c.rowcount
            if updated > 0:
                # Start follow-up timer only if not already set
                c.execute(
                    "UPDATE conversations SET followup_at=%s WHERE tg_id=%s AND followup_at IS NULL",
                    (now, tg_id)
                )
                print(f'✓✓ {tg_id}: gelesen, Follow-up Timer gestartet')

# ── USERBOT WITH AUTO-RECONNECT ───────────────────────────────────────────────
tg_client: Optional[TelegramClient] = None
_tg_client_ready = False   # True only when client is started and connected
_userbot_running = False

async def _reinit_calls(client):
    """(Re-)initialize PyTgCalls bound to the given Telethon client.
    Called every time the userbot successfully connects so calls_client
    always uses the live MTProto session."""
    global calls_client
    await asyncio.sleep(3)   # let Telethon fully settle
    if not _PYTGCALLS_OK or not client:
        return
    # Stop old instance cleanly
    if calls_client:
        try:
            await calls_client.stop()
        except Exception:
            pass
        calls_client = None
    try:
        new_client = PyTgCalls(client)
        await new_client.start()

        @new_client.on_update()
        async def _on_call_update(update):
            try:
                update_type = type(update).__name__
                chat_id_raw = getattr(update, 'chat_id', None)
                print(f'📡 pytgcalls update: {update_type} chat_id={chat_id_raw}')
                tg_id_str = str(chat_id_raw) if chat_id_raw is not None else ''
                if not tg_id_str or tg_id_str not in active_calls:
                    return
                update_str = update_type.lower()
                STREAM_END_TYPES = {
                    'StreamAudioEnded', 'StreamVideoEnded', 'StreamEnded',
                    'AudioStreamEnded', 'VideoStreamEnded',
                }
                is_stream_end = update_type in STREAM_END_TYPES or 'ended' in update_str or 'finish' in update_str
                try:
                    from pytgcalls.types import StreamEnded as _SE
                    if isinstance(update, _SE):
                        is_stream_end = True
                except ImportError:
                    pass
                if is_stream_end:
                    print(f'🔔 Stream ended for {tg_id_str} — hanging up')
                    try:
                        if hasattr(new_client, 'leave_call'):
                            await new_client.leave_call(int(tg_id_str))
                        elif hasattr(new_client, 'leave'):
                            await new_client.leave(int(tg_id_str))
                    except Exception as _e:
                        print(f'⚠️ leave_call: {_e}')
                    active_calls.pop(tg_id_str, None)
                    asyncio.create_task(ws_manager.broadcast({'type': 'call_ended', 'tg_id': tg_id_str}))
                    return
                CALL_END_TYPES = {'CallEnded', 'KickedFromGroupCallParticipant', 'ClosedVoiceChat', 'GroupCallEnded'}
                if update_type in CALL_END_TYPES or 'ended' in update_str:
                    print(f'📵 Call ended by subscriber {tg_id_str}')
                    active_calls.pop(tg_id_str, None)
                    asyncio.create_task(ws_manager.broadcast({'type': 'call_ended', 'tg_id': tg_id_str}))
            except Exception as e:
                print(f'⚠️ _on_call_update: {e}')

        calls_client = new_client
        print('✅ PyTgCalls (re-)initialized with fresh MTProto client')
    except Exception as e:
        print(f'⚠️ PyTgCalls reinit failed: {e}')


async def start_userbot():
    global tg_client, _tg_client_ready, _userbot_running
    if not (TG_API_ID and TG_API_HASH and TG_SESSION):
        print('⚠️  Userbot nicht gestartet – Env-Variablen fehlen')
        return

    _userbot_running = True
    retry_delay = 5

    while _userbot_running:
        try:
            tg_client = TelegramClient(
                StringSession(TG_SESSION), int(TG_API_ID), TG_API_HASH,
                connection_retries=5, retry_delay=3, auto_reconnect=True
            )

            @tg_client.on(events.Raw(UpdateUserStatus))
            async def on_user_status(event):
                """Track subscriber online/offline status."""
                tg_id = str(event.user_id)
                is_online = isinstance(event.status, UserStatusOnline)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: update_online_status(tg_id, is_online))
                asyncio.create_task(ws_manager.broadcast({
                    'type': 'online_status', 'tg_id': tg_id, 'is_online': is_online
                }))

            @tg_client.on(events.Raw(UpdateReadHistoryOutbox))
            async def on_outbox_read(event):
                """Fires when subscriber reads our outgoing messages."""
                if isinstance(event.peer, PeerUser):
                    tg_id = str(event.peer.user_id)
                    max_id = event.max_id
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: mark_read(tg_id, max_id))
                    asyncio.create_task(ws_manager.broadcast({
                        'type': 'read_receipt', 'tg_id': tg_id, 'max_id': max_id
                    }))

            if _PHONE_CALL_TYPES_OK:
                @tg_client.on(events.Raw(_UpdatePhoneCall))
                async def on_phone_call_update(update):
                    """Fires when subscriber hangs up or a call is discarded."""
                    try:
                        if isinstance(update.phone_call, _PhoneCallDiscarded):
                            print(f'📵 Telethon: PhoneCallDiscarded — ending active calls')
                            ended = list(active_calls.keys())
                            for tg_id_str in ended:
                                active_calls.pop(tg_id_str, None)
                                asyncio.create_task(ws_manager.broadcast({
                                    'type': 'call_ended', 'tg_id': tg_id_str
                                }))
                            # Also try to cleanly leave via pytgcalls
                            if calls_client:
                                for tg_id_str in ended:
                                    try:
                                        if hasattr(calls_client, 'leave_call'):
                                            await calls_client.leave_call(int(tg_id_str))
                                    except Exception:
                                        pass
                    except Exception as e:
                        print(f'⚠️ on_phone_call_update: {e}')
            else:
                print('⚠️ UpdatePhoneCall types not available — call hangup detection via pytgcalls only')

            @tg_client.on(events.NewMessage(incoming=True, func=lambda e: e.is_private))
            async def on_dm(event):
                sender = await event.get_sender()
                if not isinstance(sender, User) or sender.bot:
                    return
                tg_id    = str(sender.id)
                username = sender.username or ''
                phone    = getattr(sender, 'phone', None) or ''
                access_hash = str(sender.access_hash) if sender.access_hash else ''
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: ensure_conv(tg_id, username=username, phone=phone, access_hash=access_hash))
                # Deduplicate: skip if this exact telegram message was already saved
                msg_tg_id = event.id
                with db() as conn:
                    with conn.cursor() as c:
                        c.execute('SELECT 1 FROM messages WHERE tg_msg_id=%s AND tg_id=%s AND direction=%s', (msg_tg_id, tg_id, 'in'))
                        if c.fetchone():
                            return
                if event.text:          text = event.text
                elif event.photo:       text = '[📷 Photo]'
                elif event.document:    text = '[📎 Document]'
                elif event.sticker:     text = '[Sticker]'
                elif event.voice:       text = '[🎤 Voice]'
                else:                   text = '[Message]'
                await loop.run_in_executor(None, lambda: maybe_set_source(tg_id, text))
                await loop.run_in_executor(None, lambda: save_msg(tg_id, text, 'in', tg_msg_id=msg_tg_id))
                print(f'📨 {tg_id}: {text[:80]}')
                # Push to all connected CRM clients
                now_ts = datetime.now().isoformat()
                asyncio.create_task(ws_manager.broadcast({
                    'type': 'new_message',
                    'tg_id': tg_id,
                    'text': text,
                    'direction': 'in',
                    'timestamp': now_ts,
                    'tg_msg_id': msg_tg_id,
                }))
                asyncio.create_task(ws_manager.broadcast({
                    'type': 'notification',
                    'notif_type': 'message',
                    'tg_id': tg_id,
                    'text': text[:80],
                    'timestamp': now_ts,
                }))

            @tg_client.on(events.NewMessage(outgoing=True, func=lambda e: e.is_private))
            async def on_outgoing_dm(event):
                """Capture messages sent directly from Telegram (not via CRM /reply)."""
                tg_id = str(event.chat_id)
                msg_tg_id = event.id
                loop = asyncio.get_event_loop()
                # Only capture for known subscribers
                with db() as conn:
                    with conn.cursor() as c:
                        c.execute('SELECT 1 FROM conversations WHERE tg_id=%s', (tg_id,))
                        if not c.fetchone():
                            return
                # Deduplicate: skip if already saved (e.g. sent via /reply endpoint)
                with db() as conn:
                    with conn.cursor() as c:
                        c.execute('SELECT 1 FROM messages WHERE tg_msg_id=%s AND tg_id=%s AND direction=%s', (msg_tg_id, tg_id, 'out'))
                        if c.fetchone():
                            return
                if event.text:      text = event.text
                elif event.photo:   text = '[📷 Photo]'
                elif event.document:text = '[📎 Document]'
                elif event.voice:   text = '[🎤 Voice]'
                else:               text = '[Message]'
                await loop.run_in_executor(None, lambda: save_msg(tg_id, text, 'out', 'Telegram', tg_msg_id=msg_tg_id))
                print(f'📤 Telegram→CRM {tg_id}: {text[:80]}')
                asyncio.create_task(ws_manager.broadcast({
                    'type': 'new_message',
                    'tg_id': tg_id,
                    'text': text,
                    'direction': 'out',
                    'chatter': 'Telegram',
                    'timestamp': datetime.now().isoformat(),
                    'tg_msg_id': msg_tg_id,
                }))

            # ── Read receipts: sync unread when Telegram app is used ──────────
            @tg_client.on(events.MessageRead(inbox=True))
            async def on_inbox_read(event):
                """Fires when WE read incoming messages (in Telegram app or via CRM)."""
                try:
                    tg_id = str(event.chat_id)
                    with db() as conn:
                        with conn.cursor() as c:
                            c.execute('UPDATE conversations SET unread=0 WHERE tg_id=%s', (tg_id,))
                    asyncio.create_task(ws_manager.broadcast({
                        'type': 'read_update', 'tg_id': tg_id, 'unread': 0
                    }))
                except Exception as _e:
                    print(f'on_inbox_read error: {_e}')
            # ──────────────────────────────────────────────────────────────────

            await tg_client.start()
            _tg_client_ready = True
            print('✅ Userbot verbunden!')
            retry_delay = 5  # reset on success

            # ── Re-init pytgcalls on every connect cycle ──────────────────────
            # calls_client must be tied to the CURRENT tg_client instance.
            # After reconnect tg_client is a new object → old calls_client gives
            # MTProtoClientNotConnected. We (re-)init here every time we connect.
            if _PYTGCALLS_OK:
                asyncio.create_task(_reinit_calls(tg_client))
            # ─────────────────────────────────────────────────────────────────

            await tg_client.run_until_disconnected()
            _tg_client_ready = False
            print('⚠️  Userbot getrennt – reconnecting...')

        except Exception as e:
            print(f'❌ Userbot Fehler: {e} — retry in {retry_delay}s')

        if _userbot_running:
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)  # exponential backoff max 60s

# ── PAYPAL EMAIL POLLER ───────────────────────────────────────────────────────
# Detects "money received" emails from PayPal in a connected mailbox (IMAP) and
# pushes a real-time notification into the CRM. No sub-matching — chatter knows.
import imaplib as _imaplib
import email as _email
from email.header import decode_header as _decode_header

def _pp_decode(s):
    if not s:
        return ''
    try:
        out = ''
        for txt, enc in _decode_header(s):
            out += txt.decode(enc or 'utf-8', 'ignore') if isinstance(txt, bytes) else txt
        return out
    except Exception:
        return str(s)

def _pp_parse_amount(text: str):
    """Best-effort (amount_str, currency) from a PayPal subject line."""
    cur_map = {'€': 'EUR', 'EUR': 'EUR', '$': 'USD', 'USD': 'USD', '£': 'GBP', 'GBP': 'GBP'}
    m = re.search(r'(€|EUR|\$|USD|£|GBP)\s*([0-9][0-9.,]*[0-9])', text, re.I)
    if m:
        return m.group(2), cur_map.get(m.group(1).upper(), m.group(1))
    m = re.search(r'([0-9][0-9.,]*[0-9])\s*(€|EUR|\$|USD|£|GBP)', text, re.I)
    if m:
        return m.group(1), cur_map.get(m.group(2).upper(), m.group(2))
    return '', ''

def _paypal_imap_check():
    """Blocking: return list of payment-received mails as dicts (newest 30, last 3 days)."""
    host = os.environ.get('PAYPAL_IMAP_HOST', 'imap.gmail.com')
    user = os.environ.get('PAYPAL_IMAP_USER', '')
    pw = os.environ.get('PAYPAL_IMAP_PASS', '')
    folder = os.environ.get('PAYPAL_IMAP_FOLDER', 'INBOX')
    if not user or not pw:
        return []
    new_items, M = [], None
    try:
        M = _imaplib.IMAP4_SSL(host, timeout=20)
        M.login(user, pw)
        M.select(folder)
        since = (datetime.now() - timedelta(days=3)).strftime('%d-%b-%Y')
        typ, data = M.search(None, '(FROM "paypal" SINCE %s)' % since)
        if typ != 'OK' or not data or not data[0]:
            return []
        for num in data[0].split()[-30:]:
            typ, msgdata = M.fetch(num, '(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM MESSAGE-ID DATE)])')
            if typ != 'OK' or not msgdata or not msgdata[0]:
                continue
            hdr = _email.message_from_bytes(msgdata[0][1])
            subject = _pp_decode(hdr.get('Subject'))
            low = subject.lower()
            if not any(k in low for k in ('received', 'erhalten', 'sent you', 'gesendet',
                                          'payment', 'zahlung', 'geld')):
                continue
            msgid = (hdr.get('Message-ID') or hdr.get('Message-Id') or '').strip()
            uid = msgid or (subject + '|' + (hdr.get('Date') or ''))
            amount, currency = _pp_parse_amount(subject)
            new_items.append({'mail_uid': uid, 'amount': amount, 'currency': currency,
                              'sender': _pp_decode(hdr.get('From')), 'subject': subject})
    except Exception as e:
        print(f'PayPal IMAP error: {e}')
    finally:
        try:
            if M:
                M.logout()
        except Exception:
            pass
    return new_items

async def _run_paypal_poller():
    if os.environ.get('PAYPAL_IMAP_ENABLED', '').lower() not in ('1', 'true', 'yes', 'on'):
        print('ℹ️  PayPal-Mail-Poller aus (PAYPAL_IMAP_ENABLED nicht gesetzt)')
        return
    try:
        interval = int(os.environ.get('PAYPAL_POLL_SECONDS', '25'))
    except Exception:
        interval = 25
    print(f'✅ PayPal-Mail-Poller aktiv (alle {interval}s)')
    loop = asyncio.get_event_loop()
    await asyncio.sleep(8)
    while True:
        try:
            for it in await loop.run_in_executor(None, _paypal_imap_check):
                is_new = False
                try:
                    with db() as conn:
                        with conn.cursor() as c:
                            c.execute('SELECT 1 FROM paypal_notifications WHERE mail_uid=%s', (it['mail_uid'],))
                            if not c.fetchone():
                                c.execute('''INSERT INTO paypal_notifications
                                    (mail_uid, amount, currency, sender, subject, ts, seen)
                                    VALUES (%s,%s,%s,%s,%s,%s,0)''',
                                    (it['mail_uid'], it['amount'], it['currency'],
                                     it['sender'][:200], it['subject'][:300],
                                     datetime.now().isoformat()))
                                is_new = True
                except Exception as e:
                    print(f'PayPal store error: {e}')
                if is_new:
                    amt = (it['amount'] + ' ' + it['currency']).strip() or 'Betrag unbekannt'
                    await ws_manager.broadcast({
                        'type': 'notification', 'notif_type': 'paypal',
                        'amount': it['amount'], 'currency': it['currency'],
                        'text': f'Money received ({amt})',
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f'💰 PayPal received: {amt}')
        except Exception as e:
            print(f'PayPal poller loop error: {e}')
        await asyncio.sleep(interval)

# ── FASTAPI ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as _ie:
        # Non-fatal: tables may already exist from previous deployment.
        # Log and continue — userbot MUST still start.
        print(f'⚠️ init_db() warning (non-fatal): {_ie}')
    asyncio.create_task(start_userbot())
    asyncio.create_task(_run_scheduled_broadcasts())
    asyncio.create_task(_run_paypal_poller())
    # PyTgCalls is now initialized inside start_userbot() on every connect cycle
    # via _reinit_calls() — no separate _init_calls task needed.
    yield
    global _userbot_running
    _userbot_running = False
    if tg_client:
        await tg_client.disconnect()
    if calls_client:
        try:
            await calls_client.stop()
        except Exception:
            pass

app = FastAPI(title='Chatter CRM', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allow_headers=['*'],
    expose_headers=['*'],
    max_age=86400,
)

# ── SETUP ─────────────────────────────────────────────────────────────────────
SETUP_HTML = """<!DOCTYPE html><html lang="de"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chatter CRM – Setup</title><style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,sans-serif;background:#0f0f0f;color:#eee;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}.card{background:#1a1a1a;border:1px solid #333;border-radius:16px;padding:40px;width:100%;max-width:480px}h1{font-size:22px;margin-bottom:6px}.sub{color:#888;font-size:14px;margin-bottom:32px}label{display:block;font-size:13px;color:#aaa;margin-bottom:6px}input{width:100%;background:#111;border:1px solid #333;border-radius:8px;color:#eee;padding:12px 14px;font-size:15px;margin-bottom:18px;outline:none}input:focus{border-color:#6c63ff}button{width:100%;background:#6c63ff;color:#fff;border:none;border-radius:8px;padding:14px;font-size:16px;font-weight:600;cursor:pointer}.result{background:#111;border:1px solid #6c63ff;border-radius:8px;padding:16px;margin-top:24px;word-break:break-all;font-family:monospace;font-size:12px;color:#a0f0a0}.result h3{color:#6c63ff;margin-bottom:10px;font-size:14px}.copy-btn{background:#333;color:#eee;border:none;border-radius:6px;padding:8px 16px;cursor:pointer;font-size:13px;margin-top:12px;width:100%}.error{background:#2a1010;border:1px solid #f55;border-radius:8px;padding:14px;margin-top:16px;color:#f88;font-size:14px}.step{color:#6c63ff;font-size:13px;margin-bottom:20px}</style></head><body><div class="card"><h1>🔐 Chatter CRM Setup</h1><p class="sub">Session-String generieren</p>{CONTENT}</div><script>function copySession(){const el=document.getElementById('session-str');navigator.clipboard.writeText(el.innerText).then(()=>{document.getElementById('copy-btn').innerText='✓ Kopiert!';})}</script></body></html>"""
FORM_STEP1 = """<p class="step">Schritt 1 – API-Zugangsdaten</p><form method="POST" action="/setup/send-code"><label>API ID</label><input name="api_id" type="number" required><label>API Hash</label><input name="api_hash" type="text" required><label>Telefonnummer</label><input name="phone" type="text" placeholder="+49151..." required><button type="submit">SMS-Code anfordern →</button></form>"""

def render_setup(content, extra=''):
    return HTMLResponse(SETUP_HTML.replace('{CONTENT}', content + extra))

@app.get('/setup', response_class=HTMLResponse)
async def setup_get():
    return render_setup(FORM_STEP1)

@app.post('/setup/send-code', response_class=HTMLResponse)
async def setup_send_code(api_id: str = Form(...), api_hash: str = Form(...), phone: str = Form(...)):
    global setup_client, setup_phone
    try:
        setup_phone = phone.strip()
        setup_client = TelegramClient(StringSession(), int(api_id.strip()), api_hash.strip())
        await setup_client.connect()
        await setup_client.send_code_request(setup_phone)
        form2 = f"""<p class="step">Schritt 2 – SMS-Code</p><p style="color:#aaa;font-size:14px;margin-bottom:20px;">Code an <strong>{phone}</strong> gesendet.</p><form method="POST" action="/setup/verify"><label>SMS-Code</label><input name="code" type="text" required autofocus><button type="submit">Session generieren ✓</button></form>"""
        return render_setup(form2)
    except Exception as e:
        return render_setup(FORM_STEP1, f'<div class="error">❌ {e}</div>')

@app.post('/setup/verify', response_class=HTMLResponse)
async def setup_verify(code: str = Form(...)):
    global setup_client, setup_phone
    try:
        await setup_client.sign_in(setup_phone, code.strip())
        session_str = setup_client.session.save()
        await setup_client.disconnect()
        result = f"""<div class="result"><h3>✅ TG_SESSION:</h3><div id="session-str">{session_str}</div><button class="copy-btn" id="copy-btn" onclick="copySession()">📋 Kopieren</button></div><p style="color:#888;font-size:13px;margin-top:20px;">Railway → Variables → TG_SESSION einfügen.</p>"""
        return render_setup(result)
    except Exception as e:
        return render_setup(FORM_STEP1, f'<div class="error">❌ {e}</div>')

# ── HEALTH ─────────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/healthz')
def healthz():
    try:
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT COUNT(*) as n FROM conversations')
                n = c.fetchone()['n']
        return {'status': 'ok', 'ver': 'thumb-ffmpeg-v8', 'conversations': n, 'userbot': 'connected' if _tg_client_ready else 'disconnected', 'db': DB_PATH if USE_SQLITE else 'postgresql'}
    except Exception as e:
        return {'status': 'error', 'detail': str(e)}

@app.get('/status')
def status():
    return {'userbot': 'connected' if _tg_client_ready else 'disconnected', 'ws_clients': len(ws_manager._connections)}

@app.post('/admin/reconnect-userbot')
async def reconnect_userbot():
    """Force-restart the Telethon userbot (admin use only)."""
    global tg_client, _tg_client_ready
    if tg_client:
        try:
            await tg_client.disconnect()
        except Exception:
            pass
    _tg_client_ready = False
    asyncio.create_task(start_userbot())
    return {'status': 'reconnecting'}

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep alive — client sends pings, we just read them
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

# ── CONVERSATIONS ─────────────────────────────────────────────────────────────
@app.get('/conversations')
def get_conversations():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT tg_id,anon_id,internal_name,notes,last_msg,last_time,first_time,unread,msg_count,time_waster,is_muted,tg_username,tg_phone,followup_at,funnel_stage,call_followup_at,call_followup_note,is_online,last_seen FROM conversations ORDER BY last_time DESC NULLS LAST')
            rows = c.fetchall()
    return [dict(r) for r in rows]

@app.get('/online')
def get_online():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('''SELECT c.tg_id,c.anon_id,c.internal_name,c.last_seen,
                         COALESCE(SUM(s.amount),0) as total_spent
                         FROM conversations c
                         LEFT JOIN sales s USING(tg_id)
                         WHERE c.is_online=TRUE
                         GROUP BY c.tg_id,c.anon_id,c.internal_name,c.last_seen
                         ORDER BY c.last_seen DESC''')
            rows = c.fetchall()
    return [dict(r) for r in rows]

@app.get('/profile/{tg_id}')
def get_profile(tg_id: str):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT tg_id,anon_id,internal_name,notes,last_time,first_time,unread,msg_count,time_waster,tg_username,tg_phone,funnel_stage,call_followup_at,call_followup_note,payment_ref FROM conversations WHERE tg_id=%s', (tg_id,))
            row = c.fetchone()
    if not row:
        raise HTTPException(404, 'Not found')
    return dict(row)

class ProfileUpdate(BaseModel):
    internal_name: Optional[str] = None
    notes: Optional[str] = None
    time_waster: Optional[bool] = None
    is_muted: Optional[bool] = None
    funnel_stage: Optional[str] = None
    call_followup_at: Optional[str] = None
    call_followup_note: Optional[str] = None

@app.patch('/profile/{tg_id}')
def update_profile(tg_id: str, body: ProfileUpdate):
    with db() as conn:
        with conn.cursor() as c:
            if body.internal_name is not None:
                c.execute('UPDATE conversations SET internal_name=%s WHERE tg_id=%s', (body.internal_name, tg_id))
            if body.notes is not None:
                c.execute('UPDATE conversations SET notes=%s WHERE tg_id=%s', (body.notes, tg_id))
            if body.time_waster is not None:
                c.execute('UPDATE conversations SET time_waster=%s WHERE tg_id=%s', (body.time_waster, tg_id))
            if body.is_muted is not None:
                c.execute('UPDATE conversations SET is_muted=%s WHERE tg_id=%s', (body.is_muted, tg_id))
            if body.funnel_stage is not None:
                c.execute('UPDATE conversations SET funnel_stage=%s WHERE tg_id=%s', (body.funnel_stage, tg_id))
            if body.call_followup_at is not None:
                c.execute('UPDATE conversations SET call_followup_at=%s, call_followup_note=%s WHERE tg_id=%s',
                          (body.call_followup_at or None, body.call_followup_note or '', tg_id))
    return {'ok': True}

# ── ANALYTICS ─────────────────────────────────────────────────────────────────
@app.get('/analytics/my-stats')
def get_my_stats(chatter: str, period: str = 'heute'):
    """Personal stats for a single chatter filtered by period."""
    now = datetime.now()
    if period == 'heute':
        since = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == 'woche':
        since = (now - timedelta(days=7)).isoformat()
    elif period == 'monat':
        since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    else:
        since = '2000-01-01'
    with db() as conn:
        with conn.cursor() as c:
            # Sales
            c.execute('''SELECT COUNT(*) as sales_count, COALESCE(SUM(amount),0) as revenue,
                                COALESCE(AVG(amount),0) as avg_sale
                         FROM sales WHERE chatter=%s AND timestamp >= %s''', (chatter, since))
            sale_row = c.fetchone()
            # Messages sent
            c.execute('''SELECT COUNT(*) as msgs FROM messages
                         WHERE chatter=%s AND direction='out' AND timestamp >= %s''', (chatter, since))
            msg_row = c.fetchone()
            # Avg response time
            c.execute('''SELECT AVG(EXTRACT(EPOCH FROM (m_out.timestamp::timestamp - m_in.timestamp::timestamp))) as avg_resp
                         FROM messages m_in
                         JOIN LATERAL (
                           SELECT timestamp FROM messages m
                           WHERE m.tg_id=m_in.tg_id AND m.direction='out' AND m.chatter=%s
                             AND m.timestamp > m_in.timestamp
                             AND EXTRACT(EPOCH FROM (m.timestamp::timestamp - m_in.timestamp::timestamp)) BETWEEN 5 AND 3600
                           ORDER BY m.timestamp ASC LIMIT 1
                         ) m_out ON true
                         WHERE m_in.direction='in' AND m_in.timestamp >= %s''', (chatter, since))
            resp_row = c.fetchone()
            # Sales by product
            c.execute('''SELECT product, COUNT(*) as cnt, SUM(amount) as rev
                         FROM sales WHERE chatter=%s AND timestamp >= %s AND product != ''
                         GROUP BY product ORDER BY rev DESC''', (chatter, since))
            products = [dict(r) for r in c.fetchall()]
    return {
        'chatter': chatter,
        'period': period,
        'revenue': float(sale_row['revenue'] or 0),
        'sales_count': sale_row['sales_count'] or 0,
        'avg_sale': float(sale_row['avg_sale'] or 0),
        'msgs_sent': msg_row['msgs'] or 0,
        'avg_response_sec': round(float(resp_row['avg_resp'])) if resp_row and resp_row['avg_resp'] else None,
        'products': products,
    }

@app.get('/analytics/recent-chatted')
def get_recent_chatted(hours: int = 5):
    """Subscribers the userbot messaged in the last N hours."""
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute('''SELECT DISTINCT m.tg_id, c.anon_id, c.internal_name
                         FROM messages m
                         JOIN conversations c ON c.tg_id = m.tg_id
                         WHERE m.direction='out' AND m.timestamp >= %s''', (since,))
            rows = c.fetchall()
    return [dict(r) for r in rows]

@app.get('/analytics/chatters')
def get_chatter_analytics(period: str = 'alle'):
    now = datetime.now()
    if period == 'heute':
        since = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == 'woche':
        since = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == 'monat':
        since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    else:
        since = '2000-01-01'

    with db() as conn:
        with conn.cursor() as c:
            # Sales per chatter
            c.execute('''
                SELECT chatter,
                       COUNT(*) as sales_count,
                       COALESCE(SUM(amount),0) as total_revenue,
                       COALESCE(AVG(amount),0) as avg_sale,
                       COUNT(CASE WHEN LOWER(product) LIKE '%call%' THEN 1 END) as calls_count
                FROM sales WHERE chatter != '' AND timestamp >= %s
                GROUP BY chatter ORDER BY total_revenue DESC
            ''', (since,))
            sales_rows = {r['chatter']: dict(r) for r in c.fetchall()}

            # Messages sent + active time per chatter
            c.execute('''
                SELECT chatter,
                       COUNT(*) as msgs_sent,
                       MIN(timestamp) as first_msg,
                       MAX(timestamp) as last_msg
                FROM messages
                WHERE direction='out' AND chatter != '' AND timestamp >= %s
                GROUP BY chatter
            ''', (since,))
            msg_rows = {r['chatter']: dict(r) for r in c.fetchall()}

            # Average response time: time from incoming msg to next outgoing by chatter
            c.execute('''
                SELECT m_out.chatter,
                       AVG(EXTRACT(EPOCH FROM (m_out.timestamp::timestamp - m_in.timestamp::timestamp))) as avg_response_sec,
                       COUNT(*) as response_pairs
                FROM messages m_in
                JOIN LATERAL (
                    SELECT chatter, timestamp FROM messages m
                    WHERE m.tg_id = m_in.tg_id
                      AND m.direction = 'out'
                      AND m.chatter != ''
                      AND m.timestamp > m_in.timestamp
                      AND EXTRACT(EPOCH FROM (m.timestamp::timestamp - m_in.timestamp::timestamp)) BETWEEN 5 AND 3600
                    ORDER BY m.timestamp ASC LIMIT 1
                ) m_out ON true
                WHERE m_in.direction = 'in' AND m_in.timestamp >= %s
                GROUP BY m_out.chatter
            ''', (since,))
            response_rows = {r['chatter']: dict(r) for r in c.fetchall()}

    chatters = {}
    all_names = set(list(sales_rows.keys()) + list(msg_rows.keys()))
    for name in all_names:
        s = sales_rows.get(name, {'sales_count':0,'total_revenue':0,'avg_sale':0,'calls_count':0})
        m = msg_rows.get(name, {'msgs_sent':0,'first_msg':None,'last_msg':None})
        r = response_rows.get(name, {'avg_response_sec':None})

        # Active time in seconds (first to last message within period)
        active_sec = 0
        if m['first_msg'] and m['last_msg']:
            try:
                from datetime import datetime as _dt
                t1 = _dt.fromisoformat(str(m['first_msg']))
                t2 = _dt.fromisoformat(str(m['last_msg']))
                active_sec = max(0, int((t2 - t1).total_seconds()))
            except Exception:
                pass

        chatters[name] = {
            'chatter': name,
            'sales_count': s['sales_count'],
            'total_revenue': float(s['total_revenue']),
            'avg_sale': float(s['avg_sale']),
            'calls_count': s['calls_count'] or 0,
            'msgs_sent': m['msgs_sent'],
            'active_sec': active_sec,
            'avg_response_sec': round(float(r['avg_response_sec'])) if r.get('avg_response_sec') else None,
            'revenue_per_msg': round(float(s['total_revenue']) / m['msgs_sent'], 2) if m['msgs_sent'] > 0 else 0,
        }
    return sorted(chatters.values(), key=lambda x: x['total_revenue'], reverse=True)

async def _bg_read_history(tg_id: str):
    """Background: mark messages as read in Telegram."""
    if not tg_client or not tg_client.is_connected():
        return
    if active_calls:
        print('⏸ read-ack skipped — call in progress')
        return
    try:
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (tg_id,))
                row = c.fetchone()
        ah = int(row['tg_access_hash']) if row and row['tg_access_hash'] else 0
        peer = InputPeerUser(int(tg_id), ah) if ah else int(tg_id)
        await tg_client(functions.messages.ReadHistoryRequest(peer=peer, max_id=0))
    except Exception as e:
        print(f'ReadHistory skip: {e}')

@app.get('/messages/{tg_id}')
async def get_messages(tg_id: str, bg: BackgroundTasks):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('UPDATE conversations SET unread=0 WHERE tg_id=%s', (tg_id,))
            c.execute('SELECT id,text,direction,timestamp,chatter,is_read,read_at,translation,tg_msg_id FROM messages WHERE tg_id=%s ORDER BY timestamp', (tg_id,))
            rows = c.fetchall()
    bg.add_task(_bg_read_history, tg_id)
    return [dict(r) for r in rows]

@app.post('/typing/{tg_id}')
async def send_typing(tg_id: str):
    """Send typing indicator to subscriber so they see '...' in Telegram."""
    if not tg_client or not tg_client.is_connected():
        return {'ok': False}
    if active_calls:
        return {'ok': False, 'skipped': 'call_active'}
    try:
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (tg_id,))
                row = c.fetchone()
        ah = int(row['tg_access_hash']) if row and row['tg_access_hash'] else 0
        peer = InputPeerUser(int(tg_id), ah) if ah else int(tg_id)
        await tg_client(functions.messages.SetTypingRequest(
            peer=peer, action=types.SendMessageTypingAction()
        ))
        return {'ok': True}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}

# ── HOTKEYS (shared text shortcuts) ───────────────────────────────────────────
class HotkeysIn(BaseModel):
    hotkeys: str = '[]'

@app.get('/hotkeys')
def get_hotkeys():
    return {'hotkeys': get_setting('hotkeys_json', '[]')}

@app.post('/hotkeys')
def save_hotkeys(body: HotkeysIn):
    set_setting('hotkeys_json', body.hotkeys or '[]')
    return {'ok': True}

# ── DELETE / UNSEND a message (revoke = delete for everyone) ───────────────────
class MsgDeleteIn(BaseModel):
    tg_id: str
    tg_msg_id: int = 0
    db_id: int = 0

@app.post('/message/delete')
async def delete_message(body: MsgDeleteIn):
    # 1) Delete on Telegram for everyone when we have the Telegram message id
    if tg_client and tg_client.is_connected() and body.tg_msg_id:
        try:
            with db() as conn:
                with conn.cursor() as c:
                    c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (body.tg_id,))
                    row = c.fetchone()
            ah = int(row['tg_access_hash']) if row and row['tg_access_hash'] else 0
            peer = InputPeerUser(int(body.tg_id), ah) if ah else int(body.tg_id)
            await tg_client.delete_messages(peer, [body.tg_msg_id], revoke=True)
        except Exception as e:
            print(f'delete_messages error: {e}')
    # 2) Remove from the CRM database
    try:
        with db() as conn:
            with conn.cursor() as c:
                if body.db_id:
                    c.execute('DELETE FROM messages WHERE id=%s', (body.db_id,))
                elif body.tg_msg_id:
                    c.execute('DELETE FROM messages WHERE tg_id=%s AND tg_msg_id=%s', (body.tg_id, body.tg_msg_id))
    except Exception as e:
        print(f'DB delete error: {e}')
    asyncio.create_task(ws_manager.broadcast({
        'type': 'msg_deleted', 'tg_id': body.tg_id,
        'tg_msg_id': body.tg_msg_id, 'db_id': body.db_id,
    }))
    return {'ok': True}

# ── REPLY ─────────────────────────────────────────────────────────────────────
# ── LEAD SOURCE TRACKING ──────────────────────────────────────────────────────
DEFAULT_SOURCE_CODES = 'markt 1, markt 2, quoka, social media'

def _get_source_codes():
    raw = get_setting('source_codes', DEFAULT_SOURCE_CODES) or DEFAULT_SOURCE_CODES
    return [c.strip() for c in raw.replace('\n', ',').split(',') if c.strip()]

def detect_source(text, codes):
    """Return the matching source code if the (first) message carries one."""
    if not text or not codes:
        return ''
    t = text.strip().lower().lstrip('#').strip()
    for pref in ('start ', 'start:', 'ref ', 'ref-', 'ref:', 'src ', 'code '):
        if t.startswith(pref):
            t = t[len(pref):].strip()
            break
    tokens = t.split()
    first = ''.join(ch for ch in (tokens[0] if tokens else '') if ch.isalnum())
    compact = ''.join(ch for ch in t if ch.isalnum())
    for code in codes:
        cl = ''.join(ch for ch in code.lower() if ch.isalnum())
        if cl and (first == cl or compact.startswith(cl)):
            return code
    return ''

def maybe_set_source(tg_id, text):
    """On first contact, set the subscriber's source if the message carries a known code."""
    try:
        codes = _get_source_codes()
        if not codes:
            return
        code = detect_source(text, codes)
        if not code:
            return
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT source, msg_count FROM conversations WHERE tg_id=%s', (tg_id,))
                row = c.fetchone()
                if row and not (row['source'] or '').strip() and (row['msg_count'] or 0) <= 1:
                    c.execute('UPDATE conversations SET source=%s WHERE tg_id=%s', (code, tg_id))
                    print(f'🎯 Source set for {tg_id}: {code}')
    except Exception as e:
        print(f'maybe_set_source error: {e}')

class SourceCodesIn(BaseModel):
    codes: str = ''

@app.get('/sources/codes')
def get_source_codes():
    return {'codes': get_setting('source_codes', DEFAULT_SOURCE_CODES)}

@app.post('/sources/codes')
def save_source_codes(body: SourceCodesIn):
    set_setting('source_codes', body.codes or '')
    return {'ok': True}

@app.get('/analytics/sources')
def analytics_sources():
    # Daily numbers: new subs today + sales/revenue today, per source
    since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute('''SELECT COALESCE(NULLIF(c.source,''),'Direkt') as source,
                                COUNT(DISTINCT CASE WHEN c.first_time >= %s THEN c.tg_id END) as subs,
                                COUNT(DISTINCT CASE WHEN s.timestamp >= %s THEN s.tg_id END) as buyers,
                                COALESCE(SUM(CASE WHEN s.timestamp >= %s THEN s.amount ELSE 0 END),0) as revenue
                         FROM conversations c
                         LEFT JOIN sales s ON s.tg_id = c.tg_id
                         GROUP BY COALESCE(NULLIF(c.source,''),'Direkt')
                         ORDER BY subs DESC''', (since, since, since))
            rows = c.fetchall()
    return [dict(r) for r in rows]

@app.get('/analytics/today-total')
def analytics_today_total():
    """Team-wide revenue + sale count for today (for the admin shift-goal view)."""
    since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT COALESCE(SUM(amount),0) as revenue, COUNT(*) as sales FROM sales WHERE timestamp >= %s', (since,))
            row = c.fetchone()
    return {'revenue': float(row['revenue'] or 0), 'sales': row['sales'] or 0}

@app.post('/conversations/{tg_id}/unread')
async def set_conv_unread(tg_id: str):
    """Mark a conversation as unread again (so the chatter still picks it up)."""
    with db() as conn:
        with conn.cursor() as c:
            c.execute('UPDATE conversations SET unread=1 WHERE tg_id=%s', (tg_id,))
    try:
        await ws_manager.broadcast({'type': 'read_update', 'tg_id': tg_id, 'unread': 1})
    except Exception:
        pass
    return {'ok': True}

# ── REPLY ─────────────────────────────────────────────────────────────────────
class ReplyIn(BaseModel):
    tg_id: str
    text: str
    chatter: str = 'Chatter'

@app.post('/reply')
async def post_reply(body: ReplyIn, background_tasks: BackgroundTasks):
    if not tg_client or not tg_client.is_connected():
        raise HTTPException(503, 'Userbot nicht verbunden')
    # Look up access_hash from DB (no extra Telegram roundtrip needed)
    peer_int = int(body.tg_id)
    access_hash = 0
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (body.tg_id,))
            row = c.fetchone()
            if row and row['tg_access_hash']:
                access_hash = int(row['tg_access_hash'])
    peer = InputPeerUser(peer_int, access_hash) if access_hash else peer_int
    # Fire-and-forget: send in background so reply returns instantly
    background_tasks.add_task(_send_reply_bg, body, peer)
    return {'ok': True}

async def _send_reply_bg(body: ReplyIn, peer):
    """Send a chat reply in the background — keeps /reply endpoint instant."""
    global _tg_priority
    _tg_priority += 1  # pause broadcast while this send is in flight
    try:
        sent_msg = await tg_client.send_message(peer, body.text)
        tg_msg_id = sent_msg.id
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: save_msg(body.tg_id, body.text, 'out', body.chatter, tg_msg_id))
        asyncio.create_task(ws_manager.broadcast({
            'type': 'new_message',
            'tg_id': body.tg_id,
            'text': body.text,
            'direction': 'out',
            'chatter': body.chatter,
            'timestamp': datetime.now().isoformat(),
            'tg_msg_id': tg_msg_id,
        }))
    except FloodWaitError as e:
        print(f'⏳ /reply FloodWait {e.seconds}s for {body.tg_id}')
    except Exception as e:
        print(f'❌ /reply bg error for {body.tg_id}: {e}')
    finally:
        _tg_priority = max(0, _tg_priority - 1)  # resume broadcast

# ── AUTH & USERS ─────────────────────────────────────────────────────────────
import hashlib as _hashlib

def _hash_pw(pw: str) -> str:
    return _hashlib.sha256(pw.encode()).hexdigest()

class LoginIn(BaseModel):
    username: str
    password: str

@app.post('/auth/login')
def auth_login(body: LoginIn):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT id,username,email,role,display_name FROM crm_users WHERE username=%s AND password_hash=%s',
                      (body.username, _hash_pw(body.password)))
            row = c.fetchone()
    if not row:
        raise HTTPException(401, 'Falscher Benutzername oder Passwort')
    d = dict(row)
    d['display_name'] = d['display_name'] or d['username']
    return d

@app.get('/auth/users')
def list_users():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT id,username,email,role,display_name FROM crm_users ORDER BY id')
            rows = c.fetchall()
    result = [dict(r) for r in rows]
    for r in result:
        r['display_name'] = r['display_name'] or r['username']
    return result

class UserCreate(BaseModel):
    username: str
    display_name: str = ''
    email: str = ''
    password: str
    role: str = 'chatter'

@app.post('/auth/users')
def create_user(body: UserCreate):
    try:
        dn = body.display_name.strip() or body.username.strip()
        with db() as conn:
            with conn.cursor() as c:
                c.execute(
                    'INSERT INTO crm_users (username,email,password_hash,role,display_name) VALUES (%s,%s,%s,%s,%s) RETURNING id',
                    (body.username.strip(), body.email.strip(), _hash_pw(body.password), body.role, dn)
                )
                new_id = c.fetchone()['id']
        return {'ok': True, 'id': new_id}
    except Exception as e:
        raise HTTPException(400, f'Fehler: {e}')

class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None

@app.patch('/auth/users/{user_id}')
def update_user(user_id: int, body: UserUpdate):
    with db() as conn:
        with conn.cursor() as c:
            if body.display_name is not None:
                c.execute('UPDATE crm_users SET display_name=%s WHERE id=%s', (body.display_name, user_id))
            if body.email is not None:
                c.execute('UPDATE crm_users SET email=%s WHERE id=%s', (body.email, user_id))
            if body.password:
                c.execute('UPDATE crm_users SET password_hash=%s WHERE id=%s', (_hash_pw(body.password), user_id))
            if body.role is not None:
                c.execute('UPDATE crm_users SET role=%s WHERE id=%s', (body.role, user_id))
    return {'ok': True}

@app.delete('/auth/users/{user_id}')
def delete_user(user_id: int):
    with db() as conn:
        with conn.cursor() as c:
            # Prevent deleting last admin
            c.execute("SELECT COUNT(*) as n FROM crm_users WHERE role='admin'")
            if c.fetchone()['n'] <= 1:
                c.execute("SELECT role FROM crm_users WHERE id=%s", (user_id,))
                row = c.fetchone()
                if row and row['role'] == 'admin':
                    raise HTTPException(400, 'Letzter Admin kann nicht gelöscht werden')
            c.execute('DELETE FROM crm_users WHERE id=%s', (user_id,))
    return {'ok': True}

# ── CRM SETTINGS ─────────────────────────────────────────────────────────────
def get_setting(key: str, default: str = '') -> str:
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT value FROM crm_settings WHERE key=%s', (key,))
            row = c.fetchone()
    return row['value'] if row else default

def set_setting(key: str, value: str):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('INSERT INTO crm_settings (key,value) VALUES (%s,%s) ON CONFLICT (key) DO UPDATE SET value=%s',
                      (key, value, value))

@app.get('/settings/crm')
def get_crm_settings():
    keys = ['auto_online_enabled','auto_online_text','auto_online_cooldown_h','auto_online_stages','shift_goal','noones_key','noones_secret']
    result = {}
    with db() as conn:
        with conn.cursor() as c:
            for k in keys:
                c.execute('SELECT value FROM crm_settings WHERE key=%s', (k,))
                row = c.fetchone()
                result[k] = row['value'] if row else ''
    return result

class SettingsUpdate(BaseModel):
    auto_online_enabled: Optional[str] = None
    auto_online_text: Optional[str] = None
    auto_online_cooldown_h: Optional[str] = None
    auto_online_stages: Optional[str] = None
    shift_goal: Optional[str] = None
    noones_key: Optional[str] = None
    noones_secret: Optional[str] = None

@app.post('/settings/crm')
def save_crm_settings(body: SettingsUpdate):
    for k, v in body.dict().items():
        if v is not None:
            set_setting(k, v)
    return {'ok': True}

# ── AI ENDPOINTS ─────────────────────────────────────────────────────────────
import urllib.request as _urllib_req
import json as _json

def _openai_chat(messages: list, max_tokens: int = 300, temperature: float = 0.8) -> str:
    """Call OpenAI chat completions. Returns content string or raises."""
    if not OPENAI_API_KEY:
        raise HTTPException(503, 'OPENAI_API_KEY nicht gesetzt')
    payload = _json.dumps({
        'model': 'gpt-4o-mini',
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
    }).encode()
    req = _urllib_req.Request(
        'https://api.openai.com/v1/chat/completions',
        data=payload,
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_API_KEY}'}
    )
    with _urllib_req.urlopen(req, timeout=15) as resp:
        data = _json.loads(resp.read())
    return data['choices'][0]['message']['content'].strip()

class TranslateIn(BaseModel):
    text: str

@app.post('/ai/translate')
async def ai_translate(body: TranslateIn):
    """Translate incoming subscriber message to English."""
    if not body.text.strip():
        return {'ok': False, 'result': ''}
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _openai_chat([
            {'role': 'system', 'content': 'Translate the following message to English. Return ONLY the translation, nothing else.'},
            {'role': 'user', 'content': body.text}
        ], max_tokens=200, temperature=0.3))
        return {'ok': True, 'result': result}
    except Exception as e:
        raise HTTPException(500, str(e))

class StyleIn(BaseModel):
    text: str
    context: Optional[list] = None  # [{role, content}, ...] last few messages

@app.post('/ai/style')
async def ai_style(body: StyleIn):
    """Translate chatter's English text into German in Marie's natural style."""
    if not body.text.strip():
        return {'ok': False, 'result': ''}

    system_prompt = """Du bist ein Übersetzer für Marie, eine 18-jährige Deutsche die mit Fans auf Telegram chattet.

DEINE AUFGABE:
Ein Chatter schreibt auf ENGLISCH was Marie sagen soll.
Übersetze den Text ins Deutsche — so wie Marie es schreiben würde.

ABSOLUT WICHTIG:
✅ Übersetze NUR den Text des Chatters
✅ Gib NUR den deutschen Satz zurück — nichts anderes
❌ Antworte NICHT auf den Fan
❌ Füge KEINE eigenen Gedanken hinzu
❌ Füge KEINE Kommentare hinzu
❌ Stelle KEINE eigenen Fragen die der Chatter nicht gestellt hat

MARIES STIL (beim Übersetzen beachten):
- Alles kleinschreiben
- Locker, natürlich, echte Chat-Sprache
- Text-Smileys bevorzugen: :) ^^ <33 ;) — echte Emojis nur selten
- Filler-Wörter wenn passend: "irgendwie", "voll", "halt", "ne", "eigentlich"
- Nie formell, nie steif

BEISPIELE:
EN: "how are you?" → DE: "wie geht's dir so? :)"
EN: "I missed you" → DE: "ich hab dich vermisst :("
EN: "you look good" → DE: "du siehst gut aus ;)"
EN: "want to do a call?" → DE: "wollen wir einen call machen?"
EN: "I can show you more" → DE: "ich kann dir noch mehr zeigen ;)"
EN: "that's cute" → DE: "das ist süß ^^"
EN: "tell me about yourself" → DE: "erzähl mir von dir :)"
EN: "I'm busy right now" → DE: "bin gerade beschäftigt"

Gib NUR die deutsche Übersetzung zurück — einen Satz, fertig."""

    user_msg = f'Übersetze: "{body.text}"'

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.append({'role': 'user', 'content': user_msg})


    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _openai_chat(messages, max_tokens=120, temperature=0.5))
        return {'ok': True, 'result': result}
    except Exception as e:
        raise HTTPException(500, str(e))

class SuggestIn(BaseModel):
    tg_id: str
    context: Optional[list] = None  # last messages [{role,content}]

@app.post('/ai/suggest')
async def ai_suggest(body: SuggestIn):
    """Generate 3 reply suggestions for the current conversation."""
    # Get last messages from DB for context
    with db() as conn:
        with conn.cursor() as c:
            c.execute("SELECT text,direction FROM messages WHERE tg_id=%s ORDER BY timestamp DESC LIMIT 10", (body.tg_id,))
            recent = list(reversed(c.fetchall()))
            c.execute("SELECT internal_name,notes,funnel_stage FROM conversations WHERE tg_id=%s", (body.tg_id,))
            conv = c.fetchone()

    context_msgs = [
        {'role': 'user' if m['direction']=='in' else 'assistant', 'content': m['text']}
        for m in recent if not m['text'].startswith('[')
    ]

    stage_hint = ''
    if conv:
        stage_map = {'kalt':'Cold lead (first contact)','warm':'Warm (engaged)','hot':'Hot lead','angebot':'Offer was made','gebucht':'Call booked','done':'Deal closed'}
        stage_hint = f"\nCurrent funnel stage: {stage_map.get(conv['funnel_stage'] or 'kalt','Unknown')}"
        if conv['notes']: stage_hint += f"\nNotes about this fan: {conv['notes']}"

    system_prompt = f"""You are helping a chatter respond as Marie — an 18-year-old German girl selling video calls on Telegram.
The chatter speaks English and needs reply suggestions IN ENGLISH so they can read them.{stage_hint}

Generate 3 short English reply options that Marie would say to this fan.
Keep Marie's personality: casual, flirty, short (1-2 sentences), genuine chat language.
Occasionally ask a question, occasionally be playful or direct.

Return EXACTLY this JSON format (nothing else, no explanation):
["option 1", "option 2", "option 3"]

The 3 options should have different vibes:
1. Warm/curious — ask the fan something
2. Flirty/direct
3. Short and simple"""

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(context_msgs[-8:])
    messages.append({'role': 'user', 'content': 'Generate 3 reply options as a JSON array.'})

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _openai_chat(messages, max_tokens=300, temperature=0.9))
        import json as _json2
        suggestions = _json2.loads(result)
        if isinstance(suggestions, list):
            return {'ok': True, 'suggestions': suggestions[:3]}
        return {'ok': False, 'suggestions': []}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── NOTIFICATIONS FEED ───────────────────────────────────────────────────────
@app.get('/notifications')
def get_notifications(limit: int = 80):
    """Combined activity feed: new subs, messages, sales."""
    events = []
    with db() as conn:
        with conn.cursor() as c:
            # New subscribers
            c.execute('''SELECT tg_id, anon_id, internal_name, tg_username, first_time as ts
                         FROM conversations ORDER BY first_time DESC LIMIT %s''', (limit,))
            for r in c.fetchall():
                events.append({'type': 'new_sub', 'ts': str(r['ts']),
                               'tg_id': r['tg_id'], 'anon_id': r['anon_id'],
                               'name': r['internal_name'] or r['anon_id'],
                               'username': r['tg_username'] or ''})
            # Incoming messages (last N)
            c.execute('''SELECT m.tg_id, m.text, m.timestamp as ts,
                                c.anon_id, c.internal_name
                         FROM messages m
                         JOIN conversations c ON c.tg_id = m.tg_id
                         WHERE m.direction='in'
                         ORDER BY m.timestamp DESC LIMIT %s''', (limit,))
            for r in c.fetchall():
                events.append({'type': 'message', 'ts': str(r['ts']),
                               'tg_id': r['tg_id'], 'anon_id': r['anon_id'],
                               'name': r['internal_name'] or r['anon_id'],
                               'text': r['text'][:80]})
            # Sales
            c.execute('''SELECT s.tg_id, s.amount, s.product, s.chatter, s.timestamp as ts,
                                c.anon_id, c.internal_name
                         FROM sales s
                         JOIN conversations c ON c.tg_id = s.tg_id
                         ORDER BY s.timestamp DESC LIMIT %s''', (limit,))
            for r in c.fetchall():
                events.append({'type': 'sale', 'ts': str(r['ts']),
                               'tg_id': r['tg_id'], 'anon_id': r['anon_id'],
                               'name': r['internal_name'] or r['anon_id'],
                               'amount': float(r['amount']), 'product': r['product'] or '',
                               'chatter': r['chatter'] or ''})
            # PayPal payments (parsed from email)
            try:
                c.execute('''SELECT amount, currency, sender, subject, ts
                             FROM paypal_notifications ORDER BY ts DESC LIMIT %s''', (limit,))
                for r in c.fetchall():
                    amt = (str(r['amount'] or '') + ' ' + str(r['currency'] or '')).strip()
                    events.append({'type': 'paypal', 'ts': str(r['ts']),
                                   'amount': r['amount'] or '', 'currency': r['currency'] or '',
                                   'text': f"Money received ({amt or 'Betrag unbekannt'})",
                                   'sender': r['sender'] or ''})
            except Exception as _pe:
                print(f'paypal notif feed error: {_pe}')
    # Sort all events by timestamp desc, return top N
    events.sort(key=lambda x: x['ts'], reverse=True)
    return events[:limit]

# ── VAULT ────────────────────────────────────────────────────────────────────
ALLOWED_TYPES = {
    'image/jpeg','image/jpg','image/png','image/gif','image/webp',
    'image/heic','image/heif',                          # iOS formats
    'image/x-adobe-dng','image/dng','image/x-raw',      # RAW / DNG
    'image/tiff','image/x-tiff',                        # TIFF
    'video/mp4','video/quicktime','video/x-matroska','video/avi','video/x-msvideo',
}

def _vault_safe(name: str) -> str:
    """Sanitize filename: only alphanumeric, dots, dashes, underscores. Spaces → underscore."""
    import re as _re
    name = _re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    name = _re.sub(r'_+', '_', name)   # collapse multiple underscores
    name = name.strip('_')              # remove leading/trailing underscores
    return name or 'file'

def _vault_file_info(fpath: str, relpath: str, folder: str) -> dict:
    stat = os.stat(fpath)
    ext = relpath.rsplit('.', 1)[-1].lower() if '.' in relpath else ''
    ftype = 'video' if ext in ('mp4','mov','mkv','avi') else 'image'
    return {
        'name': relpath,      # e.g. "Folder/file.jpg" or "file.jpg"
        'folder': folder,
        'size': stat.st_size,
        'type': ftype,
        'url': f'/vault/file/{relpath}',
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }

@app.get('/vault/folders')
def vault_folders():
    """List all vault folders (subdirectories)."""
    folders = []
    try:
        for name in sorted(os.listdir(VAULT_DIR)):
            if os.path.isdir(os.path.join(VAULT_DIR, name)) and not name.startswith('.'):
                folders.append(name)
    except Exception as e:
        print(f'Vault folders error: {e}')
    return folders

class VaultFolderIn(BaseModel):
    name: str

@app.post('/vault/folder')
def vault_create_folder(body: VaultFolderIn):
    """Create a new vault folder."""
    safe = _vault_safe(body.name).strip()
    if not safe:
        raise HTTPException(400, 'Invalid folder name')
    fpath = os.path.join(VAULT_DIR, safe)
    os.makedirs(fpath, exist_ok=True)
    return {'ok': True, 'name': safe}

@app.delete('/vault/folder/{folder}')
def vault_delete_folder(folder: str):
    """Delete a vault folder and all its files."""
    if '..' in folder:
        raise HTTPException(400, 'Invalid folder name')
    fpath = os.path.join(VAULT_DIR, folder)
    if not os.path.isdir(fpath):
        raise HTTPException(404, 'Folder not found')
    shutil.rmtree(fpath)
    return {'ok': True}

@app.get('/vault')
def vault_list(folder: Optional[str] = None):
    """List vault files. Optional ?folder=X to filter by folder."""
    files = []
    try:
        if folder:
            # List files in specific folder — use raw name, check both original and safe version
            dir_path = os.path.join(VAULT_DIR, folder)
            if not os.path.isdir(dir_path):
                dir_path = os.path.join(VAULT_DIR, _vault_safe(folder))
            safe_folder = os.path.basename(dir_path)
            if os.path.isdir(dir_path):
                for fname in sorted(os.listdir(dir_path)):
                    fpath = os.path.join(dir_path, fname)
                    if os.path.isfile(fpath):
                        relpath = f'{safe_folder}/{fname}'
                        files.append(_vault_file_info(fpath, relpath, safe_folder))
        else:
            # List ALL files (root + all subfolders)
            for item in sorted(os.listdir(VAULT_DIR)):
                ipath = os.path.join(VAULT_DIR, item)
                if os.path.isfile(ipath):
                    files.append(_vault_file_info(ipath, item, ''))
                elif os.path.isdir(ipath) and not item.startswith('.'):
                    for fname in sorted(os.listdir(ipath)):
                        fpath = os.path.join(ipath, fname)
                        if os.path.isfile(fpath):
                            relpath = f'{item}/{fname}'
                            files.append(_vault_file_info(fpath, relpath, item))
    except Exception as e:
        print(f'Vault list error: {e}')
    return files

@app.post('/vault/upload')
async def vault_upload(file: UploadFile = File(...), folder: str = ''):
    """Upload a file to the vault, optionally into a folder."""
    # Accept common types; also allow empty content-type (some clients omit it)
    ct = (file.content_type or '').split(';')[0].strip().lower()
    if ct and ct not in ALLOWED_TYPES and ct != 'application/octet-stream':
        raise HTTPException(400, f'File type not allowed: {ct}')
    # Sanitize filename only (keep the original folder name to avoid breaking existing folders)
    safe_name = _vault_safe(file.filename or 'file') or 'file'
    if folder:
        # Use folder name as-is if it already exists, otherwise sanitize for new folders
        existing = os.path.join(VAULT_DIR, folder)
        if os.path.isdir(existing):
            target_dir = existing
            safe_folder = folder          # preserve original name (may have spaces)
        else:
            safe_folder = _vault_safe(folder)
            target_dir = os.path.join(VAULT_DIR, safe_folder)
            os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = VAULT_DIR
        safe_folder = ''
    fpath = os.path.join(target_dir, safe_name)
    if os.path.exists(fpath):
        base, ext = os.path.splitext(safe_name)
        safe_name = f'{base}_{int(datetime.now().timestamp())}{ext}'
        fpath = os.path.join(target_dir, safe_name)
    try:
        with open(fpath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except OSError as e:
        print(f'Vault write error: {e}')
        if 'No space left' in str(e):
            raise HTTPException(507, 'Disk full — Railway volume is out of space. Delete old files or expand volume.')
        raise HTTPException(500, f'Write error: {e}')
    relpath = f'{safe_folder}/{safe_name}' if safe_folder else safe_name
    return {'ok': True, 'name': relpath, 'url': f'/vault/file/{relpath}'}

@app.get('/vault/file/{filepath:path}')
def vault_serve(filepath: str):
    """Serve a vault file (supports folder/filename paths)."""
    if '..' in filepath:
        raise HTTPException(400, 'Invalid path')
    fpath = os.path.join(VAULT_DIR, filepath)
    if not os.path.isfile(fpath):
        raise HTTPException(404, 'Not found')
    return FileResponse(fpath)

@app.get('/vault/thumb/{filepath:path}')
def vault_thumb(filepath: str):
    """Small cached thumbnail for fast, clean picker previews.
    Images -> resized JPEG (Pillow). Videos -> first frame (ffmpeg). Falls back to original."""
    if '..' in filepath:
        raise HTTPException(400, 'Invalid path')
    src = os.path.join(VAULT_DIR, filepath)
    if not os.path.isfile(src):
        raise HTTPException(404, 'Not found')
    ext = filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else ''
    THUMBS_DIR = os.path.join(VAULT_DIR, '_thumbs')
    os.makedirs(THUMBS_DIR, exist_ok=True)
    key = filepath.replace('/', '__').replace('\\', '__')
    thumb_path = os.path.join(THUMBS_DIR, key + '.jpg')
    try:
        if os.path.isfile(thumb_path) and os.path.getmtime(thumb_path) >= os.path.getmtime(src):
            return FileResponse(thumb_path, media_type='image/jpeg')
    except Exception:
        pass
    # Generate a small JPEG via ffmpeg for BOTH images and videos (no Pillow dependency).
    # On any failure return 404 — NEVER the full original (that would blow up browser memory).
    video_exts = ('mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v')
    try:
        import subprocess
        cmd = ['ffmpeg', '-y']
        if ext in video_exts:
            cmd += ['-ss', '00:00:01']
        cmd += ['-i', src, '-vframes', '1', '-vf', "scale='min(360,iw)':-2", thumb_path]
        subprocess.run(cmd, capture_output=True, timeout=25)
        if os.path.isfile(thumb_path) and os.path.getsize(thumb_path) > 0:
            return FileResponse(thumb_path, media_type='image/jpeg')
    except Exception as e:
        print(f'thumb {filepath}: {e}')
    raise HTTPException(404, 'no thumb')

@app.delete('/vault/file/{filepath:path}')
def vault_delete(filepath: str):
    """Delete a vault file."""
    if '..' in filepath:
        raise HTTPException(400, 'Invalid path')
    fpath = os.path.join(VAULT_DIR, filepath)
    if not os.path.isfile(fpath):
        raise HTTPException(404, 'Not found')
    os.remove(fpath)
    return {'ok': True}

class VaultSendIn(BaseModel):
    tg_id: str
    filename: str   # can be "folder/file.jpg" or "file.jpg"
    caption: str = ''

@app.post('/vault/send')
async def vault_send(body: VaultSendIn):
    """Queue a vault file for sending. Returns instantly; the upload runs in the background
    so large images/videos don't cause the request to time out."""
    if not tg_client or not tg_client.is_connected():
        raise HTTPException(503, 'Userbot nicht verbunden')
    if '..' in body.filename:
        raise HTTPException(400, 'Invalid path')
    fpath = os.path.join(VAULT_DIR, body.filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, 'File not found in vault')
    asyncio.create_task(_send_vault_file_bg(body, fpath))
    return {'ok': True}

VIDEO_EXTS = ('mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v')

def _ffprobe_video_meta(fpath: str):
    """Best-effort (duration_s, width, height) via ffprobe. Returns (0,0,0) on failure."""
    try:
        import subprocess, json as _json
        out = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', fpath],
            capture_output=True, timeout=20)
        data = _json.loads(out.stdout.decode('utf-8', 'ignore') or '{}')
        dur, w, h = 0, 0, 0
        for s in data.get('streams', []):
            if s.get('codec_type') == 'video':
                w = int(s.get('width') or 0)
                h = int(s.get('height') or 0)
                try:
                    dur = int(float(s.get('duration') or 0))
                except Exception:
                    pass
                break
        if not dur:
            try:
                dur = int(float(data.get('format', {}).get('duration') or 0))
            except Exception:
                pass
        return dur, w, h
    except Exception as e:
        print(f'ffprobe meta error: {e}')
        return 0, 0, 0

def _make_video_thumb(fpath: str):
    """Generate a JPEG thumbnail from a video via ffmpeg. Returns path or None."""
    try:
        import subprocess
        thumb = fpath + '.thumb.jpg'
        subprocess.run(['ffmpeg', '-y', '-ss', '00:00:01', '-i', fpath, '-vframes', '1',
                        '-vf', "scale='min(320,iw)':-2", thumb],
                       capture_output=True, timeout=25)
        if os.path.isfile(thumb) and os.path.getsize(thumb) > 0:
            return thumb
    except Exception as e:
        print(f'video thumb error: {e}')
    return None

async def _send_vault_file_bg(body: VaultSendIn, fpath: str):
    global _tg_priority
    _tg_priority += 1  # pause broadcast while uploading
    thumb_to_clean = None
    try:
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (body.tg_id,))
                row = c.fetchone()
        ah = int(row['tg_access_hash']) if row and row['tg_access_hash'] else 0
        peer = InputPeerUser(int(body.tg_id), ah) if ah else int(body.tg_id)

        # Build send kwargs — videos need explicit attributes + thumbnail + streaming,
        # otherwise Telegram often rejects them or they never upload.
        ext = body.filename.rsplit('.', 1)[-1].lower() if '.' in body.filename else ''
        send_kwargs = {'caption': body.caption or None}
        loop = asyncio.get_event_loop()
        if ext in VIDEO_EXTS:
            dur, w, h = await loop.run_in_executor(None, _ffprobe_video_meta, fpath)
            thumb_to_clean = await loop.run_in_executor(None, _make_video_thumb, fpath)
            send_kwargs['supports_streaming'] = True
            send_kwargs['force_document'] = False
            if w > 0 and h > 0:
                send_kwargs['attributes'] = [types.DocumentAttributeVideo(
                    duration=dur or 0, w=w, h=h, supports_streaming=True)]
            if thumb_to_clean:
                send_kwargs['thumb'] = thumb_to_clean
        try:
            await tg_client.send_file(peer, fpath, **send_kwargs)
        except Exception as send_err:
            # Last-resort fallback: deliver as a plain document so it never silently fails.
            print(f'⚠️  video/file send failed ({send_err}); retrying as document')
            await tg_client.send_file(peer, fpath, caption=body.caption or None,
                                      force_document=True)
        display_name = body.filename.split('/')[-1]
        save_msg(body.tg_id, f'[📎 {display_name}]', 'out', 'Vault')
        await ws_manager.broadcast({
            'type': 'new_message', 'tg_id': body.tg_id,
            'text': f'[📎 {display_name}]', 'direction': 'out',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f'❌ vault send error for {body.tg_id}: {e}')
        try:
            await ws_manager.broadcast({
                'type': 'notification', 'notif_type': 'error', 'tg_id': body.tg_id,
                'text': f'Vault-Senden fehlgeschlagen: {e}', 'timestamp': datetime.now().isoformat()
            })
        except Exception:
            pass
    finally:
        if thumb_to_clean:
            try:
                os.remove(thumb_to_clean)
            except Exception:
                pass
        _tg_priority = max(0, _tg_priority - 1)

# ── PLEDGES ──────────────────────────────────────────────────────────────────

class PledgeIn(BaseModel):
    tg_id: str
    anon_id: str
    amount: float
    payment_method: str = ''
    notes: str = ''
    chatter: str = ''
    deadline: str  # ISO date string e.g. "2026-06-10"

@app.post('/pledges')
def create_pledge(body: PledgeIn):
    ts = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                'INSERT INTO pledges (tg_id,anon_id,amount,payment_method,notes,chatter,deadline,status,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id',
                (body.tg_id, body.anon_id, body.amount, body.payment_method,
                 body.notes, body.chatter, body.deadline, 'open', ts)
            )
            pid = c.fetchone()['id']
    return {'ok': True, 'id': pid}

@app.get('/pledges')
def get_pledges(status: str = ''):
    with db() as conn:
        with conn.cursor() as c:
            if status:
                c.execute('SELECT * FROM pledges WHERE status=%s ORDER BY deadline ASC', (status,))
            else:
                c.execute("SELECT * FROM pledges WHERE status != 'cancelled' ORDER BY deadline ASC")
            rows = c.fetchall()
    now = datetime.now().isoformat()[:10]
    result = []
    for r in rows:
        d = dict(r)
        d['overdue'] = d['status'] == 'open' and d['deadline'] < now
        result.append(d)
    return result

@app.get('/pledges/summary')
def get_pledges_summary():
    with db() as conn:
        with conn.cursor() as c:
            c.execute("SELECT status, SUM(amount) as total, COUNT(*) as count FROM pledges GROUP BY status")
            rows = c.fetchall()
    now = datetime.now().isoformat()[:10]
    summary = {r['status']: {'total': float(r['total'] or 0), 'count': r['count']} for r in rows}
    # Also get overdue count
    with db() as conn:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) as n, SUM(amount) as t FROM pledges WHERE status='open' AND deadline < %s", (now,))
            row = c.fetchone()
    summary['overdue'] = {'count': row['n'], 'total': float(row['t'] or 0)}
    return summary

@app.get('/pledges/{tg_id}')
def get_subscriber_pledges(tg_id: str):
    with db() as conn:
        with conn.cursor() as c:
            c.execute("SELECT * FROM pledges WHERE tg_id=%s AND status != 'cancelled' ORDER BY deadline DESC", (tg_id,))
            rows = c.fetchall()
    now = datetime.now().isoformat()[:10]
    result = []
    for r in rows:
        d = dict(r)
        d['overdue'] = d['status'] == 'open' and d['deadline'] < now
        result.append(d)
    return result

@app.post('/pledges/{pledge_id}/pay')
async def mark_pledge_paid(pledge_id: int):
    """Mark pledge as paid and auto-create an approved sale."""
    ts = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT * FROM pledges WHERE id=%s', (pledge_id,))
            p = c.fetchone()
    if not p:
        raise HTTPException(404, 'Pledge not found')
    # Create approved sale
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO sales (tg_id,anon_id,amount,product,notes,chatter,timestamp,status,payment_method) VALUES (%s,%s,%s,%s,%s,%s,%s,'approved',%s) RETURNING id",
                (p['tg_id'], p['anon_id'], p['amount'],
                 p['payment_method'] or 'Pledge',
                 f'Pledge #{pledge_id}: {p["notes"]}',
                 p['chatter'], ts, p['payment_method'])
            )
            c.execute("UPDATE pledges SET status='paid', paid_at=%s WHERE id=%s", (ts, pledge_id))
    asyncio.create_task(ws_manager.broadcast({
        'type': 'notification', 'notif_type': 'sale',
        'tg_id': p['tg_id'], 'anon_id': p['anon_id'],
        'amount': p['amount'], 'product': 'Pledge paid',
        'chatter': p['chatter'], 'timestamp': ts,
    }))
    return {'ok': True}

@app.post('/pledges/{pledge_id}/cancel')
def cancel_pledge(pledge_id: int):
    with db() as conn:
        with conn.cursor() as c:
            c.execute("UPDATE pledges SET status='cancelled' WHERE id=%s", (pledge_id,))
    return {'ok': True}

# ── PAYMENTS / YOUSAFE CSV IMPORT ────────────────────────────────────────────

@app.post('/payments/import-csv')
async def import_payments_csv(file: UploadFile = File(...)):
    """Parse a bank statement CSV and auto-match transactions to subscribers by payment_ref."""
    import csv as _csv, io as _io
    content = (await file.read()).decode('utf-8-sig', errors='replace')
    reader = _csv.DictReader(_io.StringIO(content))
    rows = list(reader)

    # Normalise header names (YouSafe / generic SEPA formats differ)
    def _find_col(headers, *candidates):
        for h in headers:
            for c in candidates:
                if c.lower() in h.lower():
                    return h
        return None

    headers = reader.fieldnames or []
    col_date   = _find_col(headers, 'date','datum','buchungstag','value date')
    col_amount = _find_col(headers, 'amount','betrag','credit','debit','value')
    col_ref    = _find_col(headers, 'reference','verwendungszweck','purpose','description','remark','memo','details')
    col_sender = _find_col(headers, 'name','sender','auftraggeber','originator','from','debtor')

    # Load all subscribers with their payment_ref
    with db() as conn:
        with conn.cursor() as c:
            c.execute("SELECT tg_id, anon_id, internal_name, payment_ref FROM conversations WHERE payment_ref != '' AND payment_ref IS NOT NULL")
            subs = {r['payment_ref'].upper(): dict(r) for r in c.fetchall()}

    matched = []
    unmatched = []
    ts_now = datetime.now().isoformat()

    for row in rows:
        raw_amount = row.get(col_amount, '0').replace(',', '.').replace('€','').strip() if col_amount else '0'
        try:
            amount = abs(float(raw_amount))
        except ValueError:
            continue
        if amount <= 0:
            continue

        ref_text  = row.get(col_ref, '').upper() if col_ref else ''
        date_text = row.get(col_date, '') if col_date else ''
        sender    = row.get(col_sender, '') if col_sender else ''

        # Try to find ZF-XXXX pattern in reference
        import re as _re
        match = _re.search(r'ZF[-\s]?(\d{4,})', ref_text)
        matched_sub = None
        if match:
            code = f'ZF-{match.group(1)}'
            matched_sub = subs.get(code)

        if matched_sub:
            # Create approved sale immediately
            with db() as conn:
                with conn.cursor() as c:
                    c.execute(
                        "INSERT INTO sales (tg_id,anon_id,amount,product,notes,chatter,timestamp,status,payment_method) VALUES (%s,%s,%s,%s,%s,%s,%s,'approved','YouSafe') RETURNING id",
                        (matched_sub['tg_id'], matched_sub['anon_id'], amount,
                         'YouSafe Transfer', f'CSV import · {date_text} · {sender}',
                         'CSV Import', ts_now)
                    )
                    sale_id = c.fetchone()['id']
            matched.append({
                'date': date_text, 'amount': amount, 'sender': sender,
                'ref': ref_text[:80], 'subscriber': matched_sub['internal_name'] or matched_sub['anon_id'],
                'tg_id': matched_sub['tg_id'], 'sale_id': sale_id
            })
        else:
            unmatched.append({
                'date': date_text, 'amount': amount,
                'sender': sender, 'ref': ref_text[:80]
            })

    return {'ok': True, 'matched': matched, 'unmatched': unmatched,
            'total': len(rows), 'matched_count': len(matched), 'unmatched_count': len(unmatched)}

@app.post('/payments/match-manual')
async def match_manual_payment(tg_id: str, amount: float, date: str = '', sender: str = '', notes: str = ''):
    """Manually match an unmatched transaction to a subscriber."""
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT anon_id, internal_name FROM conversations WHERE tg_id=%s', (tg_id,))
            row = c.fetchone()
    if not row:
        raise HTTPException(404, 'Subscriber not found')
    ts = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO sales (tg_id,anon_id,amount,product,notes,chatter,timestamp,status,payment_method) VALUES (%s,%s,%s,%s,%s,%s,%s,'approved','YouSafe') RETURNING id",
                (tg_id, row['anon_id'], amount,
                 'YouSafe Transfer', f'{notes} · {date} · {sender}'.strip(' ·'),
                 'Manual Match', ts)
            )
    return {'ok': True}

# ── FAKECHECK ────────────────────────────────────────────────────────────────
FAKECHECK_BOT = '@FakecheckAudioBot'

class FakecheckIn(BaseModel):
    tg_id: str           # subscriber to send audio to
    command: str = ''    # what to send to the bot (e.g. subscriber name, or /start)

@app.post('/fakecheck')
async def send_fakecheck(body: FakecheckIn):
    if not tg_client or not tg_client.is_connected():
        raise HTTPException(503, 'Userbot nicht verbunden')
    try:
        # 1. Send command to the fakecheck bot
        trigger = body.command.strip() or 'start'
        await tg_client.send_message(FAKECHECK_BOT, trigger)

        # 2. Wait up to 15s for bot to reply with audio
        import asyncio as _aio
        audio_msg = None
        for _ in range(30):  # poll every 0.5s for 15s
            await _aio.sleep(0.5)
            msgs = await tg_client.get_messages(FAKECHECK_BOT, limit=1)
            if msgs and msgs[0].voice:
                audio_msg = msgs[0]
                break

        if not audio_msg:
            raise HTTPException(408, 'Bot hat keine Sprachnachricht geschickt (Timeout)')

        # 3. Forward audio to subscriber
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (body.tg_id,))
                row = c.fetchone()
        ah = int(row['tg_access_hash']) if row and row['tg_access_hash'] else 0
        peer = InputPeerUser(int(body.tg_id), ah) if ah else int(body.tg_id)
        await tg_client.forward_messages(peer, audio_msg)

        # 4. Log as outgoing message
        save_msg(body.tg_id, '[🎙️ Fakecheck Audio]', 'out', 'System')

        # 5. Broadcast to CRM clients
        asyncio.create_task(ws_manager.broadcast({
            'type': 'new_message', 'tg_id': body.tg_id,
            'text': '[🎙️ Fakecheck Audio]', 'direction': 'out',
            'timestamp': datetime.now().isoformat()
        }))
        return {'ok': True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ── BROADCAST ────────────────────────────────────────────────────────────────
class BroadcastIn(BaseModel):
    text: str
    chatter: str = 'Broadcast'
    filter_stage: Optional[str] = None
    filter_min_spend: Optional[float] = None
    tg_ids: Optional[list] = None
    exclude_ids: Optional[list] = None       # exclude specific tg_ids
    exclude_stages: Optional[list] = None    # exclude by funnel stage
    exclude_lists: Optional[list] = None     # exclude all members of these list ids (whale/big/...)

@app.post('/broadcast')
async def post_broadcast(body: BroadcastIn, background_tasks: BackgroundTasks):
    if not tg_client or not tg_client.is_connected():
        raise HTTPException(503, 'Userbot nicht verbunden')
    if not body.text.strip():
        raise HTTPException(400, 'Text darf nicht leer sein')

    # Build recipient list
    with db() as conn:
        with conn.cursor() as c:
            if body.tg_ids:
                fmt = ','.join(['%s'] * len(body.tg_ids))
                c.execute(f'SELECT tg_id,tg_access_hash,funnel_stage FROM conversations WHERE tg_id IN ({fmt})', body.tg_ids)
            elif body.filter_stage:
                c.execute('SELECT tg_id,tg_access_hash,funnel_stage FROM conversations WHERE funnel_stage=%s', (body.filter_stage,))
            else:
                c.execute('SELECT tg_id,tg_access_hash,funnel_stage FROM conversations')
            all_r = c.fetchall()

    # Apply exclusions
    exclude_set = set(body.exclude_ids or [])
    # Exclude every member of the chosen lists (e.g. whales / big spenders)
    if body.exclude_lists:
        try:
            with db() as conn:
                with conn.cursor() as c:
                    fmt = ','.join(['%s'] * len(body.exclude_lists))
                    c.execute(f'SELECT tg_id FROM list_members WHERE list_id IN ({fmt})',
                              [int(x) for x in body.exclude_lists])
                    for r in c.fetchall():
                        exclude_set.add(r['tg_id'])
        except Exception as _e:
            print(f'exclude_lists error: {_e}')
    exclude_stages = set(body.exclude_stages or [])
    recipients = [r for r in all_r
                  if r['tg_id'] not in exclude_set
                  and (not exclude_stages or r.get('funnel_stage','') not in exclude_stages)]

    # Start background task — return immediately so Railway doesn't time out
    background_tasks.add_task(_run_broadcast_bg, body, recipients)
    return {'ok': True, 'started': True, 'total': len(recipients)}


async def _run_broadcast_bg(body: BroadcastIn, recipients: list):
    """Runs broadcast in background, pushes progress via WebSocket every 10 msgs."""
    sent_ok, sent_fail = 0, 0
    total = len(recipients)
    for i, r in enumerate(recipients):
        # Pause if a chat reply or call has priority — yield Telethon connection
        while _tg_priority > 0:
            await asyncio.sleep(0.1)
        try:
            ah = int(r['tg_access_hash']) if r['tg_access_hash'] else 0
            peer = InputPeerUser(int(r['tg_id']), ah) if ah else int(r['tg_id'])
            sent_msg = await tg_client.send_message(peer, body.text)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: save_msg(r['tg_id'], body.text, 'out', body.chatter, sent_msg.id))
            sent_ok += 1
            await asyncio.sleep(0)  # yield event loop for pending chat requests
            await asyncio.sleep(1.2)   # ~50 msg/min — safe for Telegram
        except FloodWaitError as e:
            wait = e.seconds + 5
            print(f'Broadcast FloodWait {wait}s')
            await asyncio.sleep(wait)
        except Exception as ex:
            print(f'Broadcast skip {r["tg_id"]}: {ex}')
            sent_fail += 1
        # Push progress every 10 messages and at the end
        if (i + 1) % 10 == 0 or (i + 1) == total:
            await ws_manager.broadcast({
                'type': 'broadcast_progress',
                'sent': sent_ok, 'failed': sent_fail,
                'total': total, 'current': i + 1
            })
    await ws_manager.broadcast({
        'type': 'broadcast_done',
        'sent': sent_ok, 'failed': sent_fail, 'total': total
    })
    print(f'✅ Broadcast done: {sent_ok}/{total} sent, {sent_fail} failed')

# ── SCHEDULED BROADCAST ──────────────────────────────────────────────────────
class ScheduledBroadcastIn(BaseModel):
    text: str
    scheduled_at: str          # ISO datetime string e.g. "2024-06-10T18:00:00"
    filter_stage: str = ''     # '' = all, or funnel stage key
    exclude_stages: list = []
    chatter: str = 'Broadcast'

@app.post('/scheduled-broadcast')
def create_scheduled_broadcast(body: ScheduledBroadcastIn):
    if not body.text.strip():
        raise HTTPException(400, 'Text darf nicht leer sein')
    ts = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute('''
                INSERT INTO scheduled_broadcasts (text, scheduled_at, filter_stage, exclude_stages, chatter, status, created_at)
                VALUES (%s, %s, %s, %s, %s, 'pending', %s) RETURNING id
            ''', (body.text, body.scheduled_at, body.filter_stage,
                  ','.join(body.exclude_stages), body.chatter, ts))
            row = c.fetchone()
    return {'ok': True, 'id': row['id']}

@app.get('/scheduled-broadcasts')
def list_scheduled_broadcasts():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT * FROM scheduled_broadcasts ORDER BY scheduled_at ASC')
            rows = c.fetchall()
    return [dict(r) for r in rows]

@app.delete('/scheduled-broadcast/{bid}')
def delete_scheduled_broadcast(bid: int):
    with db() as conn:
        with conn.cursor() as c:
            c.execute("UPDATE scheduled_broadcasts SET status='cancelled' WHERE id=%s AND status='pending'", (bid,))
    return {'ok': True}

async def _run_scheduled_broadcasts():
    """Background task: every 60s check for pending scheduled broadcasts and fire them."""
    while True:
        try:
            await asyncio.sleep(60)
            if not tg_client or not tg_client.is_connected():
                continue
            now = datetime.now().isoformat()
            with db() as conn:
                with conn.cursor() as c:
                    c.execute("""
                        SELECT * FROM scheduled_broadcasts
                        WHERE status='pending' AND scheduled_at <= %s
                        ORDER BY scheduled_at ASC
                    """, (now,))
                    due = c.fetchall()

            for job in due:
                print(f'⏰ Running scheduled broadcast id={job["id"]} at {now}')
                # Mark as running immediately to prevent double-fire
                with db() as conn:
                    with conn.cursor() as c:
                        c.execute("UPDATE scheduled_broadcasts SET status='running' WHERE id=%s AND status='pending'", (job['id'],))

                # Build recipient list
                with db() as conn:
                    with conn.cursor() as c:
                        if job['filter_stage']:
                            c.execute('SELECT tg_id,tg_access_hash FROM conversations WHERE funnel_stage=%s', (job['filter_stage'],))
                        else:
                            c.execute('SELECT tg_id,tg_access_hash FROM conversations')
                        recipients = c.fetchall()

                exclude_stages = set(s.strip() for s in (job['exclude_stages'] or '').split(',') if s.strip())
                if exclude_stages:
                    with db() as conn:
                        with conn.cursor() as c:
                            c.execute('SELECT tg_id,funnel_stage FROM conversations')
                            stage_map = {r['tg_id']: r['funnel_stage'] for r in c.fetchall()}
                    recipients = [r for r in recipients if stage_map.get(r['tg_id'], '') not in exclude_stages]

                sent_ok, sent_fail = 0, 0
                for r in recipients:
                    try:
                        ah = int(r['tg_access_hash']) if r['tg_access_hash'] else 0
                        peer = InputPeerUser(int(r['tg_id']), ah) if ah else int(r['tg_id'])
                        sent_msg = await tg_client.send_message(peer, job['text'])
                        save_msg(r['tg_id'], job['text'], 'out', job['chatter'], sent_msg.id)
                        sent_ok += 1
                        await asyncio.sleep(1.2)
                    except FloodWaitError as e:
                        await asyncio.sleep(e.seconds + 5)
                    except Exception as ex:
                        print(f'Scheduled broadcast skip {r["tg_id"]}: {ex}')
                        sent_fail += 1

                with db() as conn:
                    with conn.cursor() as c:
                        c.execute("""
                            UPDATE scheduled_broadcasts
                            SET status='sent', sent_count=%s, fail_count=%s, sent_at=%s
                            WHERE id=%s
                        """, (sent_ok, sent_fail, datetime.now().isoformat(), job['id']))
                print(f'✅ Scheduled broadcast id={job["id"]} done: {sent_ok} sent, {sent_fail} failed')

        except Exception as e:
            print(f'⚠️ Scheduled broadcast runner error: {e}')

# ── SALES ─────────────────────────────────────────────────────────────────────

class SaleSubmitIn(BaseModel):
    tg_id: str
    anon_id: str
    amount: float
    product: str = ''
    notes: str = ''
    chatter: str = 'Chatter'
    is_admin: bool = False
    payment_method: str = ''
    payment_code: str = ''  # Paysafe 16-digit or Amazon gift card code

@app.post('/sale')
async def post_sale(body: SaleSubmitIn):
    ts = datetime.now().isoformat()
    # Admins self-approve; chatters start as pending
    status = 'approved' if body.is_admin else 'pending'
    sale_id = None
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                'INSERT INTO sales (tg_id,anon_id,amount,product,notes,chatter,timestamp,status,payment_method,payment_code) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id',
                (body.tg_id, body.anon_id, body.amount, body.product, body.notes, body.chatter, ts, status, body.payment_method, body.payment_code)
            )
            sale_id = c.fetchone()['id']
    if status == 'approved':
        make_url = os.environ.get('MAKE_SALE_WEBHOOK', '')
        if make_url:
            _fire_webhook_sync(make_url, {
                'tg_id': body.tg_id, 'anon_id': body.anon_id,
                'amount': body.amount, 'product': body.product,
                'notes': body.notes, 'chatter': body.chatter, 'timestamp': ts
            })
        asyncio.create_task(ws_manager.broadcast({
            'type': 'notification', 'notif_type': 'sale',
            'tg_id': body.tg_id, 'anon_id': body.anon_id,
            'amount': body.amount, 'product': body.product,
            'chatter': body.chatter, 'timestamp': ts,
        }))
    else:
        # Notify admin of new pending sale
        asyncio.create_task(ws_manager.broadcast({
            'type': 'pending_sale',
            'sale_id': sale_id, 'tg_id': body.tg_id, 'anon_id': body.anon_id,
            'amount': body.amount, 'product': body.product,
            'chatter': body.chatter, 'timestamp': ts,
        }))
    return {'ok': True, 'id': sale_id, 'status': status, 'timestamp': ts}

@app.post('/sale/{sale_id}/screenshot')
async def upload_sale_screenshot(sale_id: int, file: UploadFile = File(...)):
    """Attach a proof screenshot to a sale."""
    ext = (file.filename or 'proof.jpg').rsplit('.', 1)[-1].lower()
    if ext not in ('jpg','jpeg','png','gif','webp','mp4'):
        raise HTTPException(400, 'Invalid file type')
    fname = f'sale_{sale_id}_{int(datetime.now().timestamp())}.{ext}'
    fpath = os.path.join(PROOFS_DIR, fname)
    with open(fpath, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    with db() as conn:
        with conn.cursor() as c:
            c.execute('UPDATE sales SET screenshot=%s WHERE id=%s', (fname, sale_id))
    return {'ok': True, 'screenshot': fname}

@app.get('/sale/{sale_id}/screenshot')
def serve_sale_screenshot(sale_id: int):
    """Serve the proof screenshot for a sale."""
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT screenshot FROM sales WHERE id=%s', (sale_id,))
            row = c.fetchone()
    if not row or not row['screenshot']:
        raise HTTPException(404, 'No screenshot')
    fpath = os.path.join(PROOFS_DIR, row['screenshot'])
    if not os.path.isfile(fpath):
        raise HTTPException(404, 'File not found')
    return FileResponse(fpath)

class ReviewIn(BaseModel):
    reviewed_by: str = 'Admin'
    reason: str = ''

@app.post('/sale/{sale_id}/approve')
async def approve_sale(sale_id: int, body: ReviewIn):
    """Admin approves a pending sale."""
    ts = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                'UPDATE sales SET status=%s, reviewed_by=%s, reviewed_at=%s WHERE id=%s RETURNING tg_id,anon_id,amount,product,chatter,timestamp',
                ('approved', body.reviewed_by, ts, sale_id)
            )
            row = c.fetchone()
    if not row:
        raise HTTPException(404, 'Sale not found')
    make_url = os.environ.get('MAKE_SALE_WEBHOOK', '')
    if make_url:
        _fire_webhook_sync(make_url, {
            'tg_id': row['tg_id'], 'anon_id': row['anon_id'],
            'amount': row['amount'], 'product': row['product'],
            'notes': '', 'chatter': row['chatter'], 'timestamp': row['timestamp']
        })
    asyncio.create_task(ws_manager.broadcast({
        'type': 'notification', 'notif_type': 'sale',
        'tg_id': row['tg_id'], 'anon_id': row['anon_id'],
        'amount': row['amount'], 'product': row['product'],
        'chatter': row['chatter'], 'timestamp': row['timestamp'],
    }))
    asyncio.create_task(ws_manager.broadcast({
        'type': 'sale_reviewed', 'sale_id': sale_id, 'status': 'approved'
    }))
    return {'ok': True}

@app.post('/sale/{sale_id}/reject')
async def reject_sale(sale_id: int, body: ReviewIn):
    """Admin rejects a pending sale with a reason."""
    ts = datetime.now().isoformat()
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                'UPDATE sales SET status=%s, reviewed_by=%s, reviewed_at=%s, rejection_reason=%s WHERE id=%s',
                ('rejected', body.reviewed_by, ts, body.reason, sale_id)
            )
    asyncio.create_task(ws_manager.broadcast({
        'type': 'sale_reviewed', 'sale_id': sale_id, 'status': 'rejected', 'reason': body.reason
    }))
    return {'ok': True}

@app.get('/sales/pending')
def get_pending_sales():
    """List all pending sales for admin review."""
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                "SELECT id,tg_id,anon_id,amount,product,notes,chatter,timestamp,screenshot FROM sales WHERE status='pending' ORDER BY timestamp DESC",
            )
            rows = c.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['screenshot_url'] = f'/sale/{r["id"]}/screenshot' if r['screenshot'] else ''
        result.append(d)
    return result

@app.get('/sales/codes')
def get_sale_codes():
    """Return all Paysafe and Amazon codes with their sale status."""
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                "SELECT id,anon_id,amount,payment_method,payment_code,status,timestamp,chatter FROM sales WHERE payment_code != '' AND payment_code IS NOT NULL ORDER BY timestamp DESC"
            )
            rows = c.fetchall()
    return [dict(r) for r in rows]

@app.get('/sales')
def get_sales(limit: int = 200):
    with db() as conn:
        with conn.cursor() as c:
            c.execute(
                'SELECT id,tg_id,anon_id,amount,product,notes,chatter,timestamp,status,screenshot,rejection_reason,payment_method,payment_code FROM sales ORDER BY timestamp DESC LIMIT %s',
                (limit,)
            )
            rows = c.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['screenshot_url'] = f'/sale/{r["id"]}/screenshot' if r['screenshot'] else ''
        result.append(d)
    return result

@app.get('/media/{tg_id}')
def get_subscriber_media(tg_id: str, limit: int = 200, offset: int = 0):
    """List all media messages for a subscriber (photos, docs, voice)."""
    def _media_type(text: str) -> str:
        t = text.lower()
        if 'photo' in t: return 'photo'
        if 'voice' in t or '🎤' in text: return 'voice'
        if 'video' in t or '🎬' in text: return 'video'
        if 'document' in t or '📎' in text: return 'document'
        if 'sticker' in t: return 'sticker'
        return 'other'

    with db() as conn:
        with conn.cursor() as c:
            c.execute('''
                SELECT id, tg_msg_id, text, direction, timestamp, chatter
                FROM messages
                WHERE tg_id=%s
                  AND tg_msg_id > 0
                  AND (text LIKE '[📷%%' OR text LIKE '[📎%%' OR text LIKE '[🎤%%'
                       OR text LIKE '[Sticker%%' OR text LIKE '[🎬%%'
                       OR text = '[Message]')
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
            ''', (tg_id, limit, offset))
            rows = c.fetchall()

    # Also check which ones are already cached on disk
    cache_dir = os.path.join(MEDIA_CACHE_DIR, str(tg_id))
    cached_ids = set()
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            try: cached_ids.add(int(f.split('.')[0]))
            except: pass

    return {
        'items': [
            {
                'id': r['id'],
                'tg_msg_id': r['tg_msg_id'],
                'type': _media_type(r['text']),
                'direction': r['direction'],
                'timestamp': r['timestamp'],
                'chatter': r.get('chatter', ''),
                'cached': r['tg_msg_id'] in cached_ids,
                'media_url': f'/messages/{tg_id}/{r["tg_msg_id"]}/media',
            }
            for r in rows
        ]
    }

@app.get('/messages/{tg_id}/{msg_id}/media')
async def download_message_media(tg_id: str, msg_id: int):
    """Download media from a specific Telegram message. Caches to disk after first download."""
    # ── Serve from cache if available ──────────────────────────────────────────
    cache_dir = os.path.join(MEDIA_CACHE_DIR, str(tg_id))
    os.makedirs(cache_dir, exist_ok=True)
    cached_matches = [f for f in os.listdir(cache_dir) if f.startswith(f'{msg_id}.') or f == str(msg_id)]
    if cached_matches:
        cached_path = os.path.join(cache_dir, cached_matches[0])
        if os.path.isfile(cached_path):
            return FileResponse(cached_path, filename=os.path.basename(cached_path))
    # ── Download from Telegram ─────────────────────────────────────────────────
    if not tg_client or not tg_client.is_connected():
        raise HTTPException(503, 'Userbot not connected')
    try:
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (tg_id,))
                row = c.fetchone()
        ah = int(row['tg_access_hash']) if row and row['tg_access_hash'] else 0
        peer = InputPeerUser(int(tg_id), ah) if ah else int(tg_id)
        msgs = await tg_client.get_messages(peer, ids=msg_id)
        if not msgs or not msgs.media:
            raise HTTPException(404, 'No media in this message')
        # Download directly into cache dir (Telethon adds the extension automatically)
        dl_path = os.path.join(cache_dir, str(msg_id))
        path = await tg_client.download_media(msgs, file=dl_path)
        if not path or not os.path.isfile(path):
            raise HTTPException(500, 'Download failed')
        return FileResponse(path, filename=os.path.basename(path))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get('/profile-photo/{tg_id}')
async def get_profile_photo(tg_id: str):
    """Return the Telegram profile photo for a subscriber. Cached to disk after first download."""
    PHOTOS_DIR = os.path.join(VAULT_DIR, '_photos')
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    # Check cache first (any file starting with tg_id)
    cached = [f for f in os.listdir(PHOTOS_DIR) if f.startswith(f'{tg_id}.')]
    if cached:
        p = os.path.join(PHOTOS_DIR, cached[0])
        if os.path.isfile(p):
            return FileResponse(p, media_type='image/jpeg')
    if not tg_client or not tg_client.is_connected():
        raise HTTPException(503, 'Userbot not connected')
    if active_calls:
        # Don't hit MTProto (photo download) while a call is active — it can drop the call.
        raise HTTPException(404, 'profile photo deferred during active call')
    try:
        with db() as conn:
            with conn.cursor() as c:
                c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (tg_id,))
                row = c.fetchone()
        ah = int(row['tg_access_hash']) if row and row['tg_access_hash'] else 0
        from telethon.tl.types import InputPeerUser
        peer = InputPeerUser(int(tg_id), ah) if ah else int(tg_id)
        dl_path = os.path.join(PHOTOS_DIR, f'{tg_id}.jpg')
        path = await tg_client.download_profile_photo(peer, file=dl_path)
        if not path or not os.path.isfile(path):
            raise HTTPException(404, 'No profile photo')
        return FileResponse(path, media_type='image/jpeg')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ── NOONES INTEGRATION ────────────────────────────────────────────────────────
import urllib.request as _urllib_req, urllib.parse as _urllib_parse, json as _json_mod

def _noones_setting(key: str) -> str:
    with db() as conn:
        with conn.cursor() as c:
            c.execute("SELECT value FROM crm_settings WHERE key=%s", (key,))
            row = c.fetchone()
    return (row['value'] if row else '') or ''

async def _noones_token(key: str, secret: str) -> str:
    data = _urllib_parse.urlencode({'grant_type':'client_credentials','client_id':key,'client_secret':secret}).encode()
    req = _urllib_req.Request('https://auth.noones.com/oauth2/token', data=data,
        headers={'Content-Type':'application/x-www-form-urlencoded'}, method='POST')
    loop = asyncio.get_event_loop()
    def _do():
        with _urllib_req.urlopen(req, timeout=10) as r: return _json_mod.loads(r.read())
    return (await loop.run_in_executor(None, _do))['access_token']

async def _noones_api(token: str, method: str, path: str, payload: dict = None):
    url = f'https://api.noones.com/noones/v1{path}'
    body = _urllib_parse.urlencode(payload or {}).encode() if payload else None
    req = _urllib_req.Request(url, data=body,
        headers={'Authorization': f'Bearer {token}', 'Content-Type':'application/x-www-form-urlencoded'},
        method=method)
    loop = asyncio.get_event_loop()
    def _do():
        with _urllib_req.urlopen(req, timeout=15) as r: return _json_mod.loads(r.read())
    return await loop.run_in_executor(None, _do)

_NOONES_SLUG = {'amazon': 'amazon-gift-card', 'paysafe': 'paysafecash'}

class NoonesRedeemBody(BaseModel):
    card_type:  str
    amount_eur: float
    code:       str
    pin:        str = ''

@app.post('/noones/redeem')
async def noones_redeem(body: NoonesRedeemBody):
    key    = _noones_setting('noones_key')
    secret = _noones_setting('noones_secret')
    if not key or not secret:
        raise HTTPException(400, 'Noones API key not configured in Settings')
    slug = _NOONES_SLUG.get(body.card_type)
    if not slug:
        raise HTTPException(400, f'Unknown card type: {body.card_type}')
    try:
        token = await _noones_token(key, secret)
        offers_resp = await _noones_api(token, 'GET',
            f'/offer/list?payment_method={slug}&fiat_currency=EUR&offer_type=buy&limit=10')
        offers = (offers_resp.get('data', {}).get('offer_list') or
                  offers_resp.get('data', []) or offers_resp.get('offers', []))
        if not offers:
            raise HTTPException(404, f'No active {body.card_type} buyers on Noones right now')
        offer_hash = offers[0].get('offer_id') or offers[0].get('id')
        trade_resp = await _noones_api(token, 'POST', '/trade/start',
            {'offer_hash': offer_hash, 'fiat_amount': str(body.amount_eur)})
        trade = trade_resp.get('data', {}).get('trade') or trade_resp.get('data', {})
        trade_hash = trade.get('trade_hash') or trade.get('id')
        if not trade_hash:
            raise HTTPException(500, f'Trade start failed: {trade_resp}')
        msg = f'Code: {body.code}' + (f'\nPIN: {body.pin}' if body.pin else '')
        await _noones_api(token, 'POST', '/trade/chat/message',
            {'trade_hash': trade_hash, 'message': msg})
        return {'ok': True, 'trade_hash': trade_hash, 'trade_url': f'https://noones.com/trade/{trade_hash}'}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f'Noones error: {e}')

# ── LISTS ─────────────────────────────────────────────────────────────────────
class ListCreate(BaseModel):
    name: str
    color: str = '#00d4aa'

@app.get('/lists')
def get_lists():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT id,name,color FROM lists ORDER BY id')
            lists = [dict(r) for r in c.fetchall()]
            for lst in lists:
                c.execute('SELECT tg_id FROM list_members WHERE list_id=%s', (lst['id'],))
                lst['members'] = [r['tg_id'] for r in c.fetchall()]
    return lists

@app.post('/lists')
def create_list(body: ListCreate):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('INSERT INTO lists (name,color) VALUES (%s,%s) RETURNING id', (body.name, body.color))
            lid = c.fetchone()['id']
    return {'ok': True, 'id': lid}

@app.delete('/lists/{list_id}')
def delete_list(list_id: int):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('DELETE FROM list_members WHERE list_id=%s', (list_id,))
            c.execute('DELETE FROM lists WHERE id=%s', (list_id,))
    return {'ok': True}

@app.post('/lists/{list_id}/members/{tg_id}')
def add_to_list(list_id: int, tg_id: str):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('INSERT INTO list_members (list_id,tg_id) VALUES (%s,%s) ON CONFLICT DO NOTHING', (list_id, tg_id))
    return {'ok': True}

@app.delete('/lists/{list_id}/members/{tg_id}')
def remove_from_list(list_id: int, tg_id: str):
    with db() as conn:
        with conn.cursor() as c:
            c.execute('DELETE FROM list_members WHERE list_id=%s AND tg_id=%s', (list_id, tg_id))
    return {'ok': True}

# ── SUBSCRIBERS ───────────────────────────────────────────────────────────────
@app.get('/subscribers')
def get_subscribers():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('''SELECT tg_id, anon_id, tg_username, tg_phone,
                                internal_name, first_time, last_time, msg_count, time_waster
                         FROM conversations ORDER BY first_time ASC''')
            rows = c.fetchall()
    return [dict(r) for r in rows]

@app.get('/export/subscribers')
def export_subscribers_csv():
    with db() as conn:
        with conn.cursor() as c:
            c.execute('''SELECT tg_id, anon_id, tg_username, tg_phone,
                                internal_name, first_time, last_time, msg_count
                         FROM conversations ORDER BY first_time ASC''')
            rows = c.fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['TG ID','Anon ID','Username','Telefon','Interner Name','Erster Kontakt','Letzter Kontakt','Nachrichten'])
    for r in rows:
        writer.writerow([
            r['tg_id'], r['anon_id'],
            f"@{r['tg_username']}" if r['tg_username'] else '—',
            f"+{r['tg_phone']}" if r['tg_phone'] else '—',
            r['internal_name'] or '—', r['first_time'] or '—',
            r['last_time'] or '—', r['msg_count'] or 0
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename="subscriber_backup.csv"'}
    )

# ── FAKE CALLS ───────────────────────────────────────────────────────────────

CALL_ALLOWED_EXTS = {'mp3','mp4','wav','ogg','aac','m4a','mov','mkv','flac'}

@app.get('/call/status')
def call_status():
    """Check if pytgcalls is available and how many calls are active."""
    return {
        'pytgcalls_available': _PYTGCALLS_OK,
        'pytgcalls_error': _PYTGCALLS_ERR,
        'calls_client_ready': calls_client is not None,
        'active': len(active_calls),
        'active_calls': active_calls,
    }

@app.get('/call/files')
def get_call_files():
    """List recordings organised by folder. Returns {folders: [{name, label, files}]}"""
    def _scan_dir(directory: str, folder_key: str) -> list:
        result = []
        try:
            for fname in sorted(os.listdir(directory)):
                fpath = os.path.join(directory, fname)
                if os.path.isfile(fpath) and not fname.startswith('.'):
                    ext = fname.rsplit('.',1)[-1].lower() if '.' in fname else ''
                    if ext not in CALL_ALLOWED_EXTS:
                        continue
                    ftype = 'video' if ext in ('mp4','mov','mkv') else 'audio'
                    result.append({
                        'name': fname,
                        'folder': folder_key,
                        'size': os.path.getsize(fpath),
                        'type': ftype,
                    })
        except Exception as e:
            print(f'call files scan error ({folder_key}): {e}')
        return result

    FOLDER_LABELS = {'': 'Alle', 'fake_checks': '✅ Fake Checks', 'paid_calls': '💰 Paid Calls'}
    folders = []
    # Root (ungrouped)
    folders.append({'key': '', 'label': 'Alle', 'files': _scan_dir(CALLS_DIR, '')})
    # Sub-folders
    for fkey in CALL_FOLDERS:
        fdir = os.path.join(CALLS_DIR, fkey)
        folders.append({'key': fkey, 'label': FOLDER_LABELS.get(fkey, fkey), 'files': _scan_dir(fdir, fkey)})
    return {'folders': folders}

@app.post('/call/upload')
async def upload_call_file(file: UploadFile = File(...), folder: str = ''):
    """Upload a call recording. folder='' for root, or 'fake_checks'/'paid_calls'."""
    if folder and folder not in CALL_FOLDERS:
        raise HTTPException(400, f'Invalid folder. Choose from: {CALL_FOLDERS}')
    ext = (file.filename or 'call.mp3').rsplit('.',1)[-1].lower()
    if ext not in CALL_ALLOWED_EXTS:
        raise HTTPException(400, f'File type .{ext} not allowed.')
    target_dir = os.path.join(CALLS_DIR, folder) if folder else CALLS_DIR
    safe_name = _vault_safe(file.filename or f'call.{ext}') or f'call.{ext}'
    fpath = os.path.join(target_dir, safe_name)
    if os.path.exists(fpath):
        base, e = os.path.splitext(safe_name)
        safe_name = f'{base}_{int(datetime.now().timestamp())}{e}'
        fpath = os.path.join(target_dir, safe_name)
    try:
        with open(fpath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except OSError as e:
        if 'No space left' in str(e):
            raise HTTPException(507, 'Disk full')
        raise HTTPException(500, str(e))
    return {'ok': True, 'name': safe_name, 'folder': folder}

# ── URL IMPORT ───────────────────────────────────────────────────────────────
_import_jobs: dict = {}   # job_id → {status, progress, filename, error, folder}

_GDRIVE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

def _gdrive_file_id(url: str) -> str | None:
    m = re.search(r'drive\.google\.com/file/d/([^/?#]+)', url)
    return m.group(1) if m else None

def _run_import(job_id: str, url: str, folder: str):
    """Background thread: download URL (Google Drive or direct) and save to CALLS_DIR/folder."""
    job = _import_jobs[job_id]
    target_dir = os.path.join(CALLS_DIR, folder) if folder else CALLS_DIR
    os.makedirs(target_dir, exist_ok=True)
    try:
        file_id = _gdrive_file_id(url)
        CHUNK = 2 * 1024 * 1024  # 2 MB chunks

        if _HAS_REQUESTS:
            # ── requests path (preferred) ─────────────────────────────
            sess = _requests.Session()
            sess.headers.update(_GDRIVE_HEADERS)

            if file_id:
                # Step 1: hit the /uc endpoint — GDrive will either start the download
                # or return an HTML confirmation page for large files
                dl_url = f'https://drive.google.com/uc?id={file_id}&export=download'
                r = sess.get(dl_url, stream=True, timeout=60, allow_redirects=True)

                # Step 2: if we got HTML, it's the virus-scan/large-file confirmation
                content_type = r.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    # Extract confirm token + uuid from the page
                    confirm_m = re.search(r'name="confirm"\s+value="([^"]+)"', r.text)
                    uuid_m    = re.search(r'name="uuid"\s+value="([^"]+)"', r.text)
                    if not confirm_m:
                        # Newer GDrive format: token in URL param
                        confirm_m = re.search(r'[?&]confirm=([^&"]+)', r.text)
                    confirm = confirm_m.group(1) if confirm_m else 't'
                    uuid_val = uuid_m.group(1) if uuid_m else ''
                    params = f'id={file_id}&export=download&confirm={confirm}'
                    if uuid_val:
                        params += f'&uuid={uuid_val}'
                    r = sess.get(
                        f'https://drive.usercontent.google.com/download?{params}',
                        stream=True, timeout=60, allow_redirects=True
                    )
            else:
                r = sess.get(url, stream=True, timeout=60, allow_redirects=True)

            r.raise_for_status()

            # Filename
            cd = r.headers.get('Content-Disposition', '')
            fname_m = re.search(r"filename\*?=(?:UTF-8'')?[\"']?([^\"';\r\n]+)", cd, re.IGNORECASE)
            if fname_m:
                raw_name = urllib.parse.unquote(fname_m.group(1).strip('" '))
            else:
                raw_name = (url.rstrip('/').split('/')[-1].split('?')[0]) or 'recording'
                if '.' not in raw_name:
                    ct = r.headers.get('Content-Type', 'video/mp4').split(';')[0].strip()
                    ext_map = {'video/mp4':'mp4','video/quicktime':'mov','audio/mpeg':'mp3','audio/ogg':'ogg','video/x-matroska':'mkv'}
                    raw_name += '.' + ext_map.get(ct, 'mp4')

            safe_name = _vault_safe(raw_name) or 'recording.mp4'
            fpath = os.path.join(target_dir, safe_name)
            if os.path.exists(fpath):
                base, e = os.path.splitext(safe_name)
                safe_name = f'{base}_{int(datetime.now().timestamp())}{e}'
                fpath = os.path.join(target_dir, safe_name)

            job['filename'] = safe_name
            total = int(r.headers.get('Content-Length', 0))
            downloaded = 0
            with open(fpath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        job['progress'] = round(downloaded / total * 100) if total else -1

        else:
            # ── urllib fallback ───────────────────────────────────────
            if file_id:
                dl_url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0&confirm=t'
            else:
                dl_url = url
            req = urllib.request.Request(dl_url, headers=_GDRIVE_HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp:
                cd = resp.headers.get('Content-Disposition', '')
                fname_m = re.search(r"filename\*?=(?:UTF-8'')?[\"']?([^\"';\r\n]+)", cd, re.IGNORECASE)
                raw_name = urllib.parse.unquote(fname_m.group(1).strip('" ')) if fname_m else 'recording.mp4'
                safe_name = _vault_safe(raw_name) or 'recording.mp4'
                fpath = os.path.join(target_dir, safe_name)
                if os.path.exists(fpath):
                    base, e = os.path.splitext(safe_name)
                    safe_name = f'{base}_{int(datetime.now().timestamp())}{e}'
                    fpath = os.path.join(target_dir, safe_name)
                job['filename'] = safe_name
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                with open(fpath, 'wb') as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        job['progress'] = round(downloaded / total * 100) if total else -1

        job['status'] = 'done'
        job['progress'] = 100
    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)

class ImportUrlBody(BaseModel):
    url: str
    folder: str = ''

@app.post('/call/import-url')
def import_call_from_url(body: ImportUrlBody):
    """Start background download of a call recording from a URL (incl. Google Drive)."""
    url = body.url.strip()
    folder = body.folder.strip()
    if not url:
        raise HTTPException(400, 'URL required')
    if folder and folder not in CALL_FOLDERS:
        raise HTTPException(400, f'Invalid folder. Choose from: {CALL_FOLDERS}')
    job_id = str(uuid.uuid4())[:8]
    _import_jobs[job_id] = {'status': 'downloading', 'progress': 0, 'filename': '', 'error': '', 'folder': folder}
    t = threading.Thread(target=_run_import, args=(job_id, url, folder), daemon=True)
    t.start()
    return {'job_id': job_id}

@app.get('/call/import-status/{job_id}')
def import_call_status(job_id: str):
    job = _import_jobs.get(job_id)
    if not job:
        raise HTTPException(404, 'Job not found')
    return job

# ─────────────────────────────────────────────────────────────────────────────

@app.delete('/call/file/{filename}')
def delete_call_file(filename: str, folder: str = ''):
    if '..' in filename or '..' in folder:
        raise HTTPException(400, 'Invalid')
    base_dir = os.path.join(CALLS_DIR, folder) if folder else CALLS_DIR
    fpath = os.path.join(base_dir, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, 'Not found')
    os.remove(fpath)
    return {'ok': True}

@app.get('/call/file/{filename}')
def serve_call_file(filename: str, folder: str = ''):
    if '..' in filename or '..' in folder:
        raise HTTPException(400, 'Invalid')
    base_dir = os.path.join(CALLS_DIR, folder) if folder else CALLS_DIR
    fpath = os.path.join(base_dir, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, 'Not found')
    return FileResponse(fpath)

class CallStartIn(BaseModel):
    tg_id: str
    filename: str = ''
    folder: str = ''   # '' = root, 'fake_checks', 'paid_calls'
    chatter: str = 'Chatter'
    url: str = ''      # Google Drive or direct URL — skips local file lookup


async def _call_timed_hangup(tg_id_str: str, media_src: str):
    """Auto-hang up after the media duration + 3s buffer."""
    try:
        import subprocess, json as _json
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', media_src],
            capture_output=True, text=True, timeout=10
        )
        info = _json.loads(result.stdout)
        duration = float(info['format']['duration'])
        print(f'⏱ Auto-hangup in {duration:.1f}s for {tg_id_str}')
    except Exception as _e:
        print(f'⚠️ ffprobe failed ({_e}) — using 300s fallback')
        duration = 300.0
    await asyncio.sleep(duration + 3)
    if tg_id_str not in active_calls:
        return  # already hung up
    print(f'⏰ Auto-hanging up {tg_id_str}')
    try:
        if hasattr(calls_client, 'leave_call'):
            await calls_client.leave_call(int(tg_id_str))
        elif hasattr(calls_client, 'leave'):
            await calls_client.leave(int(tg_id_str))
    except Exception as _e:
        print(f'⚠️ Auto-hangup leave_call error: {_e}')
    active_calls.pop(tg_id_str, None)
    await ws_manager.broadcast({'type': 'call_ended', 'tg_id': tg_id_str})


async def _run_call_play(tg_id_str: str, peer: int, stream, media_src: str):
    """Background task: actually connect the call via py-tgcalls.
    Returns immediately so /call/start responds in <1s.
    On failure sends call_ended WS so frontend can alert the chatter."""
    global calls_client, _tg_priority
    _tg_priority += 1  # pause broadcast while call is connecting
    try:
        async def _do_play():
            if hasattr(calls_client, 'play'):
                await calls_client.play(peer, stream)
            elif hasattr(calls_client, 'call'):
                await calls_client.call(peer, stream)
            else:
                raise RuntimeError(
                    f'No play/call method. Available: '
                    f'{[m for m in dir(calls_client) if not m.startswith("_")]}'
                )

        _last_err = None
        for _attempt in range(3):
            try:
                # 60s: enough time for subscriber to see ring and pick up
                await asyncio.wait_for(_do_play(), timeout=60)
                _last_err = None
                break  # success
            except asyncio.TimeoutError:
                print(f'⏰ play() timed out for {tg_id_str} — subscriber did not answer')
                try:
                    if calls_client and hasattr(calls_client, 'leave_call'):
                        await asyncio.wait_for(calls_client.leave_call(peer), timeout=5)
                except Exception:
                    pass
                if tg_client:
                    asyncio.create_task(_reinit_calls(tg_client))
                active_calls.pop(tg_id_str, None)
                await ws_manager.broadcast({
                    'type': 'call_ended', 'tg_id': tg_id_str,
                    'reason': 'unanswered',
                    'msg': 'Subscriber hat nicht abgehoben (60s).',
                })
                return
            except Exception as _e:
                _last_err = _e
                err_str = str(_e)
                if ('DH_G_A_HASH_INVALID' in err_str or 'HASH_INVALID' in err_str) and _attempt < 2:
                    print(f'🔄 DH error attempt {_attempt+1} — reiniting calls_client')
                    if tg_client and _PYTGCALLS_OK:
                        try:
                            if calls_client:
                                try: await calls_client.stop()
                                except Exception: pass
                            calls_client = PyTgCalls(tg_client)
                            await calls_client.start()
                            print(f'✅ calls_client restarted for retry {_attempt+2}')
                        except Exception as _re:
                            print(f'⚠️ reinit failed: {_re}')
                    await asyncio.sleep(2)
                    continue
                break  # non-DH error or exhausted retries

        if _last_err is not None:
            print(f'❌ Call failed after retries for {tg_id_str}: {_last_err}')
            active_calls.pop(tg_id_str, None)
            await ws_manager.broadcast({
                'type': 'call_ended', 'tg_id': tg_id_str,
                'reason': 'error',
                'msg': str(_last_err),
            })
            return

        # ── Call connected ──────────────────────────────────────────────────
        print(f'✅ Call connected for {tg_id_str}')
        asyncio.create_task(_call_timed_hangup(tg_id_str, media_src))

    except Exception as _outer:
        import traceback
        print(f'❌ _run_call_play error for {tg_id_str}: {traceback.format_exc()}')
        active_calls.pop(tg_id_str, None)
        await ws_manager.broadcast({
            'type': 'call_ended', 'tg_id': tg_id_str,
            'reason': 'error', 'msg': str(_outer),
        })
    finally:
        _tg_priority = max(0, _tg_priority - 1)  # always resume broadcast


@app.post('/call/start')
async def start_fake_call(body: CallStartIn):
    """Initiate a pre-recorded call to a subscriber via Telegram."""
    global calls_client
    if not _PYTGCALLS_OK:
        raise HTTPException(503, 'pytgcalls not installed on Railway. Add "py-tgcalls" to requirements.txt and redeploy.')
    if not calls_client:
        raise HTTPException(503, 'Calls client not ready yet (wait a few seconds after startup).')
    # Telegram only supports ONE active voice/video call per account at a time.
    # Block immediately if any call is running — avoids PyTgCalls timeout/hang.
    if active_calls:
        other = [k for k in active_calls if k != body.tg_id]
        if body.tg_id in active_calls:
            raise HTTPException(409, 'Ein Call mit diesem Subscriber läuft bereits.')
        if other:
            raise HTTPException(409, f'Es läuft bereits ein anderer Call (Subscriber {other[0]}). Bitte erst den laufenden Call beenden.')


    # ── Resolve file path OR direct/Drive URL ─────────────────────────────────
    stream_url = None   # set if streaming from URL instead of local file
    fpath = None

    if body.url:
        # Convert Google Drive share URL → direct download URL
        file_id = _gdrive_file_id(body.url)
        if file_id:
            stream_url = (
                f'https://drive.usercontent.google.com/download'
                f'?id={file_id}&export=download&authuser=0&confirm=t'
            )
        else:
            stream_url = body.url   # already a direct URL
    else:
        if not body.filename:
            raise HTTPException(400, 'Provide either filename or url.')
        base_dir = os.path.join(CALLS_DIR, body.folder) if body.folder else CALLS_DIR
        fpath = os.path.join(base_dir, body.filename)
        if '..' in body.filename or not os.path.isfile(fpath):
            raise HTTPException(404, 'Recording file not found.')

    with db() as conn:
        with conn.cursor() as c:
            c.execute('SELECT tg_access_hash FROM conversations WHERE tg_id=%s', (body.tg_id,))
            row = c.fetchone()

    peer = int(body.tg_id)

    # Pre-populate Telethon's SQLite session so py-tgcalls can resolve the user.
    ah = int(row['tg_access_hash']) if row and row.get('tg_access_hash') else 0
    if tg_client:
        resolved = False
        # Step 1: GetUsersRequest returns a full User object which we then
        # pass through get_input_entity() to persist it in the SQLite session.
        if ah:
            try:
                from telethon.tl.types import InputUser as _IU
                from telethon.tl.functions.users import GetUsersRequest as _GUR
                users = await tg_client(_GUR(id=[_IU(user_id=peer, access_hash=ah)]))
                if users:
                    await tg_client.get_input_entity(users[0])
                    print(f'✅ Entity cached via GetUsers for {peer}')
                    resolved = True
            except Exception as _e:
                print(f'⚠️ GetUsers+cache failed: {_e}')
        # Step 2: Fallback — scan recent dialogs to find and cache the user (with timeout)
        if not resolved:
            try:
                async def _scan_dialogs():
                    async for _dlg in tg_client.iter_dialogs(limit=200):
                        if getattr(_dlg.entity, 'id', None) == peer:
                            await tg_client.get_input_entity(_dlg.entity)
                            return True
                    return False
                resolved = await asyncio.wait_for(_scan_dialogs(), timeout=10)
                if resolved:
                    print(f'✅ Entity found in dialogs for {peer}')
            except asyncio.TimeoutError:
                print(f'⚠️ Dialog scan timed out for {peer}')
            except Exception as _e:
                print(f'⚠️ Dialog scan failed: {_e}')
        if not resolved:
            raise HTTPException(400, f'Cannot resolve Telegram user {peer}. The userbot may not have chatted with this user yet.')

    media_source = stream_url if stream_url else fpath
    label = stream_url if stream_url else body.filename
    # Determine if video by extension (from URL path or filename)
    src_name = (stream_url or body.filename or '').split('?')[0]
    ext = src_name.rsplit('.',1)[-1].lower() if '.' in src_name else ''
    is_video = ext in ('mp4','mov','mkv') or 'video' in (stream_url or '')

    # For Drive URLs without a clear extension, assume video/mp4
    if stream_url and not ext:
        is_video = True

    # ── Build stream object (synchronous, fast) ──────────────────────────────
    try:
        if is_video:
            try:
                from pytgcalls.types import VideoQuality
                stream = MediaStream(
                    media_source,
                    audio_parameters=AudioQuality.HIGH,
                    video_parameters=VideoQuality.HD_720p,
                )
                print(f'🎬 Video stream: {media_source} HD_720p+audio')
            except Exception as _vq_e:
                print(f'⚠️ VideoQuality.HD_720p failed ({_vq_e}), falling back to audio-only')
                stream = MediaStream(media_source, audio_parameters=AudioQuality.HIGH)
        else:
            stream = MediaStream(media_source, audio_parameters=AudioQuality.HIGH)
    except Exception as e:
        raise HTTPException(500, f'Stream build error: {e}')

    # ── Register call immediately so duplicate-call check works ──────────────
    now_ts = datetime.now().isoformat()
    active_calls[body.tg_id] = {
        'file': label,
        'chatter': body.chatter,
        'started_at': now_ts,
        'type': 'video' if is_video else 'audio',
    }
    save_msg(body.tg_id, f'[📞 Pre-recorded {"Video" if is_video else "Audio"} Call – {label}]', 'out', body.chatter)
    asyncio.create_task(ws_manager.broadcast({
        'type': 'call_started',
        'tg_id': body.tg_id,
        'file': label,
        'call_type': 'video' if is_video else 'audio',
        'chatter': body.chatter,
    }))

    # ── Fire-and-forget: play() in background ────────────────────────────────
    # play() blocks until subscriber answers (can take 30–60s).
    # Running it in background lets this endpoint return in <1s so the chatter
    # sees the call ring immediately without waiting.
    asyncio.create_task(_run_call_play(body.tg_id, peer, stream, media_source))

    return {'ok': True, 'type': 'video' if is_video else 'audio'}

@app.post('/call/stop')
async def stop_fake_call(tg_id: str):
    """Hang up the active call with a subscriber."""
    if not calls_client:
        raise HTTPException(503, 'Calls client not ready.')
    peer = int(tg_id)
    try:
        if hasattr(calls_client, 'leave_call'):
            await calls_client.leave_call(peer)
        elif hasattr(calls_client, 'leave'):
            await calls_client.leave(peer)
    except Exception as e:
        print(f'leave_call error (may already be ended): {e}')
    active_calls.pop(tg_id, None)
    asyncio.create_task(ws_manager.broadcast({'type': 'call_ended', 'tg_id': tg_id}))
    return {'ok': True}

# ── START ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT)
