# bot.py
import asyncio
import logging
import mimetypes
import os
import tempfile
import time
from collections import deque

from telethon import TelegramClient, events
from google import genai

# Optional (text fallback providers)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# ENV (minimal)
# =========================
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "").strip()

# Gemini keys: GEMINI_KEYS="k1,k2"  (fallback: GEMINI_API_KEY="k1")
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]
if not GEMINI_KEYS:
    one = os.getenv("GEMINI_API_KEY", "").strip()
    if one:
        GEMINI_KEYS = [one]

# Optional providers
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Session + behavior
SESSION_NAME = os.getenv("SESSION_NAME", "koyeb_session")
CONTACTS_ONLY = os.getenv("CONTACTS_ONLY", "0") == "1"
MEMORY_LEN = int(os.getenv("MEMORY_LEN", "10"))

# Ignore senders
IGNORE_BOTS = os.getenv("IGNORE_BOTS", "1") == "1"  # botlardan kelgan xabarlarni ignore
BLOCK_USER_IDS = set(int(x) for x in os.getenv("BLOCK_USER_IDS", "").split(",") if x.strip().isdigit())

# Models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # or "models/gemini-2.5-flash"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

# Rate limits / cooldown
GEMINI_MIN_DELAY = float(os.getenv("GEMINI_MIN_DELAY", "1.0"))
LLM_MIN_DELAY = float(os.getenv("LLM_MIN_DELAY", "0.7"))

GEMINI_COOLDOWN_SEC = int(os.getenv("GEMINI_COOLDOWN_SEC", "120"))
GROQ_COOLDOWN_SEC = int(os.getenv("GROQ_COOLDOWN_SEC", "60"))
OPENROUTER_COOLDOWN_SEC = int(os.getenv("OPENROUTER_COOLDOWN_SEC", "60"))
GEMINI_CHECK_INTERVAL_SEC = int(os.getenv("GEMINI_CHECK_INTERVAL_SEC", "30"))

# Health server
ENABLE_HEALTH = os.getenv("ENABLE_HEALTH", "1") == "1"


# =========================
# Validate required
# =========================
if not API_ID or not API_HASH:
    raise RuntimeError("API_ID va API_HASH ENV da bo‘lishi shart.")
if not GEMINI_KEYS:
    raise RuntimeError("GEMINI_KEYS yoki GEMINI_API_KEY ENV da bo‘lishi shart.")


# =========================
# Globals
# =========================
tg_client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

memory: dict[int, deque] = {}
summary_memory: dict[int, str] = {}  # user_id -> short summary

_gemini_lock = asyncio.Lock()
_last_gemini_ts = 0.0

_llm_lock = asyncio.Lock()
_last_llm_ts = 0.0

# Provider cooldown state
PROVIDER_STATE = {
    "gemini": {"cooldown_until": 0.0, "next_check": 0.0},
    "groq": {"cooldown_until": 0.0},
    "openrouter": {"cooldown_until": 0.0},
}

# OpenAI-compatible clients (optional)
groq_client = None
openrouter_client = None
if OpenAI is not None:
    if GROQ_API_KEY:
        groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    if OPENROUTER_API_KEY:
        openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


# =========================
# Health server (for Koyeb-like envs)
# =========================
async def start_health_server():
    port = int(os.environ.get("PORT", "8000"))

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            await reader.read(1024)
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain\r\n"
                b"Content-Length: 2\r\n\r\n"
                b"ok"
            )
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(handle, host="0.0.0.0", port=port)
    logging.info("Health server listening on 0.0.0.0:%s", port)
    return server


# =========================
# Helpers
# =========================
def ensure_memory(user_id: int):
    if user_id not in memory:
        memory[user_id] = deque(maxlen=MEMORY_LEN)

def safe_name(sender) -> str:
    return (getattr(sender, "first_name", None) or "Do'stim").strip()

def is_blocked(sender) -> bool:
    sid = getattr(sender, "id", None)
    return isinstance(sid, int) and sid in BLOCK_USER_IDS

def detect_mime(event, file_path: str, media_type: str) -> str:
    if getattr(event, "document", None) and getattr(event.document, "mime_type", None):
        return event.document.mime_type
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        return mime
    return "image/jpeg" if media_type == "image" else "audio/ogg"

def _is_rate_or_quota(msg_lower: str) -> bool:
    return ("429" in msg_lower) or ("resource_exhausted" in msg_lower) or ("rate" in msg_lower)

def _is_leaked_key(msg_lower: str) -> bool:
    return ("403" in msg_lower) and ("reported as leaked" in msg_lower)

def build_text_prompt(user_name: str, user_id: int) -> str:
    last_lines = "\n".join(list(memory.get(user_id, []))[-6:])
    summ = (summary_memory.get(user_id, "") or "").strip()
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"KONTEKST (xulosa): {summ}\n\n"
        f"OXIRGI XABARLAR:\n{last_lines}\n\n"
        f"Vazifa: faqat o'zbek tilida samimiy, qisqa va mantiqiy javob ber."
    )

def build_image_prompt(user_name: str, user_id: int) -> str:
    last_lines = "\n".join(list(memory.get(user_id, []))[-6:])
    summ = (summary_memory.get(user_id, "") or "").strip()
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"KONTEKST (xulosa): {summ}\n\n"
        f"OXIRGI XABARLAR:\n{last_lines}\n\n"
        f"Foydalanuvchi rasm yubordi. Rasmni tahlil qilib o'zbekcha qisqa javob ber."
    )

def build_audio_prompt(user_name: str, user_id: int) -> str:
    last_lines = "\n".join(list(memory.get(user_id, []))[-6:])
    summ = (summary_memory.get(user_id, "") or "").strip()
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"KONTEKST (xulosa): {summ}\n\n"
        f"OXIRGI XABARLAR:\n{last_lines}\n\n"
        f"Foydalanuvchi voice/audio yubordi. Mazmunini tushunib o'zbekcha qisqa javob ber."
    )


# =========================
# Gemini (text + media) with multi-key rotation
# =========================
async def _gemini_delay():
    global _last_gemini_ts
    now = time.time()
    wait = (_last_gemini_ts + GEMINI_MIN_DELAY) - now
    if wait > 0:
        await asyncio.sleep(wait)

async def gemini_text(prompt: str) -> str:
    global _last_gemini_ts
    async with _gemini_lock:
        await _gemini_delay()

        last_err = None
        for key in GEMINI_KEYS:
            try:
                client = genai.Client(api_key=key)

                def _run():
                    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                    return (getattr(resp, "text", None) or "").strip()

                txt = await asyncio.to_thread(_run)
                _last_gemini_ts = time.time()
                return txt

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if _is_rate_or_quota(msg) or _is_leaked_key(msg):
                    continue
                raise

        _last_gemini_ts = time.time()
        raise RuntimeError(f"Gemini ishlamadi (hamma key). Oxirgi xato: {last_err}")

async def gemini_health_ping() -> bool:
    try:
        txt = await gemini_text("Faqat 'ok' deb javob ber.")
        return (txt or "").strip().lower().startswith("ok")
    except Exception:
        return False

async def gemini_upload(file_path: str, mime_type: str):
    global _last_gemini_ts
    async with _gemini_lock:
        await _gemini_delay()

        last_err = None
        for key in GEMINI_KEYS:
            try:
                client = genai.Client(api_key=key)

                def _run():
                    try:
                        return client.files.upload(file=file_path, config={"mime_type": mime_type})
                    except TypeError:
                        pass
                    try:
                        return client.files.upload(file=file_path, mime_type=mime_type)
                    except TypeError:
                        pass
                    return client.files.upload(file=file_path)

                f = await asyncio.to_thread(_run)
                _last_gemini_ts = time.time()
                return f

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if _is_rate_or_quota(msg) or _is_leaked_key(msg):
                    continue
                raise

        _last_gemini_ts = time.time()
        raise RuntimeError(f"Gemini upload ishlamadi (hamma key). Oxirgi xato: {last_err}")

async def gemini_media_reply(prompt: str, uploaded_file) -> str:
    global _last_gemini_ts
    async with _gemini_lock:
        await _gemini_delay()

        last_err = None
        for key in GEMINI_KEYS:
            try:
                client = genai.Client(api_key=key)

                def _run():
                    resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, uploaded_file])
                    return (getattr(resp, "text", None) or "").strip()

                txt = await asyncio.to_thread(_run)
                _last_gemini_ts = time.time()
                return txt

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if _is_rate_or_quota(msg) or _is_leaked_key(msg):
                    continue
                raise

        _last_gemini_ts = time.time()
        raise RuntimeError(f"Gemini media reply ishlamadi (hamma key). Oxirgi xato: {last_err}")


# =========================
# Text fallback providers (Groq/OpenRouter)
# =========================
async def _llm_delay():
    global _last_llm_ts
    now = time.time()
    wait = (_last_llm_ts + LLM_MIN_DELAY) - now
    if wait > 0:
        await asyncio.sleep(wait)

async def _call_openai_compat(client, model: str, prompt: str, extra_headers: dict | None = None) -> str:
    def _run():
        kwargs = {}
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Faqat o'zbek tilida yoz. Imlo va tinish belgilarini to'g'ri qo'lla. Qisqa va aniq bo'l."},
                {"role": "user", "content": prompt},
            ],
            **kwargs
        )
        return (r.choices[0].message.content or "").strip()
    return await asyncio.to_thread(_run)


# =========================
# Provider router (Gemini -> Groq -> OpenRouter) + Gemini recheck
# =========================
async def generate_text_router(prompt: str) -> str:
    global _last_llm_ts

    now = time.time()
    gstate = PROVIDER_STATE["gemini"]

    # If Gemini is in cooldown, periodically check if it recovered
    if now < gstate.get("cooldown_until", 0.0):
        if now >= gstate.get("next_check", 0.0):
            ok = await gemini_health_ping()
            gstate["next_check"] = now + GEMINI_CHECK_INTERVAL_SEC
            if ok:
                gstate["cooldown_until"] = 0.0

    # 1) Try Gemini if not cooling down
    if time.time() >= gstate.get("cooldown_until", 0.0):
        try:
            return await gemini_text(prompt)
        except Exception as e:
            msg = str(e).lower()
            if _is_rate_or_quota(msg) or _is_leaked_key(msg):
                gstate["cooldown_until"] = time.time() + GEMINI_COOLDOWN_SEC
                gstate["next_check"] = time.time() + GEMINI_CHECK_INTERVAL_SEC
            else:
                gstate["cooldown_until"] = time.time() + 10
                gstate["next_check"] = time.time() + GEMINI_CHECK_INTERVAL_SEC

    # 2) Fallback Groq
    if groq_client and time.time() >= PROVIDER_STATE["groq"].get("cooldown_until", 0.0):
        try:
            async with _llm_lock:
                await _llm_delay()
                txt = await _call_openai_compat(groq_client, GROQ_MODEL, prompt)
                _last_llm_ts = time.time()
                if txt:
                    return txt
        except Exception as e:
            msg = str(e).lower()
            if _is_rate_or_quota(msg):
                PROVIDER_STATE["groq"]["cooldown_until"] = time.time() + GROQ_COOLDOWN_SEC

    # 3) Fallback OpenRouter
    if openrouter_client and time.time() >= PROVIDER_STATE["openrouter"].get("cooldown_until", 0.0):
        try:
            async with _llm_lock:
                await _llm_delay()
                headers = {"HTTP-Referer": "https://example.local", "X-Title": "Telegram Userbot"}
                txt = await _call_openai_compat(openrouter_client, OPENROUTER_MODEL, prompt, extra_headers=headers)
                _last_llm_ts = time.time()
                if txt:
                    return txt
        except Exception as e:
            msg = str(e).lower()
            if _is_rate_or_quota(msg):
                PROVIDER_STATE["openrouter"]["cooldown_until"] = time.time() + OPENROUTER_COOLDOWN_SEC

    return "Hozircha javob bera olmadim. Keyinroq yana urinib ko‘ring."


# =========================
# Rolling summary (to help “memory”)
# =========================
async def maybe_update_summary(user_id: int):
    if len(memory.get(user_id, [])) < 8:
        return

    prev = (summary_memory.get(user_id, "") or "").strip()
    last_lines = "\n".join(list(memory[user_id])[-8:])

    sum_prompt = (
        "Suhbatdagi muhim kontekstni (foydalanuvchining niyati, muammo, kelishuvlar) 2-4 jumlada xulosa qil. "
        "Faqat o'zbek tilida. Keraksiz tafsilotsiz.\n\n"
        f"Oldingi xulosa: {prev}\n\n"
        f"Yangi parcha:\n{last_lines}"
    )

    try:
        new_sum = await generate_text_router(sum_prompt)
        if new_sum:
            summary_memory[user_id] = new_sum.strip()
    except Exception:
        pass


# =========================
# Media handler (Gemini only)
# =========================
async def handle_media(event, user_id: int, user_name: str, media_type: str) -> str:
    tmp_path = None
    try:
        suffix = ".jpg" if media_type == "image" else ".ogg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        downloaded = await event.download_media(file=tmp_path)
        if not downloaded:
            return "Media faylni yuklab bo‘lmadi."

        file_path = downloaded if isinstance(downloaded, str) else tmp_path
        mime_type = detect_mime(event, file_path, media_type)

        prompt = build_image_prompt(user_name, user_id) if media_type == "image" else build_audio_prompt(user_name, user_id)
        uploaded = await gemini_upload(file_path, mime_type)
        return await gemini_media_reply(prompt, uploaded) or "Tushunmadim."

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================
# Telegram handler
# =========================
@tg_client.on(events.NewMessage(incoming=True))
async def pro_handler(event):
    if not event.is_private or event.out:
        return

    sender = await event.get_sender()

    # Ignore bot accounts
    if IGNORE_BOTS and getattr(sender, "bot", False):
        return

    if is_blocked(sender):
        return

    if CONTACTS_ONLY and not getattr(sender, "contact", False):
        return

    user_id = event.chat_id
    user_name = safe_name(sender)
    ensure_memory(user_id)

    try:
        # TEXT
        if event.text:
            memory[user_id].append(f"Foydalanuvchi: {event.text}")
            await maybe_update_summary(user_id)

            prompt = build_text_prompt(user_name, user_id)

            async with tg_client.action(user_id, "typing"):
                reply = await generate_text_router(prompt)
                reply = reply or "Javob chiqmadi."
                memory[user_id].append(f"Sen: {reply}")
                await asyncio.sleep(min(len(reply) * 0.04, 4))
                await event.reply(reply)
            return

        # IMAGE
        is_image = bool(event.photo)
        if not is_image and event.document and getattr(event.document, "mime_type", ""):
            is_image = event.document.mime_type.startswith("image/")
        if is_image:
            memory[user_id].append("Foydalanuvchi: [Rasm yubordi]")
            await maybe_update_summary(user_id)
            async with tg_client.action(user_id, "typing"):
                reply = await handle_media(event, user_id, user_name, "image")
                memory[user_id].append(f"Sen: {reply}")
                await event.reply(reply)
            return

        # AUDIO/VOICE
        is_audio = False
        if event.document and getattr(event.document, "mime_type", ""):
            mime = event.document.mime_type
            if mime.startswith("audio/") or mime in ("application/ogg", "application/octet-stream"):
                is_audio = True
        if not is_audio and (getattr(event, "voice", None) or getattr(event, "audio", None)):
            is_audio = True

        if is_audio:
            memory[user_id].append("Foydalanuvchi: [Voice/Audio yubordi]")
            await maybe_update_summary(user_id)
            async with tg_client.action(user_id, "typing"):
                reply = await handle_media(event, user_id, user_name, "audio")
                memory[user_id].append(f"Sen: {reply}")
                await event.reply(reply)
            return

    except Exception:
        logging.exception("HANDLER ERROR")
        try:
            await event.reply("Xatolik bo‘ldi, keyinroq yana urinib ko‘ring.")
        except Exception:
            pass


# =========================
# Main
# =========================
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    health = None
    if ENABLE_HEALTH:
        health = await start_health_server()

    try:
        await tg_client.start()
        me = await tg_client.get_me()
        logging.info("✅ BOT FAOL: @%s", getattr(me, "username", None) or me.id)
        logging.info(
            "Providers: gemini_keys=%d groq=%s openrouter=%s | ignore_bots=%s contacts_only=%s",
            len(GEMINI_KEYS),
            bool(groq_client),
            bool(openrouter_client),
            IGNORE_BOTS,
            CONTACTS_ONLY,
        )
        await tg_client.run_until_disconnected()
    finally:
        if health:
            health.close()
            await health.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
