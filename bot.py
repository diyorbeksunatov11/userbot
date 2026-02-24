import asyncio
import logging
import mimetypes
import os
import tempfile
import time
from collections import deque

from telethon import TelegramClient, events
from google import genai
from openai import OpenAI

# =========================
# REQUIRED ENV
# =========================
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]

if not API_ID or not API_HASH:
    raise RuntimeError("API_ID va API_HASH ENV da bo'lishi shart.")
if not GEMINI_KEYS:
    raise RuntimeError("GEMINI_KEYS (kamida 1 ta) ENV da bo'lishi shart.")

# =========================
# DEFAULTS
# =========================
SESSION_NAME = os.getenv("SESSION_NAME", "shaxsiy_sessiya")
MEMORY_LEN = int(os.getenv("MEMORY_LEN", "10"))
CONTACTS_ONLY = os.getenv("CONTACTS_ONLY", "0") == "1"
DEBUG_ECHO_ONLY = os.getenv("DEBUG_ECHO_ONLY", "0") == "1"

# Anti-limit
LLM_MIN_DELAY = float(os.getenv("LLM_MIN_DELAY", "0.6"))        # Groq/OpenRouter
GEMINI_MIN_DELAY = float(os.getenv("GEMINI_MIN_DELAY", "1.0"))  # Gemini

# Blocklist
BLOCK_USER_IDS = set(int(x) for x in os.getenv("BLOCK_USER_IDS", "").split(",") if x.strip().isdigit())

# Models (xohlasangiz ENV bilan o'zgartirasiz)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # kerak bo'lsa: "models/gemini-2.5-flash"

# =========================
# GLOBALS
# =========================
memory: dict[int, deque] = {}
tg_client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

_llm_lock = asyncio.Lock()
_last_llm_ts = 0.0

_gemini_lock = asyncio.Lock()
_last_gemini_ts = 0.0

# OpenAI-compatible clients
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1") if OPENROUTER_API_KEY else None


# =========================
# HEALTH SERVER
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
# HELPERS
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

def build_text_prompt(user_name: str, history_context: str) -> str:
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. "
        f"Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"Suhbat tarixi:\n{history_context}\n\n"
        f"Vazifa: Faqat o'zbek tilida, samimiy, qisqa va mantiqiy javob ber."
    )

def build_image_prompt(user_name: str, history_context: str) -> str:
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. "
        f"Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"Suhbat tarixi:\n{history_context}\n\n"
        f"Foydalanuvchi rasm yubordi. Rasmni tahlil qilib o'zbekcha qisqa javob ber."
    )

def build_audio_prompt(user_name: str, history_context: str) -> str:
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. "
        f"Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"Suhbat tarixi:\n{history_context}\n\n"
        f"Foydalanuvchi voice/audio yubordi. Mazmunini tushunib o'zbekcha qisqa javob ber."
    )


# =========================
# TEXT LLM ROUTER: Groq -> OpenRouter -> Gemini(2 key)
# =========================
async def _llm_delay():
    global _last_llm_ts
    now = time.time()
    wait = (_last_llm_ts + LLM_MIN_DELAY) - now
    if wait > 0:
        await asyncio.sleep(wait)

async def _call_openai_compat(client: OpenAI, model: str, prompt: str, extra_headers: dict | None = None) -> str:
    def _run():
        kwargs = {}
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return (r.choices[0].message.content or "").strip()
    return await asyncio.to_thread(_run)

async def _gemini_delay():
    global _last_gemini_ts
    now = time.time()
    wait = (_last_gemini_ts + GEMINI_MIN_DELAY) - now
    if wait > 0:
        await asyncio.sleep(wait)

async def _gemini_text(prompt: str) -> str:
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
                msg = str(e)
                if ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate" in msg.lower()):
                    continue
                raise

        _last_gemini_ts = time.time()
        raise RuntimeError(f"Gemini hamma keylarda limit/xato. Oxirgi xato: {last_err}")

async def generate_text_router(prompt: str) -> str:
    global _last_llm_ts
    # Bitta vaqtda bitta text call (spam bo'lmasin)
    async with _llm_lock:
        await _llm_delay()

        # 1) Groq
        if groq_client:
            try:
                txt = await _call_openai_compat(groq_client, GROQ_MODEL, prompt)
                _last_llm_ts = time.time()
                if txt:
                    return txt
            except Exception as e:
                msg = str(e)
                if ("429" not in msg) and ("rate" not in msg.lower()):
                    # Groq boshqa xato bersa ham fallback qilamiz (bot yiqilmasin)
                    logging.warning("Groq error: %s", msg[:200])

        # 2) OpenRouter
        if openrouter_client:
            try:
                # OpenRouter odatda referer/title header tavsiya qiladi (ixtiyoriy)
                headers = {
                    "HTTP-Referer": "https://example.local",
                    "X-Title": "Telegram Userbot"
                }
                txt = await _call_openai_compat(openrouter_client, OPENROUTER_MODEL, prompt, extra_headers=headers)
                _last_llm_ts = time.time()
                if txt:
                    return txt
            except Exception as e:
                msg = str(e)
                if ("429" not in msg) and ("rate" not in msg.lower()):
                    logging.warning("OpenRouter error: %s", msg[:200])

        # 3) Gemini fallback (2 key)
        return await _gemini_text(prompt)


# =========================
# MEDIA: Gemini Files (rasm/voice)
# =========================
async def upload_file_to_gemini(file_path: str, mime_type: str):
    global _last_gemini_ts
    async with _gemini_lock:
        await _gemini_delay()

        last_err = None
        for key in GEMINI_KEYS:
            try:
                client = genai.Client(api_key=key)

                def _run():
                    # SDK versiyaga qarab farq qilishi mumkin — 3 variant
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
                msg = str(e)
                if ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate" in msg.lower()):
                    continue
                raise

        _last_gemini_ts = time.time()
        raise RuntimeError(f"Gemini upload hamma keylarda limit/xato. Oxirgi xato: {last_err}")

async def gemini_media_reply(prompt: str, uploaded_file):
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
                msg = str(e)
                if ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate" in msg.lower()):
                    continue
                raise

        _last_gemini_ts = time.time()
        raise RuntimeError(f"Gemini media reply hamma keylarda limit/xato. Oxirgi xato: {last_err}")

async def handle_media_with_gemini(event, user_id: int, user_name: str, media_type: str) -> str:
    tmp_path = None
    try:
        suffix = ".jpg" if media_type == "image" else ".ogg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        downloaded = await event.download_media(file=tmp_path)
        if not downloaded:
            return "Media yuklab bo'lmadi."

        file_path = downloaded if isinstance(downloaded, str) else tmp_path
        mime_type = detect_mime(event, file_path, media_type)

        if DEBUG_ECHO_ONLY:
            return f"Test {media_type} ✅"

        history = "\n".join(memory.get(user_id, []))
        prompt = build_image_prompt(user_name, history) if media_type == "image" else build_audio_prompt(user_name, history)

        uploaded = await upload_file_to_gemini(file_path, mime_type)
        return await gemini_media_reply(prompt, uploaded) or "Tushunmadim."

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================
# TELEGRAM HANDLER
# =========================
@tg_client.on(events.NewMessage(incoming=True))
async def pro_handler(event):
    if not event.is_private or event.out:
        return

    sender = await event.get_sender()
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
            if DEBUG_ECHO_ONLY:
                await event.reply("Test reply ✅")
                return

            memory[user_id].append(f"Foydalanuvchi: {event.text}")
            history = "\n".join(memory[user_id])
            prompt = build_text_prompt(user_name, history)

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
            async with tg_client.action(user_id, "typing"):
                reply = await handle_media_with_gemini(event, user_id, user_name, "image")
                memory[user_id].append(f"Sen: {reply}")
                await event.reply(reply)
            return

        # AUDIO / VOICE
        is_audio = False
        if event.document and getattr(event.document, "mime_type", ""):
            mime = event.document.mime_type
            if mime.startswith("audio/") or mime in ("application/ogg", "application/octet-stream"):
                is_audio = True
        if not is_audio and (getattr(event, "voice", None) or getattr(event, "audio", None)):
            is_audio = True

        if is_audio:
            memory[user_id].append("Foydalanuvchi: [Voice yubordi]")
            async with tg_client.action(user_id, "typing"):
                reply = await handle_media_with_gemini(event, user_id, user_name, "audio")
                memory[user_id].append(f"Sen: {reply}")
                await event.reply(reply)
            return

    except Exception:
        logging.exception("HANDLER ERROR")
        try:
            await event.reply("Xatolik bo'ldi, keyinroq yana urinib ko'ring.")
        except Exception:
            pass


# =========================
# MAIN
# =========================
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    health = await start_health_server()
    try:
        await tg_client.start()
        me = await tg_client.get_me()
        logging.info("✅ BOT FAOL: @%s", getattr(me, "username", None) or me.id)
        logging.info("Providers: groq=%s openrouter=%s gemini_keys=%d",
                     bool(groq_client), bool(openrouter_client), len(GEMINI_KEYS))
        await tg_client.run_until_disconnected()
    finally:
        health.close()
        await health.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
