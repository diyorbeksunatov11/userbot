import asyncio
import logging
import mimetypes
import os
import tempfile
import time
from collections import deque

from telethon import TelegramClient, events
from google import genai

# =========================================================
# TELEGRAM (ENV tavsiya, lekin bo'lmasa pastdagini qo'ying)
# =========================================================
API_ID = int(os.getenv("API_ID", "36820193"))
API_HASH = os.getenv("API_HASH", "d43f9550fdfd6dbea8719112a584bee9")

# =========================================================
# GEMINI API KEYS (multi-key)  ⚠️ AMALDA ENV TAVSIYA
# =========================================================
GEMINI_KEYS = [
    "AIzaSyDX49C8wPtSZJAtKkQHNMTQlTkLscMVO9E",
    "AIzaSyBvDbJzjMW5-U8dNFxnQX5vWkj1zRq36Ik",
    "AIzaSyAvfyhVS04pZ8Wc4_fE1jjdPKFoyez2IJg",
    "AIzaSyCOiAGT7VtxQxh4e5YfThkSFY_Wxp6A-ao",
]

# =========================================================
# DEFAULTS
# =========================================================
MODEL_NAME = "gemini-2.5-flash"       # kerak bo'lsa "models/gemini-2.5-flash" qilib ko'ring
SESSION_NAME = "shaxsiy_sessiya"        # Koyeb uchun alohida session nomi ishlating
MEMORY_LEN = 10
CONTACTS_ONLY = True                 # True bo'lsa faqat kontaktlar
DEBUG_ECHO_ONLY = False               # True bo'lsa AI o'rniga test reply

# Rate-limit: har Gemini chaqiruvi orasida minimal pauza (sekund)
MIN_DELAY = 1.0

# =========================================================
# BLOCKLIST (javob bermaslik)
# =========================================================
BLOCK_USER_IDS = {8114716789}  # <-- siz bergan user_id shu yerda

# (ixtiyoriy) username bo'yicha ham bo'lsa:
BLOCK_USERNAMES = set()        # masalan: {"baduser1", "baduser2"}

# =========================================================
# GLOBALS
# =========================================================
memory: dict[int, deque] = {}
tg_client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

_gemini_lock = asyncio.Lock()
_last_call_ts = 0.0


# =========================================================
# HEALTH SERVER (Koyeb uchun foydali)
# =========================================================
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


# =========================================================
# HELPERS
# =========================================================
def ensure_memory(user_id: int):
    if user_id not in memory:
        memory[user_id] = deque(maxlen=MEMORY_LEN)


def safe_name(sender) -> str:
    return (getattr(sender, "first_name", None) or "Do'stim").strip()


def get_username(sender) -> str:
    return (getattr(sender, "username", None) or "").strip().lower()


def is_blocked(sender) -> bool:
    sid = getattr(sender, "id", None)
    if isinstance(sid, int) and sid in BLOCK_USER_IDS:
        return True
    uname = get_username(sender)
    if uname and uname in BLOCK_USERNAMES:
        return True
    return False


def detect_mime(event, file_path: str, media_type: str) -> str:
    if getattr(event, "document", None) and getattr(event.document, "mime_type", None):
        return event.document.mime_type
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        return mime
    return "image/jpeg" if media_type == "image" else "audio/ogg"


async def _global_delay():
    global _last_call_ts
    now = time.time()
    wait = (_last_call_ts + MIN_DELAY) - now
    if wait > 0:
        await asyncio.sleep(wait)


# =========================================================
# GEMINI: multi-key fallback + 429 rotation + global delay
# =========================================================
async def gemini_generate_text(contents):
    global _last_call_ts

    async with _gemini_lock:
        await _global_delay()

        last_err = None
        for key in GEMINI_KEYS:
            try:
                client = genai.Client(api_key=key)

                def _run():
                    resp = client.models.generate_content(model=MODEL_NAME, contents=contents)
                    return (getattr(resp, "text", None) or "").strip()

                text = await asyncio.to_thread(_run)
                _last_call_ts = time.time()
                return text

            except Exception as e:
                last_err = e
                msg = str(e)
                if ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate" in msg.lower()):
                    continue
                raise

        _last_call_ts = time.time()
        raise RuntimeError(f"Barcha Gemini keylar limitga yetdi. Oxirgi xato: {last_err}")


async def upload_file_to_gemini(file_path: str, mime_type: str):
    global _last_call_ts

    async with _gemini_lock:
        await _global_delay()

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
                _last_call_ts = time.time()
                return f

            except Exception as e:
                last_err = e
                msg = str(e)
                if ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate" in msg.lower()):
                    continue
                raise

        _last_call_ts = time.time()
        raise RuntimeError(f"Upload hamma keylarda limit. Oxirgi xato: {last_err}")


# =========================================================
# PROMPTS
# =========================================================
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
        f"Foydalanuvchi rasm yubordi. Rasmni tahlil qil va o'zbek tilida qisqa javob ber."
    )


def build_audio_prompt(user_name: str, history_context: str) -> str:
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. "
        f"Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"Suhbat tarixi:\n{history_context}\n\n"
        f"Foydalanuvchi voice/audio yubordi. Audio mazmunini tushunib qisqa javob ber."
    )


async def handle_media(event, user_id: int, user_name: str, media_type: str) -> str:
    tmp_path = None
    try:
        suffix = ".jpg" if media_type == "image" else ".ogg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        downloaded = await event.download_media(file=tmp_path)
        if not downloaded:
            return "Media faylni yuklab bo'lmadi."

        file_path = downloaded if isinstance(downloaded, str) else tmp_path
        mime_type = detect_mime(event, file_path, media_type)
        logging.info("MEDIA | path=%s | mime=%s | type=%s", file_path, mime_type, media_type)

        if DEBUG_ECHO_ONLY:
            return f"Test ({media_type}) ✅"

        history = "\n".join(memory.get(user_id, []))
        uploaded = await upload_file_to_gemini(file_path, mime_type)

        prompt = build_image_prompt(user_name, history) if media_type == "image" else build_audio_prompt(user_name, history)
        return await gemini_generate_text([prompt, uploaded]) or "Tushunmadim."

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================================================
# TELEGRAM HANDLER
# =========================================================
@tg_client.on(events.NewMessage(incoming=True))
async def pro_handler(event):
    if not event.is_private or event.out:
        return

    sender = await event.get_sender()

    logging.info(
        "SENDER | id=%s username=@%s contact=%s",
        getattr(sender, "id", None),
        getattr(sender, "username", None),
        getattr(sender, "contact", None),
    )

    # 1) BLOCKLIST
    if is_blocked(sender):
        logging.info("BLOCKED | ignore this user")
        return

    # 2) CONTACTS_ONLY
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

            async with tg_client.action(user_id, "typing"):
                reply = await gemini_generate_text(build_text_prompt(user_name, history))
                reply = reply or "Kechirasiz, javob chiqmadi."
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
                reply = await handle_media(event, user_id, user_name, "image")
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
            memory[user_id].append("Foydalanuvchi: [Voice/Audio yubordi]")
            async with tg_client.action(user_id, "typing"):
                reply = await handle_media(event, user_id, user_name, "audio")
                memory[user_id].append(f"Sen: {reply}")
                await event.reply(reply)
            return

    except Exception:
        logging.exception("HANDLER ERROR")
        try:
            await event.reply("Xatolik bo'ldi, keyinroq yana urinib ko'ring.")
        except Exception:
            pass


# =========================================================
# START
# =========================================================
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    health = await start_health_server()

    try:
        await tg_client.start()
        me = await tg_client.get_me()
        logging.info("✅ BOT FAOL: @%s", getattr(me, "username", None) or me.id)
        logging.info("Keys: %d | Delay: %.2fs | BlockIDs: %s", len(GEMINI_KEYS), MIN_DELAY, list(BLOCK_USER_IDS))

        await tg_client.run_until_disconnected()
    finally:
        health.close()
        await health.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
