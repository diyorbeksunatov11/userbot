import asyncio
import logging
import mimetypes
import os
import tempfile
from collections import deque

from telethon import TelegramClient, events
from google import genai

# =========================================================
# KOYEB / ENV SOZLAMALAR
# =========================================================
# Koyeb -> Environment variables:
# API_ID=...
# API_HASH=...
# GEMINI_API_KEY=...
# MODEL_NAME=models/gemini-2.5-flash      (optional)
# SESSION_NAME=shaxsiy_sessiya_pro        (optional)
# CONTACTS_ONLY=0                         (0 = hamma private chat, 1 = faqat kontaktlar)
# MEMORY_LEN=10                           (optional)
# DEBUG_ECHO_ONLY=0                       (1 = AI o'rniga test reply qaytaradi)

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")

SESSION_NAME = os.getenv("SESSION_NAME", "shaxsiy_sessiya_pro")
CONTACTS_ONLY = os.getenv("CONTACTS_ONLY", "0") == "1"   # default: 0 (muhim)
MEMORY_LEN = int(os.getenv("MEMORY_LEN", "10"))
DEBUG_ECHO_ONLY = os.getenv("DEBUG_ECHO_ONLY", "0") == "1"

# =========================================================
# GLOBAL HOLAT
# =========================================================
memory: dict[int, deque] = {}

if not GEMINI_API_KEY:
    ai_client = None
else:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)

tg_client = TelegramClient(SESSION_NAME, API_ID, API_HASH)


# =========================================================
# HEALTH SERVER (Koyeb Web Service uchun)
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
# YORDAMCHI FUNKSIYALAR
# =========================================================
def ensure_memory(user_id: int):
    if user_id not in memory:
        memory[user_id] = deque(maxlen=MEMORY_LEN)


def detect_mime(event, file_path: str, media_type: str) -> str:
    # 1) Telegram'dan kelgan mime_type
    if getattr(event, "document", None) and getattr(event.document, "mime_type", None):
        return event.document.mime_type

    # 2) extension orqali
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        return mime

    # 3) fallback
    if media_type == "image":
        return "image/jpeg"
    if media_type == "audio":
        return "audio/ogg"

    return "application/octet-stream"


async def gemini_generate_text(contents):
    """
    Gemini sync API chaqiruvini asyncio loopni bloklamaslik uchun thread ichida ishlatamiz.
    """
    if ai_client is None:
        raise RuntimeError("GEMINI_API_KEY topilmadi. ENV ga GEMINI_API_KEY qo'ying.")

    def _run():
        resp = ai_client.models.generate_content(
            model=MODEL_NAME,
            contents=contents
        )
        return (getattr(resp, "text", None) or "").strip()

    return await asyncio.to_thread(_run)


def build_text_prompt(user_name: str, history_context: str) -> str:
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. "
        f"Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"Suhbat tarixi:\n{history_context}\n\n"
        f"Vazifa: Faqat o'zbek tilida, samimiy, qisqa va mantiqiy javob ber. "
        f"Javob tabiiy bo'lsin."
    )


def build_image_prompt(user_name: str, history_context: str) -> str:
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. "
        f"Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"Suhbat tarixi:\n{history_context}\n\n"
        f"Foydalanuvchi rasm yubordi. "
        f"Rasmni tahlil qil va o'zbek tilida qisqa, foydali, tabiiy javob yoz. "
        f"Agar rasmda matn bo'lsa, muhim qismini aytib ber."
    )


def build_audio_prompt(user_name: str, history_context: str) -> str:
    return (
        f"Sen mening aqlli yordamchimsan. Isming 'AI-Bot'. "
        f"Hozir {user_name} bilan gaplashyapsan.\n\n"
        f"Suhbat tarixi:\n{history_context}\n\n"
        f"Foydalanuvchi voice/audio yubordi. "
        f"Audio mazmunini tushunib, o'zbek tilida qisqa va mantiqiy javob ber. "
        f"Javob juda uzun bo'lmasin."
    )


async def upload_file_to_gemini(file_path: str, mime_type: str):
    """
    google-genai SDK versiyasiga qarab upload signature farq qilishi mumkin.
    Shu sabab bir nechta variant sinab ko'riladi.
    """
    if ai_client is None:
        raise RuntimeError("GEMINI_API_KEY topilmadi.")

    def _run():
        # Variant A
        try:
            return ai_client.files.upload(file=file_path, config={"mime_type": mime_type})
        except TypeError:
            pass

        # Variant B
        try:
            return ai_client.files.upload(file=file_path, mime_type=mime_type)
        except TypeError:
            pass

        # Variant C (oxirgi urinish)
        return ai_client.files.upload(file=file_path)

    return await asyncio.to_thread(_run)


async def handle_media_with_gemini(event, user_id: int, user_name: str, media_type: str) -> str:
    """
    media_type = 'image' | 'audio'
    """
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

        if mime_type == "application/octet-stream":
            mime_type = "image/jpeg" if media_type == "image" else "audio/ogg"

        logging.info("MEDIA | path=%s | mime=%s | type=%s", file_path, mime_type, media_type)

        # DEBUG rejim: AI'ga bormasdan test javob
        if DEBUG_ECHO_ONLY:
            return f"Test ({media_type}) qabul qilindi ✅"

        history_context = "\n".join(memory.get(user_id, []))
        uploaded_file = await upload_file_to_gemini(file_path, mime_type)

        prompt = build_image_prompt(user_name, history_context) if media_type == "image" else build_audio_prompt(user_name, history_context)

        reply_text = await gemini_generate_text([prompt, uploaded_file])
        return reply_text or "Tushunmadim, yana bir marta yuborib ko'ring."

    except Exception as e:
        logging.exception("Media processing error")
        return f"Media ishlashda xatolik bo'ldi: {e}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def safe_name(sender):
    try:
        return (getattr(sender, "first_name", None) or "Do'stim").strip()
    except Exception:
        return "Do'stim"


# =========================================================
# TELEGRAM HANDLER
# =========================================================
@tg_client.on(events.NewMessage(incoming=True))
async def pro_handler(event):
    # DEBUG: xabar keldimi yo'qmi
    logging.info(
        "NEW MSG | private=%s out=%s text=%s photo=%s doc=%s chat_id=%s",
        event.is_private,
        event.out,
        bool(event.text),
        bool(event.photo),
        bool(event.document),
        event.chat_id,
    )

    if not event.is_private or event.out:
        logging.info("SKIP | not private or outgoing")
        return

    sender = await event.get_sender()
    user_id = event.chat_id
    user_name = safe_name(sender)

    logging.info(
        "SENDER | id=%s contact=%s name=%s",
        getattr(sender, "id", None),
        getattr(sender, "contact", None),
        user_name,
    )

    # Faqat kontaktlar rejimi (default endi OFF)
    if CONTACTS_ONLY and not getattr(sender, "contact", False):
        logging.info("SKIP | CONTACTS_ONLY=1 and sender.contact=False")
        return

    ensure_memory(user_id)

    try:
        # =====================================================
        # 1) TEXT
        # =====================================================
        if event.text:
            logging.info("TEXT | %s -> %s", user_name, event.text)

            # DEBUG rejim: AI ishlatmasdan test
            if DEBUG_ECHO_ONLY:
                await event.reply("Test reply ishladi ✅")
                return

            memory[user_id].append(f"Foydalanuvchi: {event.text}")
            history_context = "\n".join(memory[user_id])
            prompt = build_text_prompt(user_name, history_context)

            async with tg_client.action(user_id, "typing"):
                reply_text = await gemini_generate_text(prompt)
                if not reply_text:
                    reply_text = "Kechirasiz, javob tayyor bo'lmadi."

                memory[user_id].append(f"Sen: {reply_text}")

                delay = min(len(reply_text) * 0.04, 4)
                await asyncio.sleep(delay)

                await event.reply(reply_text)
                logging.info("AI(TEXT) | %s", reply_text)
            return

        # =====================================================
        # 2) IMAGE
        # =====================================================
        is_image = bool(event.photo)
        if not is_image and event.document and getattr(event.document, "mime_type", ""):
            is_image = event.document.mime_type.startswith("image/")

        if is_image:
            logging.info("IMAGE | from=%s", user_name)
            memory[user_id].append("Foydalanuvchi: [Rasm yubordi]")

            async with tg_client.action(user_id, "typing"):
                reply_text = await handle_media_with_gemini(event, user_id, user_name, "image")
                memory[user_id].append(f"Sen: {reply_text}")

                delay = min(len(reply_text) * 0.03, 4)
                await asyncio.sleep(delay)

                await event.reply(reply_text)
                logging.info("AI(IMAGE) | %s", reply_text)
            return

        # =====================================================
        # 3) VOICE / AUDIO
        # =====================================================
        # Telegram voice'ni aniqlashda event.voice/event.audio har doim bo'lmasligi mumkin.
        # Shuning uchun document mime_type + document attributesni ham tekshiramiz.
        is_audio = False

        if event.document and getattr(event.document, "mime_type", ""):
            mime = event.document.mime_type
            if mime.startswith("audio/"):
                is_audio = True
            # Ba'zi voice notelar "application/ogg" bo'lib keladi
            if mime in ("application/ogg", "application/octet-stream"):
                # document attributes ichidan voice ni topishga urinamiz
                for attr in getattr(event.document, "attributes", []) or []:
                    attr_name = attr.__class__.__name__.lower()
                    if "audio" in attr_name:
                        is_audio = True
                        break

        # Ba'zi Telethon versiyalarida event.voice/event.audio bo'lishi mumkin
        if not is_audio and (getattr(event, "voice", None) or getattr(event, "audio", None)):
            is_audio = True

        if is_audio:
            logging.info("AUDIO | from=%s", user_name)
            memory[user_id].append("Foydalanuvchi: [Voice/Audio yubordi]")

            async with tg_client.action(user_id, "typing"):
                reply_text = await handle_media_with_gemini(event, user_id, user_name, "audio")
                memory[user_id].append(f"Sen: {reply_text}")

                delay = min(len(reply_text) * 0.03, 5)
                await asyncio.sleep(delay)

                await event.reply(reply_text)
                logging.info("AI(AUDIO) | %s", reply_text)
            return

        # =====================================================
        # 4) BOSHQA MEDIA
        # =====================================================
        logging.info("SKIP | unsupported media type")
        # test uchun xohlasangiz bu line'ni oching:
        # await event.reply("Bu turdagi fayl hozircha qo'llab-quvvatlanmaydi.")
        return

    except Exception as e:
        logging.exception("HANDLER ERROR")
        if "429" in str(e):
            logging.warning("Rate limit 429. 30s kutamiz...")
            await asyncio.sleep(30)
        else:
            # Foydalanuvchiga sokin xabar
            try:
                await event.reply("Xatolik bo'ldi. Keyinroq yana urinib ko'ring.")
            except Exception:
                pass


# =========================================================
# START
# =========================================================
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Minimal config check
    if not API_ID or not API_HASH:
        raise RuntimeError("ENV xato: API_ID va API_HASH shart.")

    # Health server (Koyeb Web Service uchun)
    health_server = await start_health_server()

    # Telegram client start
    await tg_client.start()
    me = await tg_client.get_me()

    print("--- --- --- --- --- --- ---")
    print(f"✅ BOT FAOL: @{getattr(me, 'username', None) or me.id}")
    print(f"🧠 XOTIRA: {MEMORY_LEN}")
    print(f"🛡 FILTR: {'Faqat kontaktlar' if CONTACTS_ONLY else 'Hamma private chat'}")
    print(f"🧪 DEBUG_ECHO_ONLY: {DEBUG_ECHO_ONLY}")
    print(f"🖼 IMAGE: Yoqilgan")
    print(f"🎤 VOICE: Yoqilgan")
    print(f"🌐 Health: 0.0.0.0:{os.environ.get('PORT', '8000')}")
    print("--- --- --- --- --- --- ---")

    try:
        await tg_client.run_until_disconnected()
    finally:
        health_server.close()
        await health_server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot o'chirildi.")
