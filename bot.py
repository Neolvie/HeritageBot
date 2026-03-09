"""
HeritageBot — genealogy photo annotation Telegram bot.

Flow:
  1. User sends a photo (or image as document).
  2. User sends a voice message describing who/what is in the photo.
  3. Bot transcribes voice (Whisper), analyses photo (OpenRouter vision model).
  4. Bot sends:
       • Photo metadata (first message)
       • Cropped region + structured text for each person / object / animal / location
       • JSON file with full structured data (last message)
"""

import json
import logging
import os
import tempfile
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ai_client import analyze_photo, transcribe_audio
from formatters import (
    esc,
    format_animal,
    format_location,
    format_object,
    format_person,
    format_photo_metadata,
    truncate_caption,
)
from image_utils import crop_region, is_valid_bbox, prepare_image

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# user_id → {"photo_path": str}
_user_state: dict[int, dict] = {}

WELCOME = (
    "👋 Добро пожаловать в <b>HeritageBot</b>!\n\n"
    "Я помогу превратить семейные фотографии в структурированный генеалогический архив.\n\n"
    "<b>Как это работает:</b>\n"
    "1️⃣ Отправьте фотографию\n"
    "2️⃣ Запишите голосовое сообщение — расскажите, <b>кто изображён</b>, "
    "какие предметы видны, когда и где сделан снимок\n"
    "3️⃣ Получите структурированное описание каждого человека/предмета "
    "и готовый JSON-файл\n\n"
    "Команды: /reset — сбросить сессию"
)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode="HTML")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    _cleanup_state(uid)
    await update.message.reply_text("🔄 Сессия сброшена. Отправьте новую фотографию.")


# ─────────────────────────────────────────────────────────────────────────────
# PHOTO HANDLER
# ─────────────────────────────────────────────────────────────────────────────


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    _cleanup_state(uid)

    # Telegram sends photos in multiple sizes; pick the largest
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    await file.download_to_drive(tmp.name)
    prepare_image(tmp.name)

    _user_state[uid] = {"photo_path": tmp.name}

    await update.message.reply_text(
        "✅ Фото сохранено!\n\n"
        "Теперь отправьте <b>голосовое сообщение</b> — расскажите:\n"
        "• Кто изображён (ФИО, кем приходится)\n"
        "• Какие предметы / животные видны и что они значат\n"
        "• Когда и где сделан снимок\n"
        "• Любые другие важные детали",
        parse_mode="HTML",
    )


async def handle_document_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Accept images sent as documents (files)."""
    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("⚠️ Пожалуйста, отправьте изображение.")
        return

    uid = update.effective_user.id
    _cleanup_state(uid)

    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    ext = ext_map.get(doc.mime_type, ".jpg")

    file = await context.bot.get_file(doc.file_id)
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.close()
    await file.download_to_drive(tmp.name)
    prepare_image(tmp.name)

    _user_state[uid] = {"photo_path": tmp.name}

    await update.message.reply_text(
        "✅ Изображение сохранено! Теперь отправьте голосовое сообщение с описанием.",
        parse_mode="HTML",
    )


# ─────────────────────────────────────────────────────────────────────────────
# VOICE HANDLER  (main processing pipeline)
# ─────────────────────────────────────────────────────────────────────────────


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id

    if uid not in _user_state or "photo_path" not in _user_state[uid]:
        await update.message.reply_text(
            "⚠️ Сначала отправьте <b>фотографию</b>, затем голосовое сообщение.",
            parse_mode="HTML",
        )
        return

    photo_path = _user_state[uid]["photo_path"]
    if not os.path.exists(photo_path):
        await update.message.reply_text(
            "⚠️ Фото не найдено. Пожалуйста, отправьте его снова."
        )
        _cleanup_state(uid)
        return

    status_msg = await update.message.reply_text("🎙 Распознаю голосовое сообщение…")

    # Download voice
    voice_obj = update.message.voice or update.message.audio
    voice_file = await context.bot.get_file(voice_obj.file_id)
    voice_tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    voice_tmp.close()
    await voice_file.download_to_drive(voice_tmp.name)

    json_tmp_path: str | None = None

    try:
        # ── 1. Transcribe ────────────────────────────────────────────────────
        transcription = await transcribe_audio(voice_tmp.name)
        await status_msg.edit_text(
            f"✅ Расшифровка:\n<i>{esc(transcription)}</i>\n\n🔍 Анализирую фотографию…",
            parse_mode="HTML",
        )

        # ── 2. Analyse ───────────────────────────────────────────────────────
        result = await analyze_photo(photo_path, transcription)
        result["processing_metadata"] = {
            "transcription": transcription,
            "processed_at": datetime.now().isoformat(),
            "model_used": os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite"),
        }

        await status_msg.edit_text("📤 Формирую ответ…")

        chat_id = update.effective_chat.id

        # ── 3. Photo metadata ────────────────────────────────────────────────
        meta_text = format_photo_metadata(result.get("photo_metadata", {}))
        await update.message.reply_text(
            f"📸 <b>Информация о фотографии</b>\n\n{meta_text}",
            parse_mode="HTML",
        )

        # ── 4. Entity cards ──────────────────────────────────────────────────
        entity_groups = [
            ("persons",                "👤", format_person,   lambda e: e.get("full_name_display") or "Человек"),
            ("objects",                "🏺", format_object,   lambda e: e.get("name") or "Предмет"),
            ("animals",                "🐾", format_animal,   lambda e: e.get("name") or e.get("species") or "Животное"),
            ("locations_in_background","📍", format_location, lambda e: e.get("name") or "Место"),
        ]

        for category, icon, formatter, name_fn in entity_groups:
            for entity in result.get(category, []):
                name = name_fn(entity)
                body = formatter(entity)
                header = f"{icon} <b>{esc(name)}</b>\n\n{body}"

                bbox = entity.get("bounding_box")
                cropped = None
                if is_valid_bbox(bbox):
                    cropped = crop_region(photo_path, bbox)

                if cropped:
                    caption, overflow = truncate_caption(header)
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=cropped,
                        caption=caption,
                        parse_mode="HTML",
                    )
                    if overflow:
                        # Send the full text as a follow-up message
                        await update.message.reply_text(header, parse_mode="HTML")
                else:
                    await update.message.reply_text(header, parse_mode="HTML")

        # ── 5. JSON file (last message) ──────────────────────────────────────
        json_content = json.dumps(result, ensure_ascii=False, indent=2)
        json_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        )
        json_tmp.write(json_content)
        json_tmp.close()
        json_tmp_path = json_tmp.name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(json_tmp_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat_id,
                document=f,
                filename=f"heritage_{timestamp}.json",
                caption="📄 <b>Полные структурированные данные (JSON)</b>",
                parse_mode="HTML",
            )

        await status_msg.edit_text("✅ Готово!")

    except Exception as exc:
        logger.exception("Processing error for user %s", uid)
        await status_msg.edit_text(
            f"❌ Ошибка при обработке:\n<code>{esc(str(exc))}</code>\n\n"
            "Попробуйте снова или отправьте /reset",
            parse_mode="HTML",
        )

    finally:
        _safe_unlink(voice_tmp.name)
        _safe_unlink(json_tmp_path)
        _cleanup_state(uid)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _cleanup_state(uid: int) -> None:
    state = _user_state.pop(uid, {})
    _safe_unlink(state.get("photo_path"))


def _safe_unlink(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    logger.info("HeritageBot is running…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
