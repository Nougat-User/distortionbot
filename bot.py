import os
import uuid
import shutil
import logging
import asyncio
import subprocess
from typing import Optional
import numpy as np
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Предполагается, что seam_carving доступен
import seam_carving

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
load_dotenv()
TOKEN = os.getenv("TOKEN")
DISTORT_PERCENT = float(os.getenv("DISTORT_PERCENT", 36))
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 МБ
MAX_CONCURRENT_TASKS = 5  # Ограничение одновременных задач
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# Создание директорий
os.makedirs("raw", exist_ok=True)
os.makedirs("result", exist_ok=True)

def check_ffmpeg() -> None:
    """Проверяет наличие ffmpeg на сервере."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("ffmpeg успешно обнаружен")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error("ffmpeg не найден: %s", e)
        raise EnvironmentError("ffmpeg не установлен. Установите ffmpeg для обработки GIF.")

async def process_image(src_path: str, dest_path: str, distort_percent: float) -> bool:
    """Обрабатывает изображение с помощью seam carving.

    Args:
        src_path: Путь к исходному изображению.
        dest_path: Путь для сохранения обработанного изображения.
        distort_percent: Процент уменьшения размеров.

    Returns:
        bool: True, если обработка успешна, иначе False.
    """
    try:
        with Image.open(src_path) as img:
            src = np.array(img.convert("RGB"))
        src_h, src_w, _ = src.shape
        if src_h > 5000 or src_w > 5000:
            logger.warning("Изображение слишком большое: %dx%d", src_w, src_h)
            return False

        dst = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: seam_carving.resize(
                    src,
                    (int(src_w * (1 - distort_percent / 100)), int(src_h * (1 - distort_percent / 100))),
                    energy_mode='backward',
                    order='width-first',
                    keep_mask=None
                )
            ),
            timeout=30.0
        )
        Image.fromarray(dst).save(dest_path)
        return True
    except UnidentifiedImageError as e:
        logger.error("Невозможно открыть изображение %s: %s", src_path, e)
        return False
    except asyncio.TimeoutError:
        logger.error("Таймаут при обработке изображения %s", src_path)
        return False
    except Exception as e:
        logger.error("Ошибка обработки изображения %s: %s", src_path, e)
        return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /start."""
    await update.message.reply_text(
        "Отправьте фото, статический стикер или GIF.\n"
        "Используйте /distort <процент> для изменения степени искажения (0-50%).\n"
        "Пример: /distort 20"
    )

async def set_distort_percent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Устанавливает процент искажения через команду /distort."""
    global DISTORT_PERCENT
    try:
        percent = float(context.args[0])
        if 0 <= percent <= 50:
            DISTORT_PERCENT = percent
            await update.message.reply_text(f"Установлен процент искажения: {percent}%")
            logger.info("Установлен новый процент искажения: %s%%", percent)
        else:
            await update.message.reply_text("Процент должен быть от 0 до 50.")
    except (IndexError, ValueError):
        await update.message.reply_text("Укажите корректный процент, например: /distort 20")

async def distort(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает фото, применяя seam carving."""
    async with SEMAPHORE:
        await update.message.reply_text("Обрабатываю фото…")
        photo_file = await update.message.photo[-1].get_file()
        if photo_file.file_size > MAX_FILE_SIZE:
            await update.message.reply_text("Файл слишком большой (максимум 20 МБ).")
            return

        filename = f"{uuid.uuid4().hex}.jpg"
        raw_path = f"./raw/{filename}"
        result_path = f"./result/{filename}"

        try:
            await photo_file.download_to_drive(raw_path)
            logger.info("Получено фото от пользователя %s, размер: %s", update.effective_user.id, photo_file.file_size)

            if await process_image(raw_path, result_path, DISTORT_PERCENT):
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(result_path, 'rb'))
            else:
                await update.message.reply_text("Ошибка при обработке фото :(")
        except Exception as e:
            logger.error("Ошибка при скачивании или отправке фото: %s", e)
            await update.message.reply_text("Ошибка при обработке фото :(")
        finally:
            for path in (raw_path, result_path):
                if os.path.exists(path):
                    os.remove(path)

async def distort_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает статические стикеры, применяя seam carving."""
    async with SEMAPHORE:
        sticker = update.message.sticker
        if sticker.is_animated or sticker.is_video:
            await update.message.reply_text("Анимированные стикеры пока не поддерживаются 😢")
            return

        await update.message.reply_text("Обрабатываю стикер…")
        file = await context.bot.get_file(sticker.file_id)
        if file.file_size > MAX_FILE_SIZE:
            await update.message.reply_text("Файл слишком большой (максимум 20 МБ).")
            return

        filename = f"{uuid.uuid4().hex}.png"
        raw_path = f"./raw/{filename}"
        result_path = f"./result/{filename}"

        try:
            await file.download_to_drive(raw_path)
            logger.info("Получен стикер от пользователя %s, размер: %s", update.effective_user.id, file.file_size)

            if await process_image(raw_path, result_path, DISTORT_PERCENT):
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(result_path, 'rb'))
            else:
                await update.message.reply_text("Ошибка при обработке стикера :(")
        except Exception as e:
            logger.error("Ошибка при скачивании или отправке стикера: %s", e)
            await update.message.reply_text("Ошибка при обработке стикера :(")
        finally:
            for path in (raw_path, result_path):
                if os.path.exists(path):
                    os.remove(path)

async def distort_gif(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает GIF, применяя seam carving к каждому кадру."""
    async with SEMAPHORE:
        file = await context.bot.get_file(update.message.animation.file_id if update.message.animation else update.message.document.file_id)
        if file.file_size > MAX_FILE_SIZE:
            await update.message.reply_text("Файл слишком большой (максимум 20 МБ).")
            return

        await update.message.reply_text("Обрабатываю GIF…")
        uid = uuid.uuid4().hex
        raw_gif = f"./raw/{uid}.gif"
        frames_dir = f"./raw/{uid}_frames"
        result_gif = f"./result/{uid}_distorted.gif"
        os.makedirs(frames_dir, exist_ok=True)

        try:
            await file.download_to_drive(raw_gif)
            logger.info("Получен GIF от пользователя %s, размер: %s", update.effective_user.id, file.file_size)

            # Извлечение кадров
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["ffmpeg", "-i", raw_gif, f"{frames_dir}/frame_%04d.png"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            )

            # Обработка кадров
            frame_paths = [os.path.join(frames_dir, name) for name in sorted(os.listdir(frames_dir))]
            total_frames = len(frame_paths)
            halfway_point = total_frames // 2

            for i, frame_path in enumerate(frame_paths, 1):
                if not await process_image(frame_path, frame_path, DISTORT_PERCENT):
                    raise RuntimeError(f"Ошибка обработки кадра {frame_path}")
                if i == halfway_point:
                    await update.message.reply_text("Обработано 50%")

            # Сборка GIF
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "ffmpeg", "-y", "-framerate", "10", "-i", f"{frames_dir}/frame_%04d.png",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-loop", "0", result_gif
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            )

            await context.bot.send_animation(chat_id=update.effective_chat.id, animation=open(result_gif, "rb"))
        except subprocess.CalledProcessError as e:
            logger.error("Ошибка ffmpeg при обработке GIF: %s", e)
            await update.message.reply_text("Ошибка при обработке GIF :(")
        except Exception as e:
            logger.error("Ошибка обработки GIF: %s", e)
            await update.message.reply_text("Ошибка при обработке GIF :(")
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)
            for path in (raw_gif, result_gif):
                if os.path.exists(path):
                    os.remove(path)

if __name__ == "__main__":
    check_ffmpeg()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("distort", set_distort_percent))
    app.add_handler(MessageHandler(filters.PHOTO, distort))
    app.add_handler(MessageHandler(filters.Sticker.ALL, distort_sticker))
    app.add_handler(MessageHandler(filters.ANIMATION, distort_gif))
    app.run_polling()
