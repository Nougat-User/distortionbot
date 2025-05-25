import os
import uuid
import shutil
import logging
import subprocess
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import seam_carving

load_dotenv()
TOKEN = os.getenv("TOKEN")
DISTORT_PERCENT = 36

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

os.makedirs("raw", exist_ok=True)
os.makedirs("result", exist_ok=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Просто отправь мне фотку, стикер или гифку.\n"
        "Если я не отвечаю — возможно, бот занят или выключен."
    )

async def distort(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("New photo received")
    photo_file = await update.message.photo[-1].get_file()
    filename = f"{photo_file.file_unique_id}.jpg"
    raw_path = f"./raw/{filename}"
    result_path = f"./result/{filename}"
    await photo_file.download_to_drive(raw_path)

    try:
        src = np.array(Image.open(raw_path))
        src_h, src_w, _ = src.shape

        dst = seam_carving.resize(
            src,
            (int(src_w - (src_w * DISTORT_PERCENT / 100)), int(src_h - (src_h * DISTORT_PERCENT / 100))),
            energy_mode='backward',
            order='width-first',
            keep_mask=None
        )

        Image.fromarray(dst).save(result_path)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(result_path, 'rb'))
    except Exception as e:
        logging.error(f"Ошибка обработки фото: {e}")
        await update.message.reply_text("Ошибка при обработке фото :(")

async def distort_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sticker = update.message.sticker

    if not sticker.is_animated and not sticker.is_video:
        await update.message.reply_text("Обрабатываю стикер…")
        file = await context.bot.get_file(sticker.file_id)
        filename = f"{file.file_unique_id}.png"
        raw_path = f"./raw/{filename}"
        result_path = f"./result/{filename}"
        await file.download_to_drive(raw_path)

        try:
            src = np.array(Image.open(raw_path).convert("RGB"))
            src_h, src_w, _ = src.shape

            dst = seam_carving.resize(
                src,
                (int(src_w - (src_w * DISTORT_PERCENT / 100)), int(src_h - (src_h * DISTORT_PERCENT / 100))),
                energy_mode='backward',
                order='width-first',
                keep_mask=None
            )

            Image.fromarray(dst).save(result_path)
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(result_path, 'rb'))
        except Exception as e:
            logging.error(f"Ошибка обработки стикера: {e}")
            await update.message.reply_text("Ошибка при обработке стикера :(")
    else:
        await update.message.reply_text("Анимированные стикеры пока не поддерживаются 😢")

async def distort_gif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await context.bot.get_file(update.message.animation.file_id if update.message.animation else update.message.document.file_id)

    await update.message.reply_text("Обрабатываю GIF…")

    uid = uuid.uuid4().hex
    raw_gif = f"./raw/{uid}.gif"
    frames_dir = f"./raw/{uid}_frames"
    result_gif = f"./result/{uid}_distorted.gif"
    os.makedirs(frames_dir, exist_ok=True)

    await file.download_to_drive(raw_gif)

    try:
        subprocess.run(["ffmpeg", "-i", raw_gif, f"{frames_dir}/frame_%04d.png"], check=True)

        for name in sorted(os.listdir(frames_dir)):
            path = os.path.join(frames_dir, name)
            img = Image.open(path).convert("RGB")
            img.thumbnail((img.width // 2, img.height // 2), Image.LANCZOS)
            arr = np.array(img)
            h, w, _ = arr.shape
            dst = seam_carving.resize(
                arr,
                (int(w - (w * DISTORT_PERCENT / 100)), int(h - (h * DISTORT_PERCENT / 100))),
                energy_mode='backward',
                order='width-first',
                keep_mask=None
            )
            Image.fromarray(dst).save(path)

        subprocess.run([
            "ffmpeg", "-y", "-framerate", "10", "-i", f"{frames_dir}/frame_%04d.png",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-loop", "0", result_gif
        ], check=True)

        await context.bot.send_animation(chat_id=update.effective_chat.id, animation=open(result_gif, "rb"))
    except Exception as e:
        logging.error(f"Ошибка обработки gif: {e}")
        await update.message.reply_text("Ошибка при обработке gif :(")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        for f in (raw_gif, result_gif):
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, distort))
    app.add_handler(MessageHandler(filters.Sticker.ALL, distort_sticker))
    app.add_handler(MessageHandler(filters.ANIMATION, distort_gif))
    app.run_polling()
