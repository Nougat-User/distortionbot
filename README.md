# 📸 Shitted AI-refactored(for new python3) distortion bot for Telegram

This Telegram bot automatically distorts images, stickers, and animated GIFs using the **seam carving** algorithm.

Send the bot:

* A **photo** — it will return a distorted version.
* A **static sticker** — it will distort and return it.
* An **animated GIF** (from the GIF panel, not as document) — it will process and return the distorted animation.

---

## 🚀 Features

* Works with **photos**, **stickers**, and **GIF animations**.
* Applies **content-aware resizing** via seam carving.
* GIFs are optimized: resized and sped up ×3 to reduce processing time.
* Runs in Docker, supports `.env` configuration.

---

## 🐳 Run with Docker

```bash
docker compose up --build
```

Make sure your `.env` file contains your bot token:

```env
TOKEN=your_bot_token_here
```

---

## 🧩 Requirements (for manual run)

* Python 3.10+
* ffmpeg
* pip packages in `requirements.txt`

```bash
pip install -r requirements.txt
```
