FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libjpeg-dev zlib1g-dev ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
