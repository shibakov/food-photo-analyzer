FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway сам проставляет PORT, но можно задать дефолт
ENV PORT=8080

# ВАЖНО: НИКАКОГО ENTRYPOINT, порт берём из $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
