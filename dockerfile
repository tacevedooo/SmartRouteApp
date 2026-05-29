FROM python:3.13-slim

# Solo unzip y curl — Reflex instala Bun solo, no necesita nodejs/npm del sistema
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Instala dependencias con índices CPU-only
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    tensorflow-cpu && \
    pip install --no-cache-dir \
    -r requirements.txt

COPY . .

EXPOSE 3000 8000

CMD ["reflex", "run", "--env", "prod"]