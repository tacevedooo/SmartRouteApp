FROM python:3.11-slim

# 1. Instalar dependencias del sistema necesarias para Reflex y Node.js
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copiar requerimientos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar TODO el proyecto (incluye datasets, modelos y assets)
COPY . .

# Eliminar carpetas locales de entorno virtual o caché Next.js si se colaron al copiar
RUN rm -rf venv .venv .web .states

# 4. Inicializar Reflex para que descargue y configure el frontend de producción
RUN reflex init

# Hugging Face expone el puerto 7860 por defecto
EXPOSE 7860

# 5. Comando de arranque adaptado a la arquitectura de Reflex
CMD ["reflex", "run", "--env", "prod", "--frontend-port", "7860", "--backend-port", "8000"]