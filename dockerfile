FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .


EXPOSE 3000 8000

CMD ["reflex", "run", "--env", "prod"]