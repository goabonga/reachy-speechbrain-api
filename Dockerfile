FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Dépendances système nécessaires à SpeechBrain / torchaudio
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installer les deps (prod only)
RUN uv sync --no-dev

# Copier le code
COPY app.py storage.py ./

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
