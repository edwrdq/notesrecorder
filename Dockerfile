FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# FFmpeg powers both the live decode path and the final WebM -> WAV conversion.
# PortAudio headers are installed because RealtimeSTT depends on PyAudio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN mkdir -p /lectures
RUN uv sync --frozen --no-dev --no-cache

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
