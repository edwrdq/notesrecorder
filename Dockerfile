FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# FFmpeg powers both the live decode path and the final WebM -> WAV conversion.
# The container only handles browser-streamed audio, so it does not need
# RealtimeSTT's optional PyAudio microphone stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN mkdir -p /lectures
RUN uv sync --frozen --no-dev --no-cache --no-install-package pyaudio

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
