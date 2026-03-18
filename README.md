# Lecture Capture

`lecture-capture` is a Dockerized browser-only lecture recorder and transcriber. Any device with a modern browser can open the UI, capture microphone audio or browser-tab audio, stream it to a Linux server over WebSocket, watch live text updates, and then wait for a higher-accuracy Parakeet TDT pass to finish.

## What it does

- Serves a single-page FastAPI UI at `http://<tailscale-ip>:8000`.
- Captures audio in the browser with `getUserMedia()` or `getDisplayMedia()`.
- Streams `audio/webm;codecs=opus` chunks to the server over WebSocket every 250ms.
- Writes the raw browser stream to `recording.webm`.
- Decodes the live stream with FFmpeg and feeds PCM into RealtimeSTT for low-latency text updates.
- Converts `recording.webm` to `audio.wav` after stop and runs a single full Parakeet pass on CPU.
- Saves artifacts under `~/lectures/<YYYY-MM-DD_HH-MM_session-name>/`.

## Project layout

```text
lecture-capture/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── main.py
    ├── live_transcribe.py
    ├── transcribe.py
    ├── utils.py
    ├── routers/
    │   ├── __init__.py
    │   ├── session.py
    │   └── sessions.py
    ├── ws/
    │   ├── __init__.py
    │   └── audio.py
    └── static/
        ├── index.html
        └── fonts/
            └── README.md
```

## Requirements

### Host machine

- Linux server with Docker and Docker Compose plugin
- 8 CPU cores / 16 GB RAM is the intended baseline
- Tailscale account and an auth key
- Enough disk space under `~/lectures`

### Browser clients

- Chrome, Edge, or another Chromium browser is the best target
- Microphone mode works over HTTP on a Tailscale IP
- Tab-audio mode on Chromium requires a secure context: HTTPS or `localhost`

That last point matters on Chromebooks. Over plain `http://<tailscale-ip>:8000`, test microphone mode first. For tab audio on a Chromebook, you will usually need one of these:

- Serve the app over HTTPS with your own certificate
- Use Chrome's `chrome://flags/#unsafely-treat-insecure-origin-as-secure` flag and add your Tailscale origin

## Python and uv

This repo uses `uv` exclusively. No `pip`, no manual `venv`, no `conda`.

Local development setup:

```bash
cp .env.example .env
uv sync
```

The project pins CPU wheels for `torch` and `torchaudio` on Linux and Windows so the default install stays CPU-only. If a GPU is present, the app logs a warning and still runs on CPU by design.

## Docker deployment

### 1. Prepare environment

Copy the environment file:

```bash
cp .env.example .env
```

Edit `.env` and set a valid Tailscale auth key:

```bash
TS_AUTHKEY=tskey-auth-xxxxxxxx
```

Create the host lecture directory:

```bash
mkdir -p ~/lectures
```

### 2. Build and start

```bash
docker compose up --build -d
```

### 3. Find the Tailscale IP

After the containers start, read the sidecar logs or query the sidecar directly:

```bash
docker compose logs -f tailscale
docker compose exec tailscale tailscale ip -4
```

### 4. Open the app

From any browser on your tailnet:

```text
http://<tailscale-ip>:8000
```

No client install is needed on the Chromebook or phone beyond a normal browser that can reach the Tailscale IP.

## Local development

Run the FastAPI server without Docker:

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Run the standalone full-transcription helper on an existing WAV:

```bash
uv run src/transcribe.py --file /path/to/audio.wav
```

Run the standalone helper on a saved browser recording:

```bash
uv run src/transcribe.py --recording /path/to/recording.webm
```

## Runtime flow

### Browser

- User enters a session name and chooses `microphone` or `tab audio`
- UI calls `POST /session/start`
- UI opens `/ws/audio?session_id=...`
- Browser starts `MediaRecorder` and sends binary chunks every 250ms
- Live transcript messages arrive over the same WebSocket
- On stop, the browser sends `{ "type": "stop" }`

### Server

- FastAPI writes raw WebM chunks to `recording.webm`
- FFmpeg decodes the stream in real time to raw PCM
- RealtimeSTT consumes PCM with `use_microphone=False`
- When recording ends, FFmpeg converts the saved WebM to `audio.wav`
- Parakeet TDT generates `transcript.txt`
- Status is exposed through `GET /session/status`

## API summary

- `POST /session/start`
- `POST /session/stop`
- `GET /session/status`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/transcript`
- `GET /ws/audio`

## Output layout

Each session directory contains:

```text
~/lectures/2026-03-13_22-40_discrete-math/
├── recording.webm
├── audio.wav
├── transcript.txt
└── session.json
```

`session.json` stores metadata for the past-sessions sidebar and status endpoints.

## Notes

- The first live-transcription request may download local model weights for RealtimeSTT.
- The first full Parakeet request may download NeMo model weights.
- Browser disconnects are treated as an implicit stop. Whatever audio already reached the server is preserved and transcribed.
- The server intentionally runs CPU-only even when CUDA is present.

## Troubleshooting

### Docker build fails on PyAudio or PortAudio

The image installs `portaudio19-dev` because RealtimeSTT pulls in `PyAudio` even though the server does not record from a physical microphone.

### Tab audio does not start on Chromebook

This is usually the secure-context restriction for `getDisplayMedia`. Test microphone mode first, then either move the app to HTTPS or mark the Tailscale origin as secure in Chrome flags.

### Realtime text is slow or choppy

Make sure the browser is actually sending `audio/webm;codecs=opus`. Chromium browsers are the most reliable target here.

### Full transcription is slow

That is expected on CPU with Parakeet. Status polling and the live transcript are there so the browser does not look frozen.
