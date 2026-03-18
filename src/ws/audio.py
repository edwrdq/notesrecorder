from __future__ import annotations

import asyncio
import json
import logging
import queue
import subprocess
import threading
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

try:
    from ..live_transcribe import LiveTranscriptEvent, LiveTranscriber
    from ..transcribe import transcribe_recording_file
    from ..utils import (
        SessionRecord,
        SessionState,
        ensure_command,
        ffmpeg_duration_seconds,
        ffmpeg_install_hint,
        get_session_manager,
    )
except ImportError:
    from live_transcribe import LiveTranscriptEvent, LiveTranscriber
    from transcribe import transcribe_recording_file
    from utils import SessionRecord, SessionState, ensure_command, ffmpeg_duration_seconds, ffmpeg_install_hint, get_session_manager


router = APIRouter()
logger = logging.getLogger("lecture_capture.ws.audio")


class AudioSessionRuntime:
    """Own the live WebSocket recording pipeline for one session."""

    def __init__(self, session: SessionRecord) -> None:
        self.session = session
        self.manager = get_session_manager()
        self.outgoing_messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.completed = threading.Event()
        self.stop_event = threading.Event()
        self._finalize_lock = threading.Lock()
        self._finalized = False

        self.recording_file = session.recording_file
        self.audio_file = session.audio_file
        self.transcript_file = session.transcript_file
        self.recording_handle = self.recording_file.open("ab")

        ensure_command("ffmpeg", ffmpeg_install_hint())
        # FFmpeg decodes the live WebM stream from stdin into raw 16 kHz PCM so
        # RealtimeSTT can work on a queue of plain audio bytes.
        self.decoder = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "+discardcorrupt",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "s16le",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self.live_transcriber = LiveTranscriber(on_transcript=self._queue_live_transcript, logger=logger)
        self.decoder_thread = threading.Thread(target=self._decoder_stdout_worker, daemon=True)
        self.decoder_thread.start()
        self.decoder_stderr_thread = threading.Thread(target=self._decoder_stderr_worker, daemon=True)
        self.decoder_stderr_thread.start()
        self.transcription_thread: threading.Thread | None = None

        self.manager.register_runtime(session.session_id, self)
        self.manager.set_state(session.session_id, SessionState.RECORDING, progress=0)

    def push_chunk(self, chunk: bytes) -> None:
        """Persist raw browser bytes and feed them into the FFmpeg decoder."""

        if self._finalized:
            return

        self.recording_handle.write(chunk)
        self.recording_handle.flush()

        if self.decoder.stdin is not None:
            self.decoder.stdin.write(chunk)
            self.decoder.stdin.flush()

    def request_stop(self) -> None:
        """Signal the websocket loop that stop has been requested."""

        self.stop_event.set()

    def finalize(self, reason: str) -> None:
        """Finalize the audio pipeline once, then launch full transcription."""

        with self._finalize_lock:
            if self._finalized:
                return
            self._finalized = True

        logger.info("Finalizing session %s due to %s", self.session.session_id, reason)
        self.stop_event.set()
        self.manager.clear_stop_request(self.session.session_id)

        try:
            self.recording_handle.flush()
            self.recording_handle.close()
        except Exception:
            pass

        if self.decoder.stdin is not None:
            try:
                self.decoder.stdin.close()
            except Exception:
                pass

        self.decoder_thread.join(timeout=5)

        try:
            self.decoder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.decoder.kill()

        self.live_transcriber.stop()
        self.outgoing_messages.put({"type": "transcribing"})
        self.manager.set_state(self.session.session_id, SessionState.TRANSCRIBING, progress=10)

        # The full Parakeet pass runs in the background so the WebSocket sender
        # can keep delivering status updates while CPU-bound work is happening.
        self.transcription_thread = threading.Thread(target=self._run_full_transcription, daemon=True)
        self.transcription_thread.start()

    def wait_until_complete(self, timeout: float | None = None) -> bool:
        """Allow shutdown hooks and the explicit-stop path to wait for completion."""

        return self.completed.wait(timeout=timeout)

    def next_message(self, timeout: float = 0.25) -> dict[str, Any]:
        """Read the next queued outbound message for the websocket sender."""

        return self.outgoing_messages.get(timeout=timeout)

    def _queue_live_transcript(self, event: LiveTranscriptEvent) -> None:
        """Translate realtime events into outbound WebSocket messages."""

        self.outgoing_messages.put(
            {
                "type": "transcript",
                "text": event.text,
                "timestamp": event.timestamp,
            }
        )

    def _decoder_stdout_worker(self) -> None:
        """Feed FFmpeg PCM output into RealtimeSTT."""

        if self.decoder.stdout is None:
            return

        while True:
            chunk = self.decoder.stdout.read(4096)
            if not chunk:
                return
            self.live_transcriber.feed_pcm(chunk)

    def _decoder_stderr_worker(self) -> None:
        """Log decoder errors without blocking the FFmpeg subprocess."""

        if self.decoder.stderr is None:
            return

        for line in self.decoder.stderr:
            cleaned = line.decode("utf-8", errors="ignore").strip()
            if cleaned:
                logger.warning("Decoder stderr for %s: %s", self.session.session_id, cleaned)

    def _run_full_transcription(self) -> None:
        """Convert the recording and run the CPU-only Parakeet pass."""

        try:
            result = transcribe_recording_file(
                self.recording_file,
                self.audio_file,
                self.transcript_file,
                progress_callback=self._handle_progress_update,
                logger=logger,
            )
            duration = ffmpeg_duration_seconds(self.recording_file)
            self.manager.set_state(
                self.session.session_id,
                SessionState.DONE,
                progress=100,
                duration_seconds=duration,
            )
            self.outgoing_messages.put({"type": "done", "session_id": self.session.session_id})
            logger.info("Full transcription finished for %s with %s segments", self.session.session_id, result.segment_count)
        except Exception as exc:
            logger.exception("Full transcription failed for %s", self.session.session_id)
            self.manager.set_state(
                self.session.session_id,
                SessionState.ERROR,
                progress=100,
                error_message=str(exc),
            )
            self.outgoing_messages.put({"type": "error", "message": str(exc)})
        finally:
            self.manager.unregister_runtime(self.session.session_id)
            self.completed.set()

    def _handle_progress_update(self, stage: str, progress: int) -> None:
        """Persist coarse-grained transcription progress for status polling."""

        self.manager.set_state(self.session.session_id, SessionState.TRANSCRIBING, progress=progress)
        self.outgoing_messages.put({"type": "progress", "stage": stage, "progress": progress})


async def _send_outgoing_messages(websocket: WebSocket, runtime: AudioSessionRuntime) -> None:
    """Drain queued transcript/status messages onto the socket."""

    while True:
        try:
            payload = await asyncio.to_thread(runtime.next_message, 0.25)
            await websocket.send_json(payload)
        except queue.Empty:
            if runtime.completed.is_set():
                return
        except Exception:
            return


@router.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
    """Receive MediaRecorder chunks and coordinate live plus final transcription."""

    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=4400, reason="session_id is required")
        return

    manager = get_session_manager()
    try:
        session = manager.current_or_latest(session_id)
    except KeyError:
        await websocket.close(code=4404, reason="session not found")
        return

    if manager.runtime_for(session.session_id) is not None:
        await websocket.close(code=4409, reason="session already has an active audio stream")
        return

    await websocket.accept()
    try:
        runtime = AudioSessionRuntime(session)
    except Exception as exc:
        logger.exception("Failed to initialize audio runtime for %s", session.session_id)
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011, reason="audio runtime initialization failed")
        return

    sender_task = asyncio.create_task(_send_outgoing_messages(websocket, runtime))
    explicit_stop = False

    try:
        await websocket.send_json({"type": "connected", "session_id": session.session_id})
        while True:
            if manager.stop_requested(session.session_id) or runtime.stop_event.is_set():
                explicit_stop = True
                break

            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            if message.get("bytes") is not None:
                runtime.push_chunk(message["bytes"])
                continue

            text_payload = message.get("text")
            if text_payload is None:
                continue

            try:
                decoded = json.loads(text_payload)
            except json.JSONDecodeError:
                logger.warning("Ignoring non-JSON text payload for %s", session.session_id)
                continue

            if decoded.get("type") == "stop":
                explicit_stop = True
                break
    except WebSocketDisconnect:
        logger.info("Browser disconnected during session %s; finalizing partial audio.", session.session_id)
    finally:
        runtime.finalize("client-stop" if explicit_stop else "disconnect")
        if explicit_stop:
            await asyncio.to_thread(runtime.wait_until_complete, None)
        await sender_task
        try:
            await websocket.close()
        except Exception:
            pass
