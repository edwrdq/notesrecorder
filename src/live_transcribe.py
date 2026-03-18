from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class LiveTranscriptEvent:
    """Represent a realtime transcript update for the browser."""

    text: str
    timestamp: float


class LiveTranscriber:
    """Wrap RealtimeSTT so raw PCM can be fed from a browser stream."""

    def __init__(
        self,
        *,
        on_transcript: Callable[[LiveTranscriptEvent], None],
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("lecture_capture.live_transcribe")
        self.on_transcript = on_transcript
        self.started_monotonic = time.monotonic()
        self._pcm_queue: queue.Queue[bytes | None] = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self._last_text = ""

        # Import the heavy dependency lazily so CLI helpers that do not need
        # realtime transcription can still start without importing RealtimeSTT.
        try:
            # Import the recorder implementation directly instead of the package
            # top level. RealtimeSTT's __init__ imports its microphone helpers,
            # which hard-require PyAudio even when we explicitly run with
            # use_microphone=False and only feed browser audio.
            from RealtimeSTT.audio_recorder import AudioToTextRecorder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("RealtimeSTT is not installed. Run `uv sync` before starting the server.") from exc

        # The recorder is forced onto CPU even if a GPU exists, matching the
        # project-wide policy for predictable deployment on shared machines.
        self.recorder = AudioToTextRecorder(
            model=os.getenv("REALTIMESTT_MAIN_MODEL", "small.en"),
            realtime_model_type=os.getenv("REALTIMESTT_REALTIME_MODEL", "tiny.en"),
            device="cpu",
            use_microphone=False,
            spinner=False,
            enable_realtime_transcription=True,
            on_realtime_transcription_stabilized=self._handle_stabilized_text,
        )
        self.recorder.start()

        # Audio arrives from FFmpeg on one thread and is fed into RealtimeSTT
        # from a dedicated worker so the WebSocket loop stays responsive.
        self._feed_thread = threading.Thread(target=self._feed_worker, daemon=True)
        self._feed_thread.start()

    def feed_pcm(self, pcm_bytes: bytes) -> None:
        """Queue PCM samples for RealtimeSTT consumption."""

        if self._running.is_set():
            self._pcm_queue.put(pcm_bytes)

    def stop(self) -> None:
        """Shut down the feed worker and recorder cleanly."""

        if not self._running.is_set():
            return

        self._running.clear()
        self._pcm_queue.put(None)
        self._feed_thread.join(timeout=5)
        try:
            self.recorder.stop()
        except Exception:
            pass
        try:
            self.recorder.shutdown()
        except Exception:
            pass

    def _feed_worker(self) -> None:
        """Drain PCM chunks from the queue into RealtimeSTT."""

        while True:
            chunk = self._pcm_queue.get()
            if chunk is None:
                return

            try:
                self.recorder.feed_audio(chunk, original_sample_rate=16000)
            except Exception as exc:
                self.logger.warning("Realtime feed error: %s", exc)

    def _handle_stabilized_text(self, text: str) -> None:
        """Push stabilized transcript text back to the WebSocket sender queue."""

        cleaned = text.strip()
        if not cleaned or cleaned == self._last_text:
            return

        self._last_text = cleaned
        self.on_transcript(
            LiveTranscriptEvent(
                text=cleaned,
                timestamp=round(time.monotonic() - self.started_monotonic, 3),
            )
        )
