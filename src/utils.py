from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "src" / "static"
DEFAULT_LECTURES_DIR = Path.home() / "lectures"
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"

_SESSION_MANAGER: "SessionManager | None" = None


class SessionState(StrEnum):
    """Enumerate the high-level lifecycle states surfaced to the browser."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class SessionPaths:
    """Collect all filesystem locations for a single lecture session."""

    session_dir: Path
    recording_file: Path
    audio_file: Path
    transcript_file: Path
    metadata_file: Path


@dataclass(slots=True)
class SessionRecord:
    """Hold persisted and in-memory metadata about a session."""

    session_id: str
    session_name: str
    mode: str
    state: SessionState
    created_at: str
    updated_at: str
    progress: int
    session_dir: Path
    recording_file: Path
    audio_file: Path
    transcript_file: Path
    metadata_file: Path
    duration_seconds: float | None = None
    error_message: str | None = None
    stop_requested: bool = False

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize the record for disk persistence."""

        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "mode": self.mode,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "stop_requested": self.stop_requested,
            "session_dir": str(self.session_dir),
            "recording_file": str(self.recording_file),
            "audio_file": str(self.audio_file),
            "transcript_file": str(self.transcript_file),
            "metadata_file": str(self.metadata_file),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize the record for REST responses."""

        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "mode": self.mode,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "path": str(self.session_dir),
            "recording_file": str(self.recording_file),
            "audio_file": str(self.audio_file),
            "transcript_file": str(self.transcript_file),
            "download_url": f"/sessions/{self.session_id}/transcript",
            "details_url": f"/sessions/{self.session_id}",
        }

    @classmethod
    def from_metadata_dict(cls, raw: dict[str, Any]) -> "SessionRecord":
        """Rehydrate a record from disk."""

        return cls(
            session_id=str(raw["session_id"]),
            session_name=str(raw["session_name"]),
            mode=str(raw["mode"]),
            state=SessionState(str(raw["state"])),
            created_at=str(raw["created_at"]),
            updated_at=str(raw["updated_at"]),
            progress=int(raw.get("progress", 0)),
            duration_seconds=raw.get("duration_seconds"),
            error_message=raw.get("error_message"),
            stop_requested=bool(raw.get("stop_requested", False)),
            session_dir=Path(raw["session_dir"]),
            recording_file=Path(raw["recording_file"]),
            audio_file=Path(raw["audio_file"]),
            transcript_file=Path(raw["transcript_file"]),
            metadata_file=Path(raw["metadata_file"]),
        )


def project_root() -> Path:
    """Return the repository root."""

    return PROJECT_ROOT


def load_environment() -> Path:
    """Load `.env` from the repository root when it exists."""

    env_path = project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return env_path


def configure_logging() -> None:
    """Set up a simple process-wide logging format."""

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def lectures_dir() -> Path:
    """Resolve the session root directory from configuration."""

    configured = os.getenv("LECTURES_DIR", "").strip()
    if configured:
        return Path(os.path.expanduser(os.path.expandvars(configured))).resolve()
    return DEFAULT_LECTURES_DIR


def parakeet_model_name() -> str:
    """Return the configured NeMo checkpoint name."""

    return os.getenv("PARAKEET_MODEL_NAME", DEFAULT_PARAKEET_MODEL)


def command_exists(command: str) -> bool:
    """Check whether a command is available on PATH."""

    from shutil import which

    return which(command) is not None


def ensure_command(command: str, install_hint: str | None = None) -> None:
    """Raise a friendly error when a host dependency is missing."""

    if command_exists(command):
        return

    message = [f"Required command '{command}' is not available on PATH."]
    if install_hint:
        message.append(install_hint)
    raise RuntimeError(" ".join(message))


def ffmpeg_install_hint() -> str:
    """Return a container and Debian-friendly FFmpeg install hint."""

    return "Install FFmpeg with apt, for example: apt-get update && apt-get install -y ffmpeg"


def run_command(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and surface missing-command errors clearly."""

    try:
        return subprocess.run(
            args,
            check=check,
            capture_output=capture_output,
            text=text,
            cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError as exc:
        hint = ffmpeg_install_hint() if args and args[0] in {"ffmpeg", "ffprobe"} else None
        raise RuntimeError(f"Command '{args[0]}' could not be started. {hint or ''}".strip()) from exc


def utc_now_iso() -> str:
    """Return a stable local timestamp string with seconds precision."""

    return datetime.now().isoformat(timespec="seconds")


def sanitize_session_name(raw_name: str | None) -> str:
    """Map a free-form session name to a safe folder suffix."""

    candidate = (raw_name or "").strip()
    if not candidate:
        candidate = datetime.now().strftime("session-%Y-%m-%d-%H-%M")

    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate)
    candidate = re.sub(r"-{2,}", "-", candidate)
    return candidate.strip("-_.") or datetime.now().strftime("session-%Y-%m-%d-%H-%M")


def build_session_paths(session_name: str, base_dir: Path | None = None) -> SessionPaths:
    """Create a session directory and all expected output paths."""

    timestamp = datetime.now()
    safe_name = sanitize_session_name(session_name)
    session_id = f"{timestamp.strftime('%Y-%m-%d_%H-%M')}_{safe_name}"
    session_dir = (base_dir or lectures_dir()) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return SessionPaths(
        session_dir=session_dir,
        recording_file=session_dir / "recording.webm",
        audio_file=session_dir / "audio.wav",
        transcript_file=session_dir / "transcript.txt",
        metadata_file=session_dir / "session.json",
    )


def ffprobe_duration_seconds(media_file: Path) -> float | None:
    """Read media duration through ffprobe when available."""

    if not media_file.exists():
        return None

    if not command_exists("ffprobe"):
        return None

    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_file),
        ],
        check=False,
    )
    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def ffmpeg_duration_seconds(media_file: Path) -> float | None:
    """Provide a more descriptive alias for callers that care about media length."""

    return ffprobe_duration_seconds(media_file)


def transcript_text(transcript_file: Path) -> str:
    """Read transcript text safely for session detail views."""

    if not transcript_file.exists():
        return ""
    return transcript_file.read_text(encoding="utf-8")


class SessionManager:
    """Own in-memory session state and persist metadata to disk."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionRecord] = {}
        self._active_runtimes: dict[str, Any] = {}
        self._current_session_id: str | None = None

    def create_session(self, session_name: str, mode: str) -> SessionRecord:
        """Create a fresh session directory and metadata record."""

        with self._lock:
            active = self.active_session()
            if active and active.state in {SessionState.RECORDING, SessionState.TRANSCRIBING}:
                raise RuntimeError("Another session is already active. Wait for it to finish before starting a new one.")

            self.base_dir.mkdir(parents=True, exist_ok=True)
            paths = build_session_paths(session_name, self.base_dir)
            created_at = utc_now_iso()
            record = SessionRecord(
                session_id=paths.session_dir.name,
                session_name=sanitize_session_name(session_name),
                mode=mode,
                state=SessionState.IDLE,
                created_at=created_at,
                updated_at=created_at,
                progress=0,
                session_dir=paths.session_dir,
                recording_file=paths.recording_file,
                audio_file=paths.audio_file,
                transcript_file=paths.transcript_file,
                metadata_file=paths.metadata_file,
            )
            self._sessions[record.session_id] = record
            self._current_session_id = record.session_id
            self._write_metadata_locked(record)
            return record

    def active_session(self) -> SessionRecord | None:
        """Return the most recent session that is still in memory."""

        if self._current_session_id is None:
            return None
        return self._sessions.get(self._current_session_id)

    def current_or_latest(self, session_id: str | None = None) -> SessionRecord:
        """Resolve a specific session id or fall back to the current/latest session."""

        with self._lock:
            if session_id:
                record = self._sessions.get(session_id)
                if record:
                    return record
                metadata_path = self.base_dir / session_id / "session.json"
                if metadata_path.exists():
                    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                    record = SessionRecord.from_metadata_dict(raw)
                    self._sessions[record.session_id] = record
                    return record
                raise KeyError(session_id)

            if self._current_session_id and self._current_session_id in self._sessions:
                return self._sessions[self._current_session_id]

            sessions = self.list_sessions()
            if not sessions:
                raise KeyError("No sessions found.")
            latest_id = sessions[0]["session_id"]
            return self.current_or_latest(str(latest_id))

    def register_runtime(self, session_id: str, runtime: Any) -> None:
        """Track an active audio runtime so shutdown and REST stop requests can reach it."""

        with self._lock:
            self._active_runtimes[session_id] = runtime

    def unregister_runtime(self, session_id: str) -> None:
        """Drop the runtime handle once recording and transcription finish."""

        with self._lock:
            self._active_runtimes.pop(session_id, None)

    def runtime_for(self, session_id: str) -> Any | None:
        """Return the registered runtime for a session when one exists."""

        with self._lock:
            return self._active_runtimes.get(session_id)

    def request_stop(self, session_id: str | None = None) -> SessionRecord:
        """Mark a session for stop so the WebSocket loop can finalize it."""

        with self._lock:
            record = self.current_or_latest(session_id)
            record.stop_requested = True
            record.updated_at = utc_now_iso()
            self._write_metadata_locked(record)
            runtime = self._active_runtimes.get(record.session_id)
            if runtime is not None:
                runtime.request_stop()
            return record

    def stop_requested(self, session_id: str) -> bool:
        """Check whether a session has been told to stop."""

        with self._lock:
            record = self.current_or_latest(session_id)
            return record.stop_requested

    def clear_stop_request(self, session_id: str) -> None:
        """Reset the stop flag after finalization begins."""

        with self._lock:
            record = self.current_or_latest(session_id)
            record.stop_requested = False
            record.updated_at = utc_now_iso()
            self._write_metadata_locked(record)

    def set_state(
        self,
        session_id: str,
        state: SessionState,
        *,
        progress: int | None = None,
        error_message: str | None = None,
        duration_seconds: float | None = None,
    ) -> SessionRecord:
        """Update core status fields and persist them immediately."""

        with self._lock:
            record = self.current_or_latest(session_id)
            record.state = state
            record.updated_at = utc_now_iso()
            if progress is not None:
                record.progress = int(progress)
            if error_message is not None:
                record.error_message = error_message
            if duration_seconds is not None:
                record.duration_seconds = duration_seconds
            self._write_metadata_locked(record)
            return record

    def session_status(self, session_id: str | None = None) -> dict[str, Any]:
        """Return a status payload suitable for `GET /session/status`."""

        record = self.current_or_latest(session_id)
        return {
            "session_id": record.session_id,
            "state": record.state.value,
            "progress": record.progress,
            "error_message": record.error_message,
        }

    def session_detail(self, session_id: str) -> dict[str, Any]:
        """Return session metadata plus transcript text for the UI detail view."""

        record = self.current_or_latest(session_id)
        detail = record.to_public_dict()
        detail["transcript_text"] = transcript_text(record.transcript_file)
        return detail

    def list_sessions(self) -> list[dict[str, Any]]:
        """Load session metadata from disk for the sidebar."""

        sessions: list[dict[str, Any]] = []
        if not self.base_dir.exists():
            return sessions

        for metadata_path in sorted(self.base_dir.glob("*/session.json"), reverse=True):
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                record = SessionRecord.from_metadata_dict(raw)
            except Exception:
                continue

            # Keep duration reasonably fresh for completed sessions even if the
            # server was restarted before metadata could be updated.
            if record.duration_seconds is None and record.recording_file.exists():
                duration = ffprobe_duration_seconds(record.recording_file)
                if duration is not None:
                    record.duration_seconds = duration
            sessions.append(record.to_public_dict())

        return sessions

    def shutdown_active_runtimes(self) -> None:
        """Ask active audio pipelines to stop and wait briefly for cleanup."""

        with self._lock:
            runtimes = list(self._active_runtimes.values())

        for runtime in runtimes:
            try:
                runtime.request_stop()
                runtime.finalize("shutdown")
            except Exception:
                continue

        deadline = time.time() + 8
        for runtime in runtimes:
            remaining = max(0, deadline - time.time())
            try:
                runtime.wait_until_complete(timeout=remaining)
            except Exception:
                continue

    def _write_metadata_locked(self, record: SessionRecord) -> None:
        """Persist session metadata to `session.json`."""

        record.metadata_file.write_text(json.dumps(record.to_metadata_dict(), indent=2), encoding="utf-8")


def get_session_manager() -> SessionManager:
    """Return the process-wide session manager singleton."""

    global _SESSION_MANAGER
    if _SESSION_MANAGER is None:
        _SESSION_MANAGER = SessionManager(lectures_dir())
    return _SESSION_MANAGER


def log_gpu_policy(logger: logging.Logger) -> None:
    """Warn when a GPU exists because the app intentionally stays CPU-only."""

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        logger.info("Torch is not imported yet; GPU policy will be checked during transcription startup.")
        return

    if torch.cuda.is_available():
        logger.warning("CUDA GPU detected, but this application is configured to run CPU-only.")
    else:
        logger.info("No CUDA GPU detected; running CPU-only as expected.")
