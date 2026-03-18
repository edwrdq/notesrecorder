from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    from .utils import ensure_command, ffmpeg_install_hint, parakeet_model_name, run_command
except ImportError:
    from utils import ensure_command, ffmpeg_install_hint, parakeet_model_name, run_command


ProgressCallback = Callable[[str, int], None]
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, object] = {}


@dataclass(slots=True)
class TranscriptSegment:
    """Represent a single timestamped segment from NeMo."""

    start: float
    end: float
    text: str


@dataclass(slots=True)
class FullTranscriptionResult:
    """Summarize the output of a Parakeet transcription run."""

    audio_file: Path
    transcript_file: Path
    segment_count: int
    model_name: str


def ensure_asr_dependencies():
    """Import NeMo and Torch lazily for clearer startup errors."""

    try:
        import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("NeMo ASR dependencies are missing. Run `uv sync` before transcribing.") from exc

    return nemo_asr, torch


def convert_webm_to_wav(recording_file: Path, audio_file: Path) -> Path:
    """Convert the raw browser recording to a 16 kHz mono WAV for Parakeet."""

    ensure_command("ffmpeg", ffmpeg_install_hint())
    audio_file.parent.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(recording_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_file),
        ],
        check=True,
    )
    return audio_file


def load_parakeet_model(model_name: str, logger: logging.Logger) -> object:
    """Load and cache the CPU-only Parakeet model."""

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        nemo_asr, torch = ensure_asr_dependencies()
        if torch.cuda.is_available():
            logger.warning("CUDA GPU detected, but Parakeet is forced to CPU-only mode.")

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model = model.to("cpu")
        model.eval()
        _MODEL_CACHE[model_name] = model
        return model


def normalize_segments(hypothesis) -> list[TranscriptSegment]:
    """Convert NeMo timestamp output into a stable text format."""

    timestamp_data = getattr(hypothesis, "timestamp", None) or {}
    raw_segments = timestamp_data.get("segment") if isinstance(timestamp_data, dict) else None

    normalized: list[TranscriptSegment] = []
    if raw_segments:
        for segment in raw_segments:
            text = str(segment.get("segment", "")).strip()
            if not text:
                continue
            normalized.append(
                TranscriptSegment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", segment.get("start", 0.0))),
                    text=text,
                )
            )
        return normalized

    text = getattr(hypothesis, "text", "").strip()
    if text:
        normalized.append(TranscriptSegment(start=0.0, end=0.0, text=text))
    return normalized


def format_timestamp(seconds: float) -> str:
    """Format seconds as `HH:MM:SS.mmm` for transcript output."""

    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def write_transcript(segments: Iterable[TranscriptSegment], transcript_file: Path) -> None:
    """Persist the final timestamped transcript text file."""

    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text}"
        for segment in segments
        if segment.text.strip()
    ]
    transcript_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def transcribe_audio_file(
    audio_file: Path,
    transcript_file: Path,
    *,
    model_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
    logger: logging.Logger | None = None,
) -> FullTranscriptionResult:
    """Run a single full-file Parakeet transcription pass on CPU."""

    logger = logger or logging.getLogger("lecture_capture.transcribe")
    model_name = model_name or parakeet_model_name()

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

    if progress_callback is not None:
        progress_callback("loading-model", 35)
    model = load_parakeet_model(model_name, logger)

    if progress_callback is not None:
        progress_callback("transcribing", 80)
    outputs = model.transcribe([str(audio_file)], batch_size=1, timestamps=True)
    if not outputs:
        raise RuntimeError("Parakeet did not return any transcript output.")

    segments = normalize_segments(outputs[0])
    write_transcript(segments, transcript_file)

    if progress_callback is not None:
        progress_callback("done", 100)
    return FullTranscriptionResult(
        audio_file=audio_file,
        transcript_file=transcript_file,
        segment_count=len(segments),
        model_name=model_name,
    )


def transcribe_recording_file(
    recording_file: Path,
    audio_file: Path,
    transcript_file: Path,
    *,
    model_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
    logger: logging.Logger | None = None,
) -> FullTranscriptionResult:
    """Convert a saved browser recording and run Parakeet on the WAV."""

    if progress_callback is not None:
        progress_callback("converting-audio", 15)
    convert_webm_to_wav(recording_file, audio_file)
    return transcribe_audio_file(
        audio_file,
        transcript_file,
        model_name=model_name,
        progress_callback=progress_callback,
        logger=logger,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the standalone transcription entry point."""

    parser = argparse.ArgumentParser(description="Run a full Parakeet transcription pass on browser-captured audio.")
    parser.add_argument("--file", dest="audio_file", type=Path, help="Path to an existing WAV file.")
    parser.add_argument("--recording", type=Path, help="Path to a saved browser WebM recording.")
    parser.add_argument("--audio-output", type=Path, help="Override the WAV output path when using --recording.")
    parser.add_argument("--output", type=Path, help="Override the transcript output path.")
    parser.add_argument("--model", default=parakeet_model_name(), help="Parakeet model name.")
    args = parser.parse_args()

    if bool(args.audio_file) == bool(args.recording):
        parser.error("Choose exactly one of --file or --recording.")

    return args


def main() -> None:
    """Support `uv run src/transcribe.py --file audio.wav`."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = parse_args()

    if args.audio_file:
        audio_file = args.audio_file.resolve()
        transcript_file = args.output.resolve() if args.output else audio_file.with_name("transcript.txt")
        result = transcribe_audio_file(
            audio_file,
            transcript_file,
            model_name=args.model,
            progress_callback=lambda stage, progress: print(f"{stage}: {progress}%"),
        )
    else:
        recording_file = args.recording.resolve()
        audio_file = args.audio_output.resolve() if args.audio_output else recording_file.with_name("audio.wav")
        transcript_file = args.output.resolve() if args.output else recording_file.with_name("transcript.txt")
        result = transcribe_recording_file(
            recording_file,
            audio_file,
            transcript_file,
            model_name=args.model,
            progress_callback=lambda stage, progress: print(f"{stage}: {progress}%"),
        )

    print(f"Transcript written to {result.transcript_file}")
    print(f"Segments: {result.segment_count}")


if __name__ == "__main__":
    main()

