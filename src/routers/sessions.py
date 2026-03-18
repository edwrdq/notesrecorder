from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

try:
    from ..utils import get_session_manager
except ImportError:
    from utils import get_session_manager


router = APIRouter(tags=["sessions"])


@router.get("/sessions")
async def list_sessions() -> list[dict[str, object]]:
    """List completed and in-progress sessions for the sidebar."""

    return get_session_manager().list_sessions()


@router.get("/sessions/{session_id}")
async def session_detail(session_id: str) -> dict[str, object]:
    """Return metadata plus transcript text for inline browsing."""

    manager = get_session_manager()
    try:
        return manager.session_detail(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


@router.get("/sessions/{session_id}/transcript")
async def download_transcript(session_id: str) -> FileResponse:
    """Return the transcript file as a direct download."""

    manager = get_session_manager()
    try:
        record = manager.current_or_latest(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc

    if not record.transcript_file.exists():
        raise HTTPException(status_code=404, detail="Transcript not found.")

    return FileResponse(
        record.transcript_file,
        media_type="text/plain; charset=utf-8",
        filename=record.transcript_file.name,
    )

