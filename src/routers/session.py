from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

try:
    from ..utils import SessionState, get_session_manager
except ImportError:
    from utils import SessionState, get_session_manager


router = APIRouter(tags=["session"])


class StartSessionRequest(BaseModel):
    """Validate the data needed to create a session."""

    session_name: str = Field(default="", max_length=120)
    mode: str = Field(pattern="^(mic|tab)$")


class StopSessionRequest(BaseModel):
    """Allow the browser to stop a specific or current session."""

    session_id: str | None = None


@router.post("/session/start")
async def start_session(payload: StartSessionRequest) -> dict[str, str]:
    """Create a new session directory and reserve a session id."""

    manager = get_session_manager()
    try:
        record = manager.create_session(payload.session_name, payload.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "session_id": record.session_id,
        "session_name": record.session_name,
        "state": record.state.value,
    }


@router.post("/session/stop")
async def stop_session(payload: StopSessionRequest) -> dict[str, str]:
    """Request a graceful stop for the active audio WebSocket runtime."""

    manager = get_session_manager()
    try:
        record = manager.request_stop(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc

    return {
        "session_id": record.session_id,
        "status": SessionState.TRANSCRIBING.value if record.state == SessionState.RECORDING else record.state.value,
    }


@router.get("/session/status")
async def session_status(session_id: str | None = None) -> dict[str, str | int | None]:
    """Expose current session progress for the UI status poller."""

    manager = get_session_manager()
    try:
        return manager.session_status(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc

