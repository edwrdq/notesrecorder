from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .routers.session import router as session_router
    from .routers.sessions import router as sessions_router
    from .utils import STATIC_DIR, configure_logging, get_session_manager, load_environment, log_gpu_policy
    from .ws.audio import router as audio_router
except ImportError:
    from routers.session import router as session_router
    from routers.sessions import router as sessions_router
    from utils import STATIC_DIR, configure_logging, get_session_manager, load_environment, log_gpu_policy
    from ws.audio import router as audio_router


load_environment()
configure_logging()
logger = logging.getLogger("lecture_capture.main")
session_manager = get_session_manager()

app = FastAPI(title="Lecture Capture", version="0.1.0")

# The single-page UI lives under /static, while / serves index.html directly.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(session_router)
app.include_router(sessions_router)
app.include_router(audio_router)


@app.on_event("startup")
async def startup_event() -> None:
    """Ensure the lectures directory exists and log runtime policy."""

    session_manager.base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Lecture directory: %s", session_manager.base_dir)
    log_gpu_policy(logger)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Request graceful finalization for active audio sessions."""

    logger.info("Application shutdown requested; finalizing active sessions.")
    session_manager.shutdown_active_runtimes()


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-file web UI."""

    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    """Support direct execution with `uv run src/main.py`."""

    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
