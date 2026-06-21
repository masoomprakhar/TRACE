"""FastAPI entrypoint.

Run with:  uvicorn trace_cv.api.main:app --reload
Serves the REST API under /api and the dashboard (if built) at /.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from trace_cv import __version__
from trace_cv.api.live_routes import router as live_router
from trace_cv.api.routes import router

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def create_app() -> FastAPI:
    app = FastAPI(
        title="TRACE — Traffic Rule Analysis & Compliance Engine",
        version=__version__,
        description="Automated traffic-violation detection from images.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(live_router)

    if _DASHBOARD_DIR.exists():
        # Mounted last so /api/* routes take precedence.
        app.mount(
            "/", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard"
        )
    else:  # pragma: no cover
        @app.get("/")
        def root() -> dict:
            return {"service": "TRACE", "version": __version__, "docs": "/docs"}

    return app


app = create_app()
