from __future__ import annotations

import logging
from fastapi import FastAPI

from core.paths import paths_summary

# Import new routers
from routes import novel_router, capabilities_router, tts_router, llm_router, imagegen_router
from routes.videogen import router as videogen_router

app = FastAPI(title="Turbo-Super-Novel API", version="0.2.0")
logger = logging.getLogger(__name__)

# Include new routers
app.include_router(novel_router)
app.include_router(capabilities_router)
app.include_router(tts_router)
app.include_router(llm_router)
app.include_router(imagegen_router)
app.include_router(videogen_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/paths")
def paths() -> dict:
    return paths_summary()
