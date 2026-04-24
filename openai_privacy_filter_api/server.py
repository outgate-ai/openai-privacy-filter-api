"""FastAPI app exposing Ollama-compatible endpoints backed by OPF."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .models import (
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    TagModel,
    TagsResponse,
    VersionResponse,
)
from .opf_engine import OPFEngine

logger = logging.getLogger(__name__)

MODEL_NAME = "openai-privacy-filter"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _extract_last_user_content(messages) -> str | None:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return None


def create_app(*, device: Literal["cpu", "cuda"] = "cuda", model_path: str | None = None) -> FastAPI:
    engine = OPFEngine(device=device, model_path=model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.load()
        yield

    app = FastAPI(
        title="OpenAI Privacy Filter API",
        version=__version__,
        description="Ollama-compatible /api/chat server for the OpenAI Privacy Filter model.",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": engine.state, "error": engine.error}

    @app.get("/api/version")
    async def api_version() -> VersionResponse:
        return VersionResponse(version=__version__)

    @app.get("/api/tags")
    async def api_tags() -> TagsResponse:
        return TagsResponse(
            models=[
                TagModel(
                    name=MODEL_NAME,
                    model=MODEL_NAME,
                    modified_at=_iso_now(),
                )
            ]
        )

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest, request: Request):
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")

        if engine.state != "ready":
            raise HTTPException(status_code=503, detail=f"engine not ready (state={engine.state})")

        last_user = _extract_last_user_content(req.messages)
        if last_user is None:
            raise HTTPException(status_code=400, detail="no user message found in request")

        t0 = time.perf_counter_ns()
        try:
            spans = engine.redact(last_user)
        except Exception as exc:
            logger.exception("redaction failed")
            raise HTTPException(status_code=500, detail=f"redaction failed: {exc}") from exc
        total_duration = time.perf_counter_ns() - t0

        payload = [{"text": s.text, "category": s.category} for s in spans]
        content = json.dumps(payload, ensure_ascii=False)

        prompt_eval_count = len(last_user)

        response = ChatResponse(
            model=req.model or MODEL_NAME,
            created_at=_iso_now(),
            message=ChatResponseMessage(role="assistant", content=content),
            done=True,
            done_reason="stop",
            total_duration=total_duration,
            prompt_eval_count=prompt_eval_count,
            eval_count=0,
        )
        return JSONResponse(content=response.model_dump())

    @app.post("/api/generate")
    async def api_generate_unsupported():
        raise HTTPException(status_code=400, detail="/api/generate is not supported; use /api/chat")

    @app.post("/api/embeddings")
    async def api_embeddings_unsupported():
        raise HTTPException(status_code=400, detail="/api/embeddings is not supported")

    return app
