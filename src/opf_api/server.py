"""FastAPI app exposing Ollama-compatible endpoints backed by OPF."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import MODEL_NAME_DEFAULT, __version__
from .engine import Engine, OPFEngine
from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    TagModel,
    TagsResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _extract_last_user_content(messages: Iterable[ChatMessage]) -> str | None:
    for msg in reversed(list(messages)):
        if msg.role == "user":
            return msg.content
    return None


def create_app(
    *,
    engine: Engine | None = None,
    device: str = "cuda",
    model_path: str | None = None,
    model_name: str = MODEL_NAME_DEFAULT,
    context_window_length: int | None = None,
    output_mode: str = "typed",
    decode_mode: str = "viterbi",
    viterbi_calibration_path: str | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    If ``engine`` is provided it is used as-is (tests inject a fake). Otherwise
    an ``OPFEngine`` is constructed and loaded in the lifespan startup hook.
    """
    owned_engine = engine is None
    if engine is None:
        engine = OPFEngine(  # type: ignore[arg-type]
            device=device,
            model_path=model_path,
            context_window_length=context_window_length,
            output_mode=output_mode,
            decode_mode=decode_mode,
            viterbi_calibration_path=viterbi_calibration_path,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if owned_engine:
            engine.load()
        yield

    app = FastAPI(
        title="OpenAI Privacy Filter API",
        version=__version__,
        description="Ollama-compatible /api/chat server for the OpenAI Privacy Filter model.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed rid=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rid=%s %s %s -> %d %.1fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health")
    async def health():
        return {"status": engine.state, "error": engine.error}

    @app.get("/api/version")
    async def api_version() -> VersionResponse:
        return VersionResponse(version=__version__)

    @app.get("/api/tags")
    async def api_tags() -> TagsResponse:
        return TagsResponse(
            models=[TagModel(name=model_name, model=model_name, modified_at=_iso_now())]
        )

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")

        if engine.state != "ready":
            raise HTTPException(
                status_code=503, detail=f"engine not ready (state={engine.state})"
            )

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

        response = ChatResponse(
            model=req.model or model_name,
            created_at=_iso_now(),
            message=ChatResponseMessage(role="assistant", content=content),
            done=True,
            done_reason="stop",
            total_duration=total_duration,
            prompt_eval_count=len(last_user),
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
