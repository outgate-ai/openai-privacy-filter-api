"""FastAPI app exposing Ollama-compatible endpoints backed by OPF."""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import MODEL_NAME_DEFAULT, __version__
from .categories import map_category
from .engine import Engine, OPFEngine
from .merge import SOURCE_OPF, SOURCE_PRESIDIO, MergedDetection, merge_detections
from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIChoiceMessage,
    OpenAIUsage,
    TagModel,
    TagsResponse,
    VersionResponse,
)
from .precision import (
    DEFAULT_NOISY_CATEGORIES,
    apply_precision_filters,
    is_bare_word_secret,
)
from .preprocess import normalize_whitespace
from .presidio import DEFAULT_ENTITIES, PresidioClient, map_entity

logger = logging.getLogger(__name__)

# 1-2 char spans (a stray initial, a single digit) are almost always false positives.
MIN_DETECTION_LEN = 3


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _extract_last_user_content(messages: Iterable[ChatMessage]) -> str | None:
    for msg in reversed(list(messages)):
        if msg.role == "user":
            return msg.content
    return None


def _flatten_openai_content(content: Any) -> str:
    """Reduce an OpenAI message content (str or list of parts) to text.

    Every part that carries text contributes; non-text parts (image_url,
    audio, etc.) are dropped because the redactor only understands text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(p for p in parts if p)
    return str(content)


def _join_openai_messages_for_scan(messages: Iterable[Any]) -> str:
    """Concatenate every message in role-tagged form so the redactor
    sees the whole conversation, not just the last user turn. Same
    `[role] text` shape guardrail's regional service uses, so detection
    outputs are interchangeable across providers.
    """
    chunks: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None) or "unknown"
        raw = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        text = _flatten_openai_content(raw)
        if text:
            chunks.append(f"[{role}] {text}")
    return "\n".join(chunks)


def _extract_presented_token(request: Request) -> str | None:
    """Pull a bearer/api-key token from either Authorization or X-API-Key."""
    auth = request.headers.get("authorization", "")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() or None
    api_key = request.headers.get("x-api-key", "").strip()
    return api_key or None


def parse_auth_tokens(configured: str | None) -> tuple[str, ...]:
    """Split OPF_API_AUTH_TOKEN into the accepted tokens.

    A comma-separated value lets one instance serve callers that hold
    different tokens (two regions sharing a GPU box, for example).
    Whitespace around each token is ignored; empty segments are dropped.
    """
    if not configured:
        return ()
    return tuple(t.strip() for t in configured.split(",") if t.strip())


def _make_auth_dependency(expected_token: str | None):
    """Build a FastAPI dependency that enforces auth when a token is configured."""
    if not expected_token:
        async def _noop() -> None:
            return None

        return _noop

    # A configured but unparsable value (e.g. ",") must not silently
    # disable auth, so the dependency is installed with no valid token.
    accepted = parse_auth_tokens(expected_token)

    def _matches(presented: str) -> bool:
        ok = False
        for token in accepted:
            ok = hmac.compare_digest(presented, token) or ok
        return ok

    async def _require_auth(request: Request) -> None:
        presented = _extract_presented_token(request)
        if presented is None or not _matches(presented):
            client_host = request.client.host if request.client else "?"
            logger.warning(
                "auth failed rid=%s ip=%s path=%s reason=%s",
                getattr(request.state, "request_id", "?"),
                client_host,
                request.url.path,
                "missing" if presented is None else "mismatch",
            )
            raise HTTPException(
                status_code=401,
                detail="invalid or missing authentication token",
                headers={"WWW-Authenticate": 'Bearer realm="opf-api"'},
            )

    return _require_auth


def create_app(
    *,
    engine: Engine | None = None,
    presidio_client: PresidioClient | None = None,
    device: str = "cuda",
    model_path: str | None = None,
    model_name: str = MODEL_NAME_DEFAULT,
    context_window_length: int | None = None,
    output_mode: str = "typed",
    decode_mode: str = "viterbi",
    viterbi_calibration_path: str | None = None,
    auth_token: str | None = None,
    normalize_whitespace_input: bool = True,
    presidio_enabled: bool = False,
    presidio_url: str = "http://presidio:3000",
    presidio_language: str = "en",
    presidio_score_threshold: float = 0.5,
    presidio_entities: tuple[str, ...] | None = DEFAULT_ENTITIES,
    presidio_timeout_ms: int = 3000,
    presidio_fail_open: bool = True,
    precision_filters: bool = True,
    noisy_categories: tuple[str, ...] = DEFAULT_NOISY_CATEGORIES,
) -> FastAPI:
    """Build the FastAPI app.

    If ``engine`` is provided it is used as-is (tests inject a fake). Otherwise
    an ``OPFEngine`` is constructed and loaded in the lifespan startup hook.
    """
    noisy_set = frozenset(noisy_categories)

    owned_presidio = presidio_client is None and presidio_enabled
    if owned_presidio:
        presidio_client = PresidioClient(
            url=presidio_url,
            language=presidio_language,
            score_threshold=presidio_score_threshold,
            entities=presidio_entities,
            timeout_ms=presidio_timeout_ms,
        )

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
        try:
            yield
        finally:
            if owned_presidio and presidio_client is not None:
                await presidio_client.aclose()

    app = FastAPI(
        title="OpenAI Privacy Filter API",
        version=__version__,
        description="Ollama-compatible /api/chat server for the OpenAI Privacy Filter model.",
        lifespan=lifespan,
    )

    async def _detect(text: str, request_id: str) -> list[dict[str, str]]:
        """Run OPF, optionally add a Presidio pass, and merge the two.

        With the Presidio pass off the response stays exactly what it was
        before the pass existed: OPF order, no ``source`` field.
        """
        if presidio_client is None:
            return [
                {
                    "text": span.text,
                    "category": map_category(span.category),
                    "source_category": span.category,
                }
                for span in engine.redact(text)
                if len(span.text) >= MIN_DETECTION_LEN
                and not (
                    precision_filters
                    and span.category == "secret"
                    and is_bare_word_secret(span.text)
                )
            ]

        detections = [
            MergedDetection(
                text=span.text,
                category=map_category(span.category),
                source_category=span.category,
                source=SOURCE_OPF,
            )
            for span in engine.redact(text)
        ]

        try:
            presidio_spans = await presidio_client.analyze(text)
        except Exception as exc:
            # Presidio is the secondary detector: by default a failure
            # degrades to OPF-only rather than dropping the whole scan.
            logger.warning("presidio analyze failed rid=%s error=%s", request_id, exc)
            if not presidio_fail_open:
                raise HTTPException(
                    status_code=503, detail="presidio analyzer unavailable"
                ) from exc
        else:
            detections.extend(
                MergedDetection(
                    text=span.text,
                    category=map_entity(span.entity_type),
                    source_category=span.entity_type,
                    source=SOURCE_PRESIDIO,
                )
                for span in presidio_spans
            )

        merged = merge_detections(detections, min_length=MIN_DETECTION_LEN)
        if precision_filters:
            merged = apply_precision_filters(
                merged,
                corroboration_available=True,
                noisy_categories=noisy_set,
            )
        return [d.as_dict() for d in merged]

    require_auth = _make_auth_dependency(auth_token)
    if presidio_client is not None:
        logger.info(
            "presidio pass enabled (url=%s, entities=%s, score>=%.2f, fail_open=%s)",
            presidio_url,
            "*" if presidio_entities is None else ",".join(presidio_entities),
            presidio_score_threshold,
            presidio_fail_open,
        )

    if auth_token:
        logger.info(
            "authentication enabled on /api/* endpoints (%d accepted token(s))",
            len(parse_auth_tokens(auth_token)),
        )
    else:
        logger.info("authentication disabled (set OPF_API_AUTH_TOKEN to enable)")

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

    @app.get("/api/version", dependencies=[Depends(require_auth)])
    async def api_version() -> VersionResponse:
        return VersionResponse(version=__version__)

    @app.get("/api/tags", dependencies=[Depends(require_auth)])
    async def api_tags() -> TagsResponse:
        return TagsResponse(
            models=[TagModel(name=model_name, model=model_name, modified_at=_iso_now())]
        )

    @app.post("/api/chat", dependencies=[Depends(require_auth)])
    async def api_chat(req: ChatRequest, request: Request):
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")

        if not req.model or req.model != model_name:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"model {req.model!r} not found, "
                    f"this server only serves {model_name!r}"
                ),
            )

        if engine.state != "ready":
            raise HTTPException(
                status_code=503, detail=f"engine not ready (state={engine.state})"
            )

        last_user = _extract_last_user_content(req.messages)
        if last_user is None:
            raise HTTPException(status_code=400, detail="no user message found in request")

        if normalize_whitespace_input:
            last_user = normalize_whitespace(last_user)

        t0 = time.perf_counter_ns()
        try:
            detections = await _detect(last_user, getattr(request.state, "request_id", "?"))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("redaction failed")
            raise HTTPException(
                status_code=500, detail="internal error during redaction"
            ) from exc
        total_duration = time.perf_counter_ns() - t0

        content = json.dumps({"detections": detections}, ensure_ascii=False)

        response = ChatResponse(
            model=model_name,
            created_at=_iso_now(),
            message=ChatResponseMessage(role="assistant", content=content),
            done=True,
            done_reason="stop",
            total_duration=total_duration,
            prompt_eval_count=len(last_user),
            eval_count=0,
        )
        return JSONResponse(content=response.model_dump())

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    @app.post("/chat/completions", dependencies=[Depends(require_auth)])
    async def openai_chat_completions(req: OpenAIChatRequest, request: Request):
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        if not req.model or req.model != model_name:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"model {req.model!r} not found, "
                    f"this server only serves {model_name!r}"
                ),
            )
        if engine.state != "ready":
            raise HTTPException(
                status_code=503, detail=f"engine not ready (state={engine.state})"
            )

        # Every message, not just the last user turn: system prompts carry
        # credentials, tool messages carry agent-crafted parameters.
        scan_input = _join_openai_messages_for_scan(req.messages)
        if not scan_input:
            raise HTTPException(status_code=400, detail="no text content found in request")

        if normalize_whitespace_input:
            scan_input = normalize_whitespace(scan_input)

        try:
            detections = await _detect(scan_input, getattr(request.state, "request_id", "?"))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("redaction failed")
            raise HTTPException(
                status_code=500, detail="internal error during redaction"
            ) from exc

        content = json.dumps({"detections": detections}, ensure_ascii=False)

        response = OpenAIChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model_name,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChoiceMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=len(scan_input),
                completion_tokens=0,
                total_tokens=len(scan_input),
            ),
        )
        return JSONResponse(content=response.model_dump())

    @app.post("/api/generate")
    async def api_generate_unsupported():
        raise HTTPException(status_code=400, detail="/api/generate is not supported; use /api/chat")

    @app.post("/api/embeddings")
    async def api_embeddings_unsupported():
        raise HTTPException(status_code=400, detail="/api/embeddings is not supported")

    return app
