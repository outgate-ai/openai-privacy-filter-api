"""CLI entrypoint: ``opf-api`` or ``python -m opf_api``.

Configuration precedence: CLI flag > environment variable > built-in default.
See ENVIRONMENT.md for the full list of supported variables.
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .config import Config
from .server import create_app


def _parse_entities(value: str | None) -> tuple[str, ...] | None:
    """CLI counterpart of OPF_API_PRESIDIO_ENTITIES. ``None`` leaves the
    configured value alone; ``*`` means every supported entity."""
    if value is None:
        return None
    if value.strip() == "*":
        return ()
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opf-api",
        description="Ollama-compatible HTTP server for the OpenAI Privacy Filter.",
    )
    parser.add_argument("--host", default=None, help="Host to bind (env: OPF_API_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (env: OPF_API_PORT)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Inference device (env: OPF_API_DEVICE)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override OPF checkpoint directory (env: OPF_API_MODEL_PATH)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name advertised in /api/tags (env: OPF_API_MODEL_NAME)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level: debug|info|warning|error (env: OPF_API_LOG_LEVEL)",
    )
    parser.add_argument(
        "--context-window-length",
        type=int,
        default=None,
        help="Max context length in tokens, up to 131072 (env: OPF_API_CONTEXT_WINDOW_LENGTH)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["typed", "redacted"],
        default=None,
        help="typed keeps per-category labels; redacted collapses to one (env: OPF_API_OUTPUT_MODE)",
    )
    parser.add_argument(
        "--decode-mode",
        choices=["viterbi", "argmax"],
        default=None,
        help="Span decoder: viterbi (coherent) or argmax (faster) (env: OPF_API_DECODE_MODE)",
    )
    parser.add_argument(
        "--viterbi-calibration-path",
        default=None,
        help="Path to a Viterbi calibration artifact for precision/recall tuning "
        "(env: OPF_API_VITERBI_CALIBRATION_PATH)",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="If set, /api/* endpoints require Authorization: Bearer <token> or "
        "X-API-Key: <token> (env: OPF_API_AUTH_TOKEN). Comma-separate several "
        "tokens to accept any of them. Unset means auth disabled.",
    )
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--normalize-whitespace",
        dest="normalize_whitespace",
        action="store_true",
        default=None,
        help="Collapse all whitespace runs (incl. literal \\\\n / \\\\r / \\\\t) to "
        "single spaces before redaction. Improves recall on multi-line input. "
        "Default on. (env: OPF_API_NORMALIZE_WHITESPACE=true)",
    )
    norm_group.add_argument(
        "--no-normalize-whitespace",
        dest="normalize_whitespace",
        action="store_false",
        default=None,
        help="Pass content to the model unchanged (env: OPF_API_NORMALIZE_WHITESPACE=false).",
    )
    presidio_group = parser.add_mutually_exclusive_group()
    presidio_group.add_argument(
        "--presidio",
        dest="presidio_enabled",
        action="store_true",
        default=None,
        help="Run a Presidio analyzer pass alongside OPF and merge the detections "
        "(env: OPF_API_PRESIDIO_ENABLED=true). Off by default.",
    )
    presidio_group.add_argument(
        "--no-presidio",
        dest="presidio_enabled",
        action="store_false",
        default=None,
        help="OPF only (env: OPF_API_PRESIDIO_ENABLED=false).",
    )
    parser.add_argument(
        "--presidio-url",
        default=None,
        help="Base URL of the Presidio analyzer (env: OPF_API_PRESIDIO_URL)",
    )
    parser.add_argument(
        "--presidio-language",
        default=None,
        help="Language code passed to the analyzer (env: OPF_API_PRESIDIO_LANGUAGE)",
    )
    parser.add_argument(
        "--presidio-score-threshold",
        type=float,
        default=None,
        help="Drop analyzer results below this confidence (env: OPF_API_PRESIDIO_SCORE_THRESHOLD)",
    )
    parser.add_argument(
        "--presidio-entities",
        default=None,
        help="Comma-separated entity allowlist, or * for every supported entity "
        "(env: OPF_API_PRESIDIO_ENTITIES)",
    )
    parser.add_argument(
        "--presidio-timeout-ms",
        type=int,
        default=None,
        help="Per-request analyzer timeout in milliseconds (env: OPF_API_PRESIDIO_TIMEOUT_MS)",
    )
    presidio_fail_group = parser.add_mutually_exclusive_group()
    presidio_fail_group.add_argument(
        "--presidio-fail-open",
        dest="presidio_fail_open",
        action="store_true",
        default=None,
        help="An analyzer error degrades the scan to OPF-only. Default on. "
        "(env: OPF_API_PRESIDIO_FAIL_OPEN=true)",
    )
    presidio_fail_group.add_argument(
        "--presidio-fail-closed",
        dest="presidio_fail_open",
        action="store_false",
        default=None,
        help="An analyzer error fails the request with 503 "
        "(env: OPF_API_PRESIDIO_FAIL_OPEN=false).",
    )
    args = parser.parse_args(argv)

    cfg = Config.from_env().override(
        host=args.host,
        port=args.port,
        device=args.device,
        model_path=args.model_path,
        model_name=args.model_name,
        log_level=args.log_level,
        context_window_length=args.context_window_length,
        output_mode=args.output_mode,
        decode_mode=args.decode_mode,
        viterbi_calibration_path=args.viterbi_calibration_path,
        auth_token=args.auth_token,
        normalize_whitespace=args.normalize_whitespace,
        presidio_enabled=args.presidio_enabled,
        presidio_url=args.presidio_url,
        presidio_language=args.presidio_language,
        presidio_score_threshold=args.presidio_score_threshold,
        presidio_entities=_parse_entities(args.presidio_entities),
        presidio_timeout_ms=args.presidio_timeout_ms,
        presidio_fail_open=args.presidio_fail_open,
    )

    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_app(
        device=cfg.device,
        model_path=cfg.model_path,
        model_name=cfg.model_name,
        context_window_length=cfg.context_window_length,
        output_mode=cfg.output_mode,
        decode_mode=cfg.decode_mode,
        viterbi_calibration_path=cfg.viterbi_calibration_path,
        auth_token=cfg.auth_token,
        normalize_whitespace_input=cfg.normalize_whitespace,
        presidio_enabled=cfg.presidio_enabled,
        presidio_url=cfg.presidio_url,
        presidio_language=cfg.presidio_language,
        presidio_score_threshold=cfg.presidio_score_threshold,
        presidio_entities=cfg.presidio_entities,
        presidio_timeout_ms=cfg.presidio_timeout_ms,
        presidio_fail_open=cfg.presidio_fail_open,
        precision_filters=cfg.precision_filters,
        noisy_categories=cfg.noisy_categories,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
