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
    args = parser.parse_args(argv)

    cfg = Config.from_env().override(
        host=args.host,
        port=args.port,
        device=args.device,
        model_path=args.model_path,
        model_name=args.model_name,
        log_level=args.log_level,
    )

    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_app(
        device=cfg.device,
        model_path=cfg.model_path,
        model_name=cfg.model_name,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
