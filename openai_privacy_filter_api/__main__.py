"""CLI entrypoint: `opf-api` or `python -m openai_privacy_filter_api`."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .server import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opf-api",
        description="Ollama-compatible HTTP server for the OpenAI Privacy Filter.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11435, help="Port to bind (default: 11435)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override OPF checkpoint directory (default: $OPF_CHECKPOINT or ~/.opf/privacy_filter)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Uvicorn/root log level (default: info)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_app(device=args.device, model_path=args.model_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
