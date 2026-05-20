from __future__ import annotations

import resource

# The model loader can fan out across many safetensors shards.
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(_soft, 65536), max(_hard, 65536)))
except Exception:
    pass

import argparse
import os

from aiohttp import web

from .app import create_app


BACKEND_CHOICES = ("auto", "cpu", "hybrid", "npu")


def _default_backend() -> str:
    value = os.getenv("PCCX_BACKEND", "auto").strip().lower()
    return value if value in BACKEND_CHOICES else "auto"


def main() -> None:
    parser = argparse.ArgumentParser(description="PCCX KV260 Gemma daemon")
    parser.add_argument("--model", help="path to the local Gemma model directory")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", default=7860, type=int, help="bind port")
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=_default_backend(),
        help="inference backend: auto, cpu, hybrid, or strict npu",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="start HTTP and WebSocket routes without loading a model",
    )
    args = parser.parse_args()
    model_path = None if args.no_model else args.model
    app = create_app(
        None,
        host=args.host,
        port=args.port,
        model_path=model_path,
        backend=args.backend,
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
