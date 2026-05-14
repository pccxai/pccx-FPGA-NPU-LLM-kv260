"""Command-line smoke runner for the Gemma package.

Depends on Core Secrets 1-7 through GemmaInferenceSession.
"""
from __future__ import annotations

import argparse
import sys

from .inference import GemmaInferenceSession
from .weights import DEFAULT_MODEL_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Gemma 3N E4B smoke prompt.")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="model directory")
    parser.add_argument("--prompt", default="hello", help="prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--cpu", action="store_true", help="disable NPU dispatch")
    args = parser.parse_args(argv)

    session = GemmaInferenceSession(args.model, use_npu=not args.cpu)
    try:
        for piece, _event in session.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            print(piece, end="", flush=True)
        print()
    except FileNotFoundError as exc:
        print(f"model weights unavailable: {exc}", file=sys.stderr)
        return 2
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
