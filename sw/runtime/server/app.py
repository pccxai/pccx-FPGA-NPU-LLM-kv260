"""aiohttp application factory for the KV260 inference daemon."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web

from .trace_emitter import TraceEmitter


VERSION = "v002.1-v12d"
DEFAULT_MODEL_ID = "google/gemma-3n-E4B-it"
BITSTREAM_PATH = Path("/lib/firmware/xilinx/pccx_npu_bd/pccx_npu_bd.bit.bin")


@dataclass
class ServerState:
    session: Any = None
    host: str = "0.0.0.0"
    port: int = 7860
    started_at: float = field(default_factory=time.monotonic)
    elapsed_load_sec: float = 0.0
    load_started_at: Optional[float] = None
    loading: bool = False
    load_error: Optional[str] = None
    model_path: Optional[str] = None
    histories: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    tokens_total: int = 0
    tokens_per_sec_last: float = 0.0

    @property
    def model_loaded(self) -> bool:
        return self.session is not None and not self.loading and self.load_error is None

    @property
    def model_id(self) -> str:
        for attr in ("model_id", "model_name", "name"):
            value = getattr(self.session, attr, None)
            if value:
                return str(value)
        return DEFAULT_MODEL_ID

    @property
    def backend(self) -> str:
        value = getattr(self.session, "backend", None)
        if value in {"cpu", "npu_uca", "hybrid"}:
            return str(value)
        return "cpu"

    def ensure_session(self, session_id: str) -> List[Dict[str, str]]:
        return self.histories.setdefault(session_id, [])

    def reset_session(self, session_id: str) -> None:
        self.histories.pop(session_id, None)

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        history = self.ensure_session(session_id)
        history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        self.histories[session_id] = history[-16:]

    async def load_model(self, model_path: str) -> None:
        self.model_path = model_path
        self.loading = True
        self.load_error = None
        self.load_started_at = time.monotonic()
        try:
            session = await asyncio.to_thread(_construct_gemma_session, model_path)
            await _maybe_call_lifecycle(session, model_path)
            self.session = session
        except Exception as exc:  # pragma: no cover - exercised on board images.
            self.session = None
            self.load_error = f"{type(exc).__name__}: {exc}"
        finally:
            started = self.load_started_at or time.monotonic()
            self.elapsed_load_sec = time.monotonic() - started
            self.load_started_at = None
            self.loading = False

    async def close_session(self) -> None:
        session = self.session
        if session is None:
            return
        close = getattr(session, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


SERVER_STATE_KEY = web.AppKey("server_state", ServerState)
TRACE_EMITTER_KEY = web.AppKey("trace_emitter", TraceEmitter)
MODEL_LOAD_TASK_KEY = web.AppKey("model_load_task", asyncio.Task)


@web.middleware
async def _cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    if not response.prepared:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "86400"
    return response


def create_app(
    session: Any,
    *,
    host: str,
    port: int,
    model_path: Optional[str] = None,
) -> web.Application:
    app = web.Application(middlewares=[_cors_middleware])
    app[SERVER_STATE_KEY] = ServerState(
        session=session,
        host=host,
        port=port,
        model_path=os.path.expanduser(model_path) if model_path else None,
    )
    app[TRACE_EMITTER_KEY] = TraceEmitter()

    from .routes_http import register_http_routes
    from .routes_ws import register_ws_routes

    register_http_routes(app)
    register_ws_routes(app)

    static_dir = Path(__file__).with_name("static")
    app.router.add_static("/static/", path=static_dir, name="static")

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


async def _on_startup(app: web.Application) -> None:
    state: ServerState = app[SERVER_STATE_KEY]
    if state.session is None and state.model_path:
        app[MODEL_LOAD_TASK_KEY] = asyncio.create_task(state.load_model(state.model_path))


async def _on_cleanup(app: web.Application) -> None:
    task = app.get(MODEL_LOAD_TASK_KEY)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    state: ServerState = app[SERVER_STATE_KEY]
    await state.close_session()


def health_payload(state: ServerState) -> Dict[str, Any]:
    if state.loading:
        started = state.load_started_at or state.started_at
        return {
            "status": "loading",
            "model": state.model_id,
            "model_loaded": False,
            "elapsed_load_sec": round(time.monotonic() - started, 3),
            "version": VERSION,
        }
    if state.load_error is not None:
        return {
            "status": "error",
            "model": state.model_id,
            "model_loaded": False,
            "elapsed_load_sec": round(state.elapsed_load_sec, 3),
            "version": VERSION,
            "error": state.load_error,
        }
    return {
        "status": "ready",
        "model": state.model_id,
        "model_loaded": state.model_loaded,
        "elapsed_load_sec": round(state.elapsed_load_sec, 3),
        "version": VERSION,
    }


def status_payload(state: ServerState) -> Dict[str, Any]:
    npu = read_npu_status()
    return {
        "model_id": state.model_id,
        "model_loaded": state.model_loaded,
        "session_count": len(state.histories),
        "tokens_total": state.tokens_total,
        "tokens_per_sec_last": round(state.tokens_per_sec_last, 3),
        "npu_mmio_stat_hex": npu["npu_mmio_stat_hex"],
        "npu_busy": npu["npu_busy"],
        "npu_done": npu["npu_done"],
        "npu_available": npu["npu_available"],
        "backend": state.backend,
        "uptime_sec": int(time.monotonic() - state.started_at),
        "bitstream_sha": bitstream_sha256(),
        "host": socket.gethostname(),
        "ram_mb": ram_mb(),
    }


def read_npu_status() -> Dict[str, Any]:
    fallback = {
        "npu_mmio_stat_hex": "0x00000000",
        "npu_busy": False,
        "npu_done": False,
        "npu_available": False,
    }
    try:
        module = importlib.import_module("sw.runtime.npu")
        raw = module.npu_status()
    except Exception:
        return fallback
    if isinstance(raw, int):
        stat = raw & 0xFFFF_FFFF
        return {
            "npu_mmio_stat_hex": f"0x{stat:08x}",
            "npu_busy": bool(stat & 0x1),
            "npu_done": bool(stat & 0x2),
            "npu_available": True,
        }
    if isinstance(raw, dict):
        stat_value = raw.get("npu_mmio_stat_hex", raw.get("mmio_stat_hex"))
        if stat_value is None:
            stat_value = raw.get("npu_mmio_stat", raw.get("mmio_stat", 0))
        stat = _parse_stat_value(stat_value)
        return {
            "npu_mmio_stat_hex": f"0x{stat:08x}",
            "npu_busy": bool(raw.get("npu_busy", raw.get("busy", stat & 0x1))),
            "npu_done": bool(raw.get("npu_done", raw.get("done", stat & 0x2))),
            "npu_available": bool(raw.get("npu_available", raw.get("available", True))),
        }
    return fallback


def bitstream_sha256(path: Path = BITSTREAM_PATH) -> str:
    if not path.exists():
        return "0" * 64
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "0" * 64
    return digest.hexdigest()


def ram_mb() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size / (1024 * 1024))
        except (OSError, ValueError):
            return 0


def _parse_stat_value(value: Any) -> int:
    if isinstance(value, str):
        try:
            return int(value, 16) & 0xFFFF_FFFF
        except ValueError:
            return 0
    try:
        return int(value) & 0xFFFF_FFFF
    except (TypeError, ValueError):
        return 0


def _construct_gemma_session(model_path: str) -> Any:
    module = importlib.import_module("sw.runtime.gemma")
    session_cls = getattr(module, "GemmaInferenceSession")
    for kwargs in (
        {"model_path": model_path},
        {"model_dir": model_path},
        {"model": model_path},
    ):
        try:
            return session_cls(**kwargs)
        except TypeError:
            continue
    return session_cls(model_path)


async def _maybe_call_lifecycle(session: Any, model_path: str) -> None:
    for method_name in ("load", "open", "initialize"):
        method = getattr(session, method_name, None)
        if method is None:
            continue
        try:
            result = method()
        except TypeError:
            result = method(model_path)
        if inspect.isawaitable(result):
            await result
