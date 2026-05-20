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

from .telemetry import DmesgWatcher, TelemetrySink, collect_system_metrics
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
    backend_requested: str = "auto"
    backend_reason: str = ""
    histories: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    session_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tokens_total: int = 0
    tokens_per_sec_last: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    npu_busy_started_at: Optional[float] = None
    npu_busy_total_sec: float = 0.0
    npu_last_mmio_stat_hex: str = "0x00000000"

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
        if value in {"cpu", "npu", "npu_uca", "hybrid"}:
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

    def record_inference(
        self,
        *,
        session_id: str,
        prompt_length_chars: int,
        max_new_tokens: int,
        tokens: int,
        elapsed_sec: float,
        finish_reason: str,
    ) -> Dict[str, Any]:
        latency_ms = elapsed_sec * 1000.0
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 1024:
            del self.latencies_ms[:-1024]
        tok_per_sec = tokens / max(elapsed_sec, 1e-9)
        metrics = {
            "prompt_length_chars": prompt_length_chars,
            "max_new_tokens": max_new_tokens,
            "tokens": tokens,
            "elapsed_sec": round(elapsed_sec, 3),
            "tokens_per_sec": round(tok_per_sec, 3),
            "latency_ms": round(latency_ms, 3),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "finish_reason": finish_reason,
        }
        self.session_metrics[session_id] = metrics
        return metrics

    @property
    def latency_p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 0.50)

    @property
    def latency_p99_ms(self) -> float:
        return _percentile(self.latencies_ms, 0.99)

    def annotate_npu_sample(self, npu: Dict[str, Any]) -> Dict[str, Any]:
        now = time.monotonic()
        busy = bool(npu["npu_busy"])
        if busy and self.npu_busy_started_at is None:
            self.npu_busy_started_at = now
        elif not busy and self.npu_busy_started_at is not None:
            self.npu_busy_total_sec += now - self.npu_busy_started_at
            self.npu_busy_started_at = None

        busy_total_sec = self.npu_busy_total_sec
        if self.npu_busy_started_at is not None:
            busy_total_sec += now - self.npu_busy_started_at

        sample = dict(npu)
        sample["npu_busy_duration_ms"] = round(busy_total_sec * 1000.0, 3)
        sample["npu_mmio_stat_changed"] = (
            sample["npu_mmio_stat_hex"] != self.npu_last_mmio_stat_hex
        )
        self.npu_last_mmio_stat_hex = sample["npu_mmio_stat_hex"]
        return sample

    async def load_model(
        self,
        model_path: str,
        trace_emitter: Optional[TraceEmitter] = None,
    ) -> None:
        self.model_path = model_path
        self.loading = True
        self.load_error = None
        self.load_started_at = time.monotonic()
        try:
            session = await asyncio.to_thread(
                _construct_gemma_session,
                model_path,
                self.backend_requested,
            )
            await _maybe_call_lifecycle(session, model_path)
            self.session = session
            self.backend_reason = str(getattr(session, "backend_reason", ""))
        except Exception as exc:  # pragma: no cover - exercised on board images.
            self.session = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.backend_reason = str(exc)
            if trace_emitter is not None:
                trace_emitter.emit("model_load_error", {"error_type": type(exc).__name__})
        finally:
            started = self.load_started_at or time.monotonic()
            self.elapsed_load_sec = time.monotonic() - started
            self.load_started_at = None
            self.loading = False
            if self.session is not None and trace_emitter is not None:
                trace_emitter.emit(
                    "model_load_done",
                    {
                        "elapsed_load_sec": round(self.elapsed_load_sec, 3),
                        "backend": self.backend,
                    },
                )

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
TELEMETRY_SINK_KEY = web.AppKey("telemetry_sink", TelemetrySink)
MODEL_LOAD_TASK_KEY = web.AppKey("model_load_task", asyncio.Task)
TELEMETRY_TASK_KEY = web.AppKey("telemetry_task", asyncio.Task)
DMESG_WATCHER_KEY = web.AppKey("dmesg_watcher", DmesgWatcher)
TELEMETRY_INTERVAL_KEY = web.AppKey("telemetry_interval_sec", float)


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
    backend: str = "auto",
    telemetry_dir: Optional[Path | str] = None,
    telemetry_retention_days: int = 30,
    telemetry_interval_sec: float = 30.0,
) -> web.Application:
    app = web.Application(middlewares=[_cors_middleware])
    telemetry_sink = TelemetrySink.from_env(
        directory=telemetry_dir,
        retention_days=telemetry_retention_days,
    )
    app[SERVER_STATE_KEY] = ServerState(
        session=session,
        host=host,
        port=port,
        model_path=os.path.expanduser(model_path) if model_path else None,
        backend_requested=backend,
    )
    app[TELEMETRY_SINK_KEY] = telemetry_sink
    app[TRACE_EMITTER_KEY] = TraceEmitter(telemetry_sink=telemetry_sink)
    app[DMESG_WATCHER_KEY] = DmesgWatcher()
    app[TELEMETRY_INTERVAL_KEY] = telemetry_interval_sec

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
    emitter: TraceEmitter = app[TRACE_EMITTER_KEY]
    sink: TelemetrySink = app[TELEMETRY_SINK_KEY]
    emitter.emit(
        "daemon_start",
        {
            "version": VERSION,
            "telemetry_dir": "~/.local/state/pccx-kv260/telemetry",
            "telemetry_file": sink.path.name,
            "telemetry_retention_days": sink.retention_days,
        },
    )
    if state.session is None and state.model_path:
        emitter.emit("model_load_start", {"model_dir_set": True})
        app[MODEL_LOAD_TASK_KEY] = asyncio.create_task(
            state.load_model(state.model_path, emitter)
        )
    interval = app[TELEMETRY_INTERVAL_KEY]
    if interval > 0:
        app[TELEMETRY_TASK_KEY] = asyncio.create_task(_telemetry_loop(app, interval))


async def _on_cleanup(app: web.Application) -> None:
    telemetry_task = app.get(TELEMETRY_TASK_KEY)
    if telemetry_task is not None and not telemetry_task.done():
        telemetry_task.cancel()
        try:
            await telemetry_task
        except asyncio.CancelledError:
            pass
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
    npu = state.annotate_npu_sample(read_npu_status())
    system = collect_system_metrics()
    total_ram_mb = system["ram_total_mb"] or ram_mb()
    return {
        "model_id": state.model_id,
        "model_loaded": state.model_loaded,
        "session_count": len(state.histories),
        "tokens_total": state.tokens_total,
        "tokens_per_sec_last": round(state.tokens_per_sec_last, 3),
        "latency_p50_ms": state.latency_p50_ms,
        "latency_p99_ms": state.latency_p99_ms,
        "npu_mmio_stat_hex": npu["npu_mmio_stat_hex"],
        "npu_busy": npu["npu_busy"],
        "npu_busy_duration_ms": npu["npu_busy_duration_ms"],
        "npu_done": npu["npu_done"],
        "npu_available": npu["npu_available"],
        "npu_axil_command_count": npu["npu_axil_command_count"],
        "npu_token_valid": npu["npu_token_valid"],
        "npu_last_token": npu["npu_last_token"],
        "npu_readback_bytes_last": npu["npu_readback_bytes_last"],
        "npu_mmio_stat_changed": npu["npu_mmio_stat_changed"],
        "backend": state.backend,
        "backend_requested": state.backend_requested,
        "backend_reason": state.backend_reason,
        "uptime_sec": int(time.monotonic() - state.started_at),
        "bitstream_sha": bitstream_sha256(),
        "host": socket.gethostname(),
        "ram_mb": total_ram_mb,
        "ram_used_mb": system["ram_used_mb"],
        "cpu_load": system["cpu_load"],
        "temperature_c": system["temperature_c"],
    }


def read_npu_status() -> Dict[str, Any]:
    fallback = {
        "npu_mmio_stat_hex": "0x00000000",
        "npu_busy": False,
        "npu_done": False,
        "npu_available": False,
        "npu_axil_command_count": 0,
        "npu_last_cycle_count": 0,
        "npu_token_valid": False,
        "npu_last_token": None,
        "npu_readback_bytes_last": 0,
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
            "npu_axil_command_count": 0,
            "npu_last_cycle_count": 0,
            "npu_token_valid": False,
            "npu_last_token": None,
            "npu_readback_bytes_last": 0,
        }
    if isinstance(raw, dict):
        stat_value = raw.get("npu_mmio_stat_hex", raw.get("mmio_stat_hex", raw.get("mmio_hex")))
        if stat_value is None:
            stat_value = raw.get("npu_mmio_stat", raw.get("mmio_stat", 0))
        stat = _parse_stat_value(stat_value)
        return {
            "npu_mmio_stat_hex": f"0x{stat:08x}",
            "npu_busy": bool(raw.get("npu_busy", raw.get("busy", stat & 0x1))),
            "npu_done": bool(raw.get("npu_done", raw.get("done", stat & 0x2))),
            "npu_available": bool(raw.get("npu_available", raw.get("available", True))),
            "npu_axil_command_count": _parse_int_value(
                raw.get(
                    "npu_axil_command_count",
                    raw.get("axil_command_count", raw.get("command_count", 0)),
                )
            ),
            "npu_last_cycle_count": _parse_int_value(
                raw.get("npu_last_cycle_count", raw.get("last_cycle_count", 0))
            ),
            "npu_token_valid": bool(raw.get("npu_token_valid", raw.get("token_valid", False))),
            "npu_last_token": _parse_optional_int_value(
                raw.get("npu_last_token", raw.get("last_token"))
            ),
            "npu_readback_bytes_last": _parse_int_value(
                raw.get("npu_readback_bytes_last", raw.get("readback_bytes_last", 0))
            ),
        }
    return fallback


async def _telemetry_loop(app: web.Application, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        state: ServerState = app[SERVER_STATE_KEY]
        emitter: TraceEmitter = app[TRACE_EMITTER_KEY]
        emitter.emit("system_sample", status_payload(state))
        watcher: DmesgWatcher = app[DMESG_WATCHER_KEY]
        for event in await asyncio.to_thread(watcher.collect):
            emitter.emit(event["kind"], event["data"])


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


def _parse_int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_optional_int_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)


def _construct_gemma_session(model_path: str, backend: str = "auto") -> Any:
    module = importlib.import_module("sw.runtime.gemma")
    session_cls = getattr(module, "GemmaInferenceSession")
    for kwargs in (
        {"model_path": model_path, "backend": backend},
        {"model_dir": model_path, "backend": backend},
        {"model": model_path, "backend": backend},
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
