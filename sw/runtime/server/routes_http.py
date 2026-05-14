"""HTTP routes for the runtime daemon."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from aiohttp import web

from .app import (
    SERVER_STATE_KEY,
    TRACE_EMITTER_KEY,
    ServerState,
    health_payload,
    status_payload,
)
from .trace_emitter import TraceEmitter


def register_http_routes(app: web.Application) -> None:
    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/trace", trace)
    app.router.add_route("OPTIONS", "/{tail:.*}", options)


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(Path(__file__).with_name("static") / "chat.html")


async def health(request: web.Request) -> web.Response:
    state: ServerState = request.app[SERVER_STATE_KEY]
    payload = health_payload(state)
    http_status = 503 if payload["status"] in {"loading", "error"} else 200
    return web.json_response(payload, status=http_status)


async def status(request: web.Request) -> web.Response:
    state: ServerState = request.app[SERVER_STATE_KEY]
    return web.json_response(status_payload(state))


async def trace(request: web.Request) -> web.StreamResponse:
    emitter: TraceEmitter = request.app[TRACE_EMITTER_KEY]
    session_id = request.query.get("session") or None
    since = _parse_since(request.query.get("since"))
    queue = emitter.subscribe(session_id=session_id, since=since)
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                event = emitter.heartbeat(session_id=session_id)
            line = json.dumps(
                emitter.to_wire(event),
                separators=(",", ":"),
                sort_keys=True,
            )
            await response.write(f"{line}\n".encode("utf-8"))
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        emitter.unsubscribe(queue)
    return response


async def options(_request: web.Request) -> web.Response:
    return web.Response(status=204)


def _parse_since(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
