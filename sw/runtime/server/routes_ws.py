"""WebSocket chat route for token streaming."""
from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from aiohttp import WSMsgType, web

from .app import SERVER_STATE_KEY, TRACE_EMITTER_KEY, ServerState
from .trace_emitter import TraceEmitter


def register_ws_routes(app: web.Application) -> None:
    app.router.add_get("/api/chat", chat)


async def chat(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    state: ServerState = request.app[SERVER_STATE_KEY]
    emitter: TraceEmitter = request.app[TRACE_EMITTER_KEY]
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            await _handle_text_frame(ws, state, emitter, msg.data)
        elif msg.type == WSMsgType.ERROR:
            break
    return ws


async def _handle_text_frame(
    ws: web.WebSocketResponse,
    state: ServerState,
    emitter: TraceEmitter,
    raw: str,
) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await _send_error(ws, "invalid JSON", None)
        return
    session_id = str(payload.get("session_id") or uuid.uuid4())
    frame_type = payload.get("type")
    if frame_type == "reset":
        state.reset_session(session_id)
        await ws.send_json(
            {
                "type": "done",
                "finish_reason": "stop",
                "tokens": 0,
                "elapsed_sec": 0.0,
                "tok_per_sec": 0.0,
                "session_id": session_id,
            }
        )
        return
    if frame_type != "user_message":
        await _send_error(ws, "unsupported frame type", session_id)
        return
    content = str(payload.get("content") or "").strip()
    if not content:
        await _send_error(ws, "content is required", session_id)
        return
    if not state.model_loaded:
        await _send_error(ws, "model is not loaded", session_id)
        return
    await _stream_response(ws, state, emitter, payload, session_id, content)


async def _stream_response(
    ws: web.WebSocketResponse,
    state: ServerState,
    emitter: TraceEmitter,
    payload: Dict[str, Any],
    session_id: str,
    content: str,
) -> None:
    history = list(state.ensure_session(session_id))
    messages = history + [{"role": "user", "content": content}]
    temperature = _float(payload.get("temperature"), 0.7)
    top_p = _float(payload.get("top_p"), 0.95)
    max_new_tokens = max(1, min(_int(payload.get("max_new_tokens"), 128), 4096))
    started = time.monotonic()
    chunks: List[str] = []
    tokens = 0
    finish_reason = "stop"
    try:
        async for item in _generate_items(
            state.session,
            content=content,
            messages=messages,
            history=history,
            session_id=session_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            trace_emitter=emitter,
        ):
            token_text, item_finish = _token_text(item)
            if item_finish:
                finish_reason = item_finish
            if token_text is None:
                continue
            if tokens >= max_new_tokens:
                finish_reason = "length"
                break
            token_idx = tokens
            tokens += 1
            chunks.append(token_text)
            state.tokens_total += 1
            emitter.emit(
                "token",
                {"content": token_text, "token_idx": token_idx},
                session_id=session_id,
            )
            await ws.send_json(
                {
                    "type": "token",
                    "content": token_text,
                    "session_id": session_id,
                    "token_idx": token_idx,
                }
            )
    except Exception as exc:
        await _send_error(ws, f"{type(exc).__name__}: {exc}", session_id)
        return
    elapsed = max(time.monotonic() - started, 1e-9)
    tok_per_sec = tokens / elapsed
    state.tokens_per_sec_last = tok_per_sec
    state.append_turn(session_id, content, "".join(chunks))
    await ws.send_json(
        {
            "type": "done",
            "finish_reason": finish_reason,
            "tokens": tokens,
            "elapsed_sec": round(elapsed, 3),
            "tok_per_sec": round(tok_per_sec, 3),
            "session_id": session_id,
        }
    )


async def _generate_items(
    session: Any,
    *,
    content: str,
    messages: List[Dict[str, str]],
    history: List[Dict[str, str]],
    session_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    trace_emitter: TraceEmitter,
) -> AsyncIterator[Any]:
    result = _invoke_generate(
        session,
        content=content,
        messages=messages,
        history=history,
        session_id=session_id,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        trace_emitter=trace_emitter,
    )
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, str):
        yield result
        return
    if hasattr(result, "__aiter__"):
        async for item in result:
            yield item
        return
    iterator = iter(result)
    while True:
        item = await asyncio.to_thread(_next_or_none, iterator)
        if item is _END:
            break
        yield item


def _invoke_generate(session: Any, **values: Any) -> Any:
    method = getattr(session, "generate")
    aliases = {
        "content": values["content"],
        "prompt": values["content"],
        "text": values["content"],
        "user_message": values["content"],
        "messages": values["messages"],
        "history": values["history"],
        "turns": values["history"],
        "session_id": values["session_id"],
        "temperature": values["temperature"],
        "top_p": values["top_p"],
        "max_new_tokens": values["max_new_tokens"],
        "max_tokens": values["max_new_tokens"],
        "trace_emitter": values["trace_emitter"],
        "trace": values["trace_emitter"],
    }
    canonical = {
        "content": values["content"],
        "messages": values["messages"],
        "history": values["history"],
        "session_id": values["session_id"],
        "temperature": values["temperature"],
        "top_p": values["top_p"],
        "max_new_tokens": values["max_new_tokens"],
        "trace_emitter": values["trace_emitter"],
    }
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(
            values["content"],
            temperature=values["temperature"],
            top_p=values["top_p"],
            max_new_tokens=values["max_new_tokens"],
        )
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        kwargs = dict(canonical)
        for name in params:
            if name in aliases:
                kwargs[name] = aliases[name]
        return method(**kwargs)
    kwargs = {name: aliases[name] for name in params if name in aliases}
    has_prompt = any(
        name in kwargs
        for name in ("content", "prompt", "text", "user_message", "messages")
    )
    if has_prompt:
        return method(**kwargs)
    return method(values["content"], **kwargs)


def _token_text(item: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(item, str):
        return item, None
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace"), None
    if isinstance(item, dict):
        finish = item.get("finish_reason")
        for key in ("content", "token", "text", "piece"):
            if key in item and item[key] is not None:
                return str(item[key]), finish
        return None, finish
    return str(item), None


async def _send_error(
    ws: web.WebSocketResponse,
    message: str,
    session_id: Optional[str],
) -> None:
    payload = {"type": "error", "message": message, "session_id": session_id or ""}
    await ws.send_json(payload)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_END = object()


def _next_or_none(iterator: Iterable[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _END
