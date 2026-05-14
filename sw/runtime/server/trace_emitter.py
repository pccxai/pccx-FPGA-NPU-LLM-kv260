"""In-process best-effort trace event fanout."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


TraceEvent = Dict[str, Any]


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    session_id: Optional[str]


class TraceEmitter:
    """Small pub/sub buffer for HTTP trace subscribers.

    The emitter is intentionally lossy. Trace output is useful for diagnostics,
    but it must not stall token generation if a browser tab stops reading.
    """

    def __init__(self, *, max_events: int = 2048, queue_size: int = 512):
        self._events: Deque[TraceEvent] = deque(maxlen=max_events)
        self._subscribers: List[_Subscriber] = []
        self._queue_size = queue_size

    def emit(
        self,
        kind: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> TraceEvent:
        payload = dict(data or {})
        if session_id is not None and "session_id" not in payload:
            payload["session_id"] = session_id
        event: TraceEvent = {
            "ts": float(ts if ts is not None else time.time()),
            "kind": kind,
            "data": payload,
            "_session_id": session_id or payload.get("session_id"),
        }
        self._events.append(event)
        for subscriber in list(self._subscribers):
            if not self._matches_session(event, subscriber.session_id):
                continue
            subscriber.loop.call_soon_threadsafe(self._offer, subscriber, event)
        return event

    def subscribe(
        self,
        *,
        session_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        subscriber = _Subscriber(loop=loop, queue=queue, session_id=session_id)
        self._subscribers.append(subscriber)
        for event in list(self._events):
            if since is not None and event["ts"] <= since:
                continue
            if self._matches_session(event, session_id):
                self._offer(subscriber, event)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers = [
            subscriber
            for subscriber in self._subscribers
            if subscriber.queue is not queue
        ]

    def heartbeat(self, *, session_id: Optional[str] = None) -> TraceEvent:
        data: Dict[str, Any] = {"heartbeat": True}
        if session_id is not None:
            data["session_id"] = session_id
        return {
            "ts": time.time(),
            "kind": "npu_status",
            "data": data,
            "_session_id": session_id,
        }

    @staticmethod
    def to_wire(event: TraceEvent) -> TraceEvent:
        return {
            "ts": event["ts"],
            "kind": event["kind"],
            "data": event.get("data", {}),
        }

    @staticmethod
    def _matches_session(event: TraceEvent, session_id: Optional[str]) -> bool:
        if session_id is None:
            return True
        event_session = event.get("_session_id") or event.get("data", {}).get("session_id")
        return event_session == session_id

    @staticmethod
    def _offer(subscriber: _Subscriber, event: TraceEvent) -> None:
        queue = subscriber.queue
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass
