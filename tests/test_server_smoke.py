from __future__ import annotations

import asyncio
import json
import unittest

from aiohttp.test_utils import TestClient, TestServer

from sw.runtime.server import create_app
from sw.runtime.server.app import TRACE_EMITTER_KEY


class MockGemmaInferenceSession:
    model_id = "google/gemma-3n-E4B-it"
    backend = "cpu"

    def generate(
        self,
        content,
        history,
        session_id,
        temperature,
        top_p,
        max_new_tokens,
        trace_emitter,
    ):
        assert content == "hello"
        assert session_id == "session-a"
        assert temperature == 0.7
        assert top_p == 0.95
        assert max_new_tokens == 8
        assert history == []
        trace_emitter.emit("dispatch", {"phase": "mock"}, session_id=session_id)
        yield "hi"
        yield {"content": " there"}


class ServerSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = create_app(MockGemmaInferenceSession(), host="127.0.0.1", port=0)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_health_and_status_shapes(self):
        health = await self.client.get("/api/health")
        self.assertEqual(health.status, 200)
        health_payload = await health.json()
        self.assertEqual(health_payload["status"], "ready")
        self.assertTrue(health_payload["model_loaded"])
        self.assertEqual(health_payload["version"], "v002.1-v12d")

        status = await self.client.get("/api/status")
        self.assertEqual(status.status, 200)
        status_payload = await status.json()
        self.assertEqual(status_payload["model_id"], "google/gemma-3n-E4B-it")
        self.assertEqual(status_payload["backend"], "cpu")
        self.assertEqual(status_payload["npu_mmio_stat_hex"], "0x00000000")
        self.assertFalse(status_payload["npu_available"])
        self.assertEqual(len(status_payload["bitstream_sha"]), 64)
        self.assertIn("host", status_payload)
        self.assertIn("ram_mb", status_payload)

    async def test_websocket_streams_token_and_done_frames(self):
        ws = await self.client.ws_connect("/api/chat")
        await ws.send_json(
            {
                "type": "user_message",
                "content": "hello",
                "session_id": "session-a",
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 8,
            }
        )
        first = await ws.receive_json(timeout=1)
        second = await ws.receive_json(timeout=1)
        done = await ws.receive_json(timeout=1)
        self.assertEqual(first, {
            "type": "token",
            "content": "hi",
            "session_id": "session-a",
            "token_idx": 0,
        })
        self.assertEqual(second["type"], "token")
        self.assertEqual(second["content"], " there")
        self.assertEqual(second["token_idx"], 1)
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["finish_reason"], "stop")
        self.assertEqual(done["tokens"], 2)
        await ws.close()

        status = await self.client.get("/api/status")
        payload = await status.json()
        self.assertEqual(payload["session_count"], 1)
        self.assertEqual(payload["tokens_total"], 2)
        self.assertGreater(payload["tokens_per_sec_last"], 0)

    async def test_trace_stream_returns_ndjson_events(self):
        response = await self.client.get("/api/trace?session=session-a&since=0")
        emitter = self.client.server.app[TRACE_EMITTER_KEY]
        emitter.emit("layer", {"idx": 1}, session_id="session-a")
        line = await asyncio.wait_for(response.content.readline(), timeout=1)
        response.close()
        payload = json.loads(line.decode("utf-8"))
        self.assertEqual(payload["kind"], "layer")
        self.assertEqual(payload["data"]["idx"], 1)
        self.assertEqual(payload["data"]["session_id"], "session-a")

    async def test_no_model_mode_reports_unloaded_and_chat_error(self):
        app = create_app(None, host="127.0.0.1", port=0)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            health = await client.get("/api/health")
            self.assertEqual(health.status, 200)
            payload = await health.json()
            self.assertEqual(payload["status"], "ready")
            self.assertFalse(payload["model_loaded"])

            ws = await client.ws_connect("/api/chat")
            await ws.send_json(
                {
                    "type": "user_message",
                    "content": "hello",
                    "session_id": "empty",
                }
            )
            frame = await ws.receive_json(timeout=1)
            self.assertEqual(frame["type"], "error")
            self.assertEqual(frame["session_id"], "empty")
            await ws.close()
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
