# KV260 Runtime Daemon

This daemon exposes the PCCX Gemma runtime over HTTP and WebSocket on port
7860. It is intentionally a local board service: it reports health and status,
serves a small browser chat UI, and streams generated token frames as the Gemma
session yields them.

## Install

From the repository root on the KV260 image:

```bash
python3 -m pip install --upgrade pip wheel
python3 -m pip install -r sw/runtime/server/requirements_daemon.txt
```

Install the board runtime dependencies before starting the daemon. The
requirements file includes `torch` only so safetensors-backed model loading can
reuse the Gemma port's loader path; daemon request handling stays in Python and
numpy unless the session implementation selects another backend.

## File Descriptor Limit

The legacy Gemma GUI raised `RLIMIT_NOFILE` before importing the model loader
because large sharded checkpoints can open many files. The daemon keeps that
same guard in `python -m sw.runtime.server` and the systemd template also sets:

```ini
LimitNOFILE=65536
```

## Run

For a direct shell run:

```bash
python3 -m sw.runtime.server \
    --model ~/models/gemma-3n-e4b-int4 \
    --host 0.0.0.0 \
    --port 7860
```

For route and UI checks without loading a model:

```bash
python3 -m sw.runtime.server --no-model --host 0.0.0.0 --port 7860
```

The convenience wrapper uses the same default model path:

```bash
sw/runtime/server/run_daemon.sh
```

The systemd unit at
`sw/runtime/server/service/pccx-gemma-daemon.service` is a template for a future
board install. Copy it into `/etc/systemd/system/`, adjust `User`,
`WorkingDirectory`, and the model path, then enable it explicitly when ready.

## Endpoints

Health:

```bash
curl http://kria:7860/api/health
```

Status:

```bash
curl http://kria:7860/api/status
```

Trace stream:

```bash
curl -N "http://kria:7860/api/trace?session=<uuid>&since=0"
```

Browser chat:

```text
http://kria:7860/
```

WebSocket chat clients connect to:

```text
ws://kria:7860/api/chat
```

Client request frame:

```json
{"type":"user_message","content":"hello","session_id":"<uuid>","temperature":0.7,"top_p":0.95,"max_new_tokens":128}
```

The daemon sends `token` frames followed by one `done` frame. It keeps the last
8 user and assistant turns in memory for each `session_id`. Send this frame to
clear one session:

```json
{"type":"reset","session_id":"<uuid>"}
```

## pccx-launcher

Point pccx-launcher at the board daemon base URL:

```text
http://kria:7860
```

Use `/api/health` for readiness, `/api/status` for telemetry, and
`/api/chat` for WebSocket token streaming. A launcher should treat
`model_loaded:false` as a blocked chat state even when the HTTP service itself
is healthy.
