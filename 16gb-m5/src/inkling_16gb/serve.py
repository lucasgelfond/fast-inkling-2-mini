"""OpenAI-compatible localhost server: GET /v1/models, POST /v1/chat/completions (streaming or not).

Thinking goes to `reasoning_content`, answers to `content`, tool calls to `tool_calls`; `tools`, tool results and
`reasoning_effort` (none..max) go through the model's own chat template; image parts (`image_url` with a data: URL,
http(s) URL or file path) go through the vision tower. Prompt prefixes are cached across requests (see engine.py), and
`usage.prompt_tokens_details.cached_tokens` reports how much of the prompt was reused.

Single-threaded on purpose: MLX streams belong to the thread that loaded the model, and a 16 GB Mac has room for
one model and one KV cache anyway, so requests are served one at a time.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from .engine import Engine


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    alias = "Inkling-2-Mini"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [{"id": self.alias, "object": "model", "owned_by": "local"}]})
        if path in ("", "/health"):
            return self._json(200, {"ok": True})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            effort = req.get("reasoning_effort") or (req.get("reasoning") or {}).get("effort") or "none"
            max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 4096)
            temp = float(req.get("temperature") or 0.0)
            top_p = float(req.get("top_p") or 1.0)
            tools = req.get("tools") if req.get("tool_choice") != "none" else None
            t_enc = time.time()
            prompt = self.engine.encode(req.get("messages", []), effort, tools)
            t_enc = time.time() - t_enc
        except Exception as e:
            traceback.print_exc()
            return self._json(400, {"error": {"message": f"{type(e).__name__}: {e}"}})

        print(f"[start] prompt {len(prompt['ids'])} tok, {len(req.get('messages', []))} messages, "
              f"{len(tools or [])} tools, effort={prompt['effort']}", flush=True)
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.alias}
        stream = bool(req.get("stream"))
        t0 = time.time()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})

        parts = {"content": [], "reasoning_content": []}
        calls = []
        info = None
        try:
            for ch, x in self.engine.run(prompt, max_tokens, temp, top_p):
                if ch == "done":
                    info = x
                    break
                if ch == "tool_call":
                    call = {"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                            "function": {"name": x["name"], "arguments": x["arguments"]}}
                    calls.append(call)
                    if stream:
                        send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": len(calls) - 1, **call}]},
                                                   "finish_reason": None}]})
                    continue
                parts[ch].append(x)
                if stream:
                    send({**base, "choices": [{"index": 0, "delta": {ch: x}, "finish_reason": None}]})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:  # e.g. Metal OOM on a huge prompt: report it, drop the cache, keep serving
            traceback.print_exc()
            self.engine.reset_cache()
            if stream:
                try:
                    send({"error": {"message": f"{type(e).__name__}: {e}"}})
                except OSError:
                    pass
                return
            return self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})

        finish = "tool_calls" if calls else info["finish_reason"]
        usage = {"prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
                 "total_tokens": info["prompt_tokens"] + info["completion_tokens"],
                 "prompt_tokens_details": {"cached_tokens": info["cached_tokens"]}}
        tpc = f", {info['tokens_per_check']:.2f} tok/check" if info.get("tokens_per_check") else ""
        print(f"[req] prompt {info['prompt_tokens']} tok ({info['cache']}: {info['cached_tokens']} reused, "
              f"{info['prompt_tokens'] - info['cached_tokens']} new @ {info['prompt_tps']:.0f} tok/s) | "
              f"gen {info['completion_tokens']} tok @ {info['generation_tps']:.1f} tok/s{tpc}"
              f"{' | ' + str(len(calls)) + ' tool call(s)' if calls else ''} | effort={prompt['effort']} | "
              f"{time.time() - t0:.1f}s | encode {t_enc*1e3:.0f} ms, " + ", ".join(f"{k} {v*1e3:.0f} ms" for k, v in info.get("timing", {}).items()), flush=True)
        if stream:
            try:
                send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                if (req.get("stream_options") or {}).get("include_usage", True):
                    send({**base, "choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if calls else "")}
        if parts["reasoning_content"]:
            msg["reasoning_content"] = "".join(parts["reasoning_content"])
        if calls:
            msg["tool_calls"] = calls
        self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": self.alias,
                         "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})


def main(engine_kwargs: dict, port: int = 8090, alias: str = "Inkling-2-Mini-9.6GB") -> None:
    srv = HTTPServer(("127.0.0.1", port), Handler)  # bind first: fail fast if the port is taken
    Handler.engine = Engine(**engine_kwargs)
    Handler.alias = alias
    print(f"serving {alias} on http://127.0.0.1:{port}/v1", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
