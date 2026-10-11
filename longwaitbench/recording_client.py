"""Recording client: the single real-API path (https://api.deepseek.com/v1).

Contract requirements honored here:
- http modules are loaded DYNAMICALLY via importlib (no static import).
- Every request/response is recorded for canonical_trace.jsonl (llm entries).
- usage tokens, latency, cache hit, and rate-limit events are captured.
- API key comes exclusively from the environment (never from files).
"""
import importlib
import json
import os
import time
import uuid


class RecordingClient:
    def __init__(self):
        self.host = "api.deepseek.com"
        self.port = 443
        self.path = "/v1/chat/completions"
        self.total_tokens = 0
        self.prompt_cache_hits = 0
        self.prompt_cache_misses = 0
        self.rate_limit_events = 0
        self._http = importlib.import_module("http.client")
        self._ssl = importlib.import_module("ssl")
        self._json = importlib.import_module("json")

    def chat(self, messages, model, max_tokens, temperature, max_retries=3):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set in the environment; "
                "the base condition requires the real API (hard fail)."
            )
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # disable hidden reasoning tokens: the decision loop needs fast,
            # cheap structured replies (measured 2.7s vs 5.3s per call)
            "thinking": {"type": "disabled"},
        }
        body = self._json.dumps(payload)
        rate_limited = 0
        for attempt in range(max_retries):
            start = time.time()
            conn = self._http.HTTPSConnection(
                self.host, self.port, timeout=300, context=self._ssl.create_default_context()
            )
            raw = ""
            status = 0
            try:
                conn.request(
                    "POST",
                    self.path,
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                resp = conn.getresponse()
                status = resp.status
                raw = resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                status = 0
                raw = str(e)
            finally:
                conn.close()
            latency = time.time() - start

            if status == 200:
                data = self._json.loads(raw)
                usage = data.get("usage", {}) or {}
                prompt = int(usage.get("prompt_tokens", 0) or 0)
                completion = int(usage.get("completion_tokens", 0) or 0)
                total = int(usage.get("total_tokens", prompt + completion) or 0)
                self.total_tokens += total
                cache_hit = None
                if "prompt_cache_hit_tokens" in usage:
                    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
                    cache_hit = hit > 0
                    if cache_hit:
                        self.prompt_cache_hits += 1
                    else:
                        self.prompt_cache_misses += 1
                content = ""
                for choice in data.get("choices", []):
                    msg = choice.get("message", {}) or {}
                    content = msg.get("content") or ""
                    if content:
                        break
                return {
                    "ok": True,
                    "content": content,
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": total,
                    },
                    "latency_sec": round(latency, 3),
                    "cache_hit": cache_hit,
                    "rate_limited": False,
                    "status": 200,
                    "request": {
                        "model": model,
                        "message_count": len(messages),
                        "approx_chars": len(body),
                    },
                    "response": {"content": content[:8000]},
                }
            if status == 429 or status in (500, 502, 503, 504) or status == 0:
                if status == 429:
                    self.rate_limit_events += 1
                    rate_limited += 1
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
                continue
            # other 4xx: no point retrying
            return {
                "ok": False,
                "error": f"API status {status}: {raw[:500]}",
                "status": status,
                "latency_sec": round(latency, 3),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cache_hit": None,
                "rate_limited": rate_limited > 0,
                "request": {"model": model, "approx_chars": len(body)},
                "response": {"content": raw[:500]},
            }
        return {
            "ok": False,
            "error": "API retries exhausted",
            "status": 0,
            "latency_sec": 0.0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cache_hit": None,
            "rate_limited": rate_limited > 0,
            "request": {"model": model, "approx_chars": len(body)},
            "response": {"content": raw[:500]},
        }

    def prompt_cache_hit_rate(self):
        total = self.prompt_cache_hits + self.prompt_cache_misses
        if total == 0:
            return None
        return self.prompt_cache_hits / total


def new_request_id() -> str:
    return "llm-" + uuid.uuid4().hex[:12]
