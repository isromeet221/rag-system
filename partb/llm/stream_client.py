"""Stream tokens from LiteLLM (OpenAI API) or direct Ollama (dev fallback)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from partb.logger import time_it, async_time_it

from partb.config import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    MODE_CONFIG,
    OLLAMA_URL,
    USE_OLLAMA_DIRECT,
)


@time_it
def _prompt_from_messages(messages: list[dict[str, str]]) -> str:
    """Ollama /api/generate expects a single prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "user":
            parts.append(f"User:\n{content}")
        else:
            parts.append(f"Assistant:\n{content}")
    return "\n\n".join(parts)


@async_time_it
async def stream_llm(
    messages: list[dict[str, str]],
    mode: str,
    cfg: dict[str, Any],
) -> AsyncIterator[dict]:
    timeout = cfg.get("llm_timeout_s", 600.0)
    if USE_OLLAMA_DIRECT:
        async for ev in _stream_ollama(messages, mode, cfg, timeout):
            yield ev
        return
    async for ev in _stream_litellm(messages, mode, cfg, timeout):
        yield ev


@async_time_it
async def _stream_litellm(
    messages: list[dict[str, str]],
    mode: str,
    cfg: dict[str, Any],
    timeout: float,
) -> AsyncIterator[dict]:
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

    body = {
        "model": cfg["litellm_model"],
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield {
                        "type": "error",
                        "message": f"LLM HTTP {resp.status_code}: {err.decode(errors='replace')[:500]}",
                    }
                    return
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].lstrip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        yield {"type": "token", "content": content}
        except httpx.TimeoutException:
            yield {"type": "error", "message": f"LLM timeout after {timeout}s"}
        except Exception as e:
            yield {"type": "error", "message": f"LLM stream error: {e}"}


@async_time_it
async def _stream_ollama(
    messages: list[dict[str, str]],
    mode: str,
    cfg: dict[str, Any],
    timeout: float,
) -> AsyncIterator[dict]:
    model = cfg.get("ollama_model") or cfg.get("litellm_model")
    prompt = _prompt_from_messages(messages)
    url = f"{OLLAMA_URL}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": True}

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    yield {"type": "error", "message": f"Ollama HTTP {resp.status_code}"}
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("response") or ""
                    if token:
                        yield {"type": "token", "content": token}
        except httpx.TimeoutException:
            yield {"type": "error", "message": f"Ollama timeout after {timeout}s"}
        except Exception as e:
            yield {"type": "error", "message": f"Ollama stream error: {e}"}