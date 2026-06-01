"""Stream tokens from LiteLLM (OpenAI API) or direct Ollama (dev fallback)."""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from partb.logger import time_it, async_time_it, logger

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


async def stream_llm(
    messages: list[dict[str, str]],
    mode: str,
    cfg: dict[str, Any],
) -> AsyncIterator[dict]:
    timeout = cfg.get("llm_timeout_s", 600.0)
    provider = "ollama" if USE_OLLAMA_DIRECT else "litellm"
    model = cfg.get("ollama_model") if USE_OLLAMA_DIRECT else cfg.get("litellm_model")
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    logger.info("[LLM] Stream start | provider=%s | model=%s | mode=%s | messages=%s | prompt_chars=%s | timeout=%ss", provider, model, mode, len(messages), prompt_chars, timeout)
    if USE_OLLAMA_DIRECT:
        async for ev in _stream_ollama(messages, mode, cfg, timeout):
            yield ev
        return
    async for ev in _stream_litellm(messages, mode, cfg, timeout):
        yield ev


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

    t0 = time.perf_counter()
    token_count = 0
    char_count = 0
    logger.info("[LiteLLM] Request | url=%s | model=%s | mode=%s", url, body["model"], mode)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                logger.info("[LiteLLM] Response opened | status=%s | model=%s", resp.status_code, body["model"])
                if resp.status_code != 200:
                    err = await resp.aread()
                    err_text = err.decode(errors='replace')[:1000]
                    logger.error("[LiteLLM] HTTP error | status=%s | body=%s", resp.status_code, err_text)
                    yield {
                        "type": "error",
                        "message": f"LLM HTTP {resp.status_code}: {err_text[:500]}",
                    }
                    return
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].lstrip()
                    if data == "[DONE]":
                        logger.info("[LiteLLM] Stream done marker received | tokens=%s | chars=%s | elapsed=%.2fs", token_count, char_count, time.perf_counter() - t0)
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
                        token_count += 1
                        char_count += len(content)
                        if token_count == 1:
                            logger.info("[LiteLLM] First token received | elapsed=%.2fs", time.perf_counter() - t0)
                        yield {"type": "token", "content": content}
                logger.info("[LiteLLM] Stream complete | model=%s | tokens=%s | chars=%s | elapsed=%.2fs", body["model"], token_count, char_count, time.perf_counter() - t0)
        except httpx.TimeoutException:
            logger.error("[LiteLLM] Timeout | timeout=%s | elapsed=%.2fs", timeout, time.perf_counter() - t0)
            yield {"type": "error", "message": f"LLM timeout after {timeout}s"}
        except Exception as e:
            logger.exception("[LiteLLM] Stream error")
            yield {"type": "error", "message": f"LLM stream error: {e}"}


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

    t0 = time.perf_counter()
    token_count = 0
    char_count = 0
    logger.info("[Ollama] Request | url=%s | model=%s | mode=%s", url, model, mode)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            async with client.stream("POST", url, json=body) as resp:
                logger.info("[Ollama] Response opened | status=%s | model=%s", resp.status_code, model)
                if resp.status_code != 200:
                    err = await resp.aread()
                    logger.error("[Ollama] HTTP error | status=%s | body=%s", resp.status_code, err.decode(errors='replace')[:1000])
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
                        token_count += 1
                        char_count += len(token)
                        if token_count == 1:
                            logger.info("[Ollama] First token received | elapsed=%.2fs", time.perf_counter() - t0)
                        yield {"type": "token", "content": token}
                logger.info("[Ollama] Stream complete | model=%s | tokens=%s | chars=%s | elapsed=%.2fs", model, token_count, char_count, time.perf_counter() - t0)
        except httpx.TimeoutException:
            logger.error("[Ollama] Timeout | timeout=%s | elapsed=%.2fs", timeout, time.perf_counter() - t0)
            yield {"type": "error", "message": f"Ollama timeout after {timeout}s"}
        except Exception as e:
            logger.exception("[Ollama] Stream error")
            yield {"type": "error", "message": f"Ollama stream error: {e}"}