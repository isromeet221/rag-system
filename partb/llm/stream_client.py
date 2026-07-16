"""Stream tokens from LiteLLM (OpenAI API) or direct Ollama (dev fallback)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from partb.logger import time_it, logger

import time

from partb.config import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    OLLAMA_URL,
    OLLAMA_LB_URL,
    OLLAMA_STREAM_PORT,
    USE_LITELLM_FALLBACK,
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
    model = cfg.get("ollama_model", "mistral:7b-instruct")
    prompt = _prompt_from_messages(messages)
    
    # Try Ollama LB first
    lb_failed = False
    tokens_yielded = 0
    try:
        async for ev in _stream_via_ollama_lb(prompt, model, mode, timeout):
            if ev.get("type") == "error":
                logger.warning("[OLLAMA-LB] yielded error: %s", ev.get("message"))
                if tokens_yielded == 0:
                    lb_failed = True
                    break
                else:
                    yield ev
                    return
            if ev.get("type") == "token":
                tokens_yielded += 1
            yield ev
        if not lb_failed:
            return
    except Exception as e:
        logger.warning("[OLLAMA-LB] Failed, falling back to LiteLLM: %s", e)
        
    # Fallback to LiteLLM if enabled
    litellm_failed = False
    tokens_yielded = 0
    if USE_LITELLM_FALLBACK:
        try:
            async for ev in _stream_litellm(messages, mode, cfg, timeout):
                if ev.get("type") == "error":
                    logger.warning("[LITELLM] yielded error: %s", ev.get("message"))
                    if tokens_yielded == 0:
                        litellm_failed = True
                        break
                    else:
                        yield ev
                        return
                if ev.get("type") == "token":
                    tokens_yielded += 1
                yield ev
            if not litellm_failed:
                return
        except Exception as e:
            logger.warning("[LITELLM] Failed, falling back to direct Ollama: %s", e)
    
    # Final fallback to direct Ollama
    async for ev in _stream_ollama(messages, mode, cfg, timeout):
        yield ev


async def _stream_via_ollama_lb(
    prompt: str,
    model: str,
    mode: str,
    timeout: float,
) -> AsyncIterator[dict]:
    lb_url = f"{OLLAMA_LB_URL.rstrip('/')}/load_balancer"
    t0 = time.perf_counter()
    t_first_token = None
    token_count = 0
    char_count = 0
    allocated_server = None
    allocated_model = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 5)) as client:
        # -- 1. Ask load balancer for a GPU server --
        try:
            lb_resp = await client.post(
                lb_url,
                json={"mode": mode},
                timeout=httpx.Timeout(10.0),
            )
            if lb_resp.status_code != 200:
                err_text = lb_resp.text[:500]
                logger.error("[OLLAMA-LB] Load balancer error | status=%s | body=%s", lb_resp.status_code, err_text)
                yield {"type": "error", "message": f"Ollama LB HTTP {lb_resp.status_code}: {err_text}"}
                return
            lb_data = lb_resp.json()
            allocated_server = lb_data.get("ip")
            allocated_model = lb_data.get("model", model)
            if not allocated_server or allocated_server == "No server available":
                logger.error("[OLLAMA-LB] No server available | response=%s", lb_data)
                yield {"type": "error", "message": "No GPU server available via Ollama LB."}
                return
        except httpx.TimeoutException:
            logger.error("[OLLAMA-LB] Load balancer timed out")
            yield {"type": "error", "message": "Ollama LB timed out."}
            return
        except Exception as e:
            logger.exception("[OLLAMA-LB] Load balancer call failed")
            yield {"type": "error", "message": f"Ollama LB error: {e}"}
            return

        logger.info("[OLLAMA-LB] Server allocated | server=%s | model=%s | mode=%s", allocated_server, allocated_model, mode)

        # -- 2. Stream tokens from the allocated GPU server --
        stream_url = f"http://{allocated_server}:{OLLAMA_STREAM_PORT}/ollama"
        body = {"prompt": prompt, "model": allocated_model, "stream": True}

        logger.info("[OLLAMA-LB] Stream request | url=%s | model=%s | prompt_chars=%s", stream_url, allocated_model, len(prompt))
        try:
            async with client.stream("POST", stream_url, json=body) as resp:
                logger.info("[OLLAMA-LB] Response opened | status=%s | server=%s", resp.status_code, allocated_server)
                if resp.status_code != 200:
                    err = await resp.aread()
                    err_text = err.decode(errors='replace')[:1000]
                    logger.error("[OLLAMA-LB] Stream HTTP error | status=%s | body=%s", resp.status_code, err_text)
                    yield {"type": "error", "message": f"Stream HTTP {resp.status_code}: {err_text[:500]}"}
                    _release_ollama_lb_server(allocated_server)
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
                            t_first_token = time.perf_counter()
                            logger.info("[OLLAMA-LB] First token received | cold_startup_time=%.2fs", t_first_token - t0)
                        yield {"type": "token", "content": token}

                t_end = time.perf_counter()
                if t_first_token:
                    logger.info("[OLLAMA-LB] Stream complete | server=%s | model=%s | tokens=%s | chars=%s | cold_startup_time=%.2fs | response_time=%.2fs", allocated_server, allocated_model, token_count, char_count, t_first_token - t0, t_end - t_first_token)
                else:
                    logger.info("[OLLAMA-LB] Stream complete | server=%s | model=%s | tokens=%s | chars=%s | elapsed=%.2fs", allocated_server, allocated_model, token_count, char_count, t_end - t0)

                # -- 3. Release the GPU server back to the pool --
                _release_ollama_lb_server(allocated_server)

        except httpx.TimeoutException:
            logger.error("[OLLAMA-LB] Stream timeout | timeout=%s | server=%s | elapsed=%.2fs", timeout, allocated_server, time.perf_counter() - t0)
            _release_ollama_lb_server(allocated_server)
            yield {"type": "error", "message": f"Stream timeout after {timeout}s on {allocated_server}"}
        except Exception as e:
            logger.exception("[OLLAMA-LB] Stream error | server=%s", allocated_server)
            _release_ollama_lb_server(allocated_server)
            yield {"type": "error", "message": f"Stream error on {allocated_server}: {e}"}


def _release_ollama_lb_server(server_ip: str | None) -> None:
    if not server_ip:
        return
    try:
        resp = httpx.post(
            f"{OLLAMA_LB_URL.rstrip('/')}/release_server",
            json={"url": server_ip},
            timeout=5.0,
        )
        logger.info("[OLLAMA-LB] Server released | server=%s | status=%s", server_ip, resp.status_code)
    except Exception as e:
        logger.warning("[OLLAMA-LB] Release server failed | server=%s | error=%s", server_ip, e)


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
        "model": cfg.get("litellm_model") or cfg.get("ollama_model", "mistral:7b-instruct"),
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
