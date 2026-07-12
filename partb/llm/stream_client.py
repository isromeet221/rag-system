"""Stream tokens via Ollama Load Balancer (olb.py) → GPU server running Ollama."""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from partb.logger import time_it, async_time_it, logger

from partb.config import OLLAMA_LB_URL, OLLAMA_STREAM_PORT


def _prompt_from_messages(messages: list[dict[str, str]]) -> str:
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
    prompt_chars = len(prompt)
    logger.info("[OLLAMA-LB] Stream start | model=%s | mode=%s | messages=%s | prompt_chars=%s | timeout=%ss", model, mode, len(messages), prompt_chars, timeout)
    async for ev in _stream_via_ollama_lb(prompt, model, mode, timeout):
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

        except httpx.TimeoutException:
            logger.error("[OLLAMA-LB] Stream timeout | timeout=%s | server=%s | elapsed=%.2fs", timeout, allocated_server, time.perf_counter() - t0)
            yield {"type": "error", "message": f"Stream timeout after {timeout}s on {allocated_server}"}
        except Exception as e:
            logger.exception("[OLLAMA-LB] Stream error | server=%s", allocated_server)
            yield {"type": "error", "message": f"Stream error on {allocated_server}: {e}"}
        finally:
            # -- 3. Release the GPU server back to the pool --
            _release_ollama_lb_server(allocated_server)


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
