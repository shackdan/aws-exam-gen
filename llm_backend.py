"""
llm_backend.py
──────────────
Backend-agnostic LLM client used by generator.py and reviewer.py.

Two backends are supported, selected via the LLM_BACKEND env var
(see config.py):
  - "ollama" (default) — talks to a local Ollama server through the
    `ollama` package's native /api/chat endpoint.
  - "mlx"    — talks to an `mlx_lm.server` instance (Apple's
    MLX runtime, Metal-accelerated on Apple Silicon) through its
    OpenAI-compatible /v1/chat/completions endpoint.

Both backends expose the same `.chat(...)` signature so the rest of
the pipeline never branches on which one is active.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple

from config import LLM_BACKEND, MLX_CONFIG, OLLAMA_CONFIG

log = logging.getLogger("aws_exam_gen.llm_backend")


class LLMClient(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str: ...


# ─────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────

class OllamaLLMClient:
    """Adapts ollama.Client to the shared LLMClient interface."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model   = model

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        kwargs: Dict[str, Any] = {}
        if json_mode:
            kwargs["format"] = "json"
        response = self._client.chat(
            model    = self.model,
            messages = messages,
            options  = {
                "temperature": temperature,
                "top_p"      : top_p,
                "num_ctx"    : OLLAMA_CONFIG.num_ctx,
                "num_predict": max_tokens,
            },
            keep_alive= OLLAMA_CONFIG.keep_alive,
            **kwargs,
        )
        return response["message"]["content"]


def _init_ollama_client(model: Optional[str] = None) -> OllamaLLMClient:
    import ollama

    model = model or OLLAMA_CONFIG.model
    log.info(
        f"Initialising Ollama client → model: {model} | "
        f"num_ctx: {OLLAMA_CONFIG.num_ctx} | "
        f"keep_alive: {OLLAMA_CONFIG.keep_alive}"
    )
    try:
        client = ollama.Client(
            host    = OLLAMA_CONFIG.base_url,
            timeout = OLLAMA_CONFIG.timeout,
        )
        client.list()  # warm-up ping
        log.info("Ollama client connected successfully.")
    except Exception as err:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_CONFIG.base_url}. "
            f"Ensure Ollama is running. Error: {err}"
        ) from err
    try:
        client.show(model)  # fail fast if the model isn't pulled
    except Exception as err:
        raise RuntimeError(
            f"Ollama model '{model}' is not available. "
            f"Run: ollama pull {model}. Error: {err}"
        ) from err
    return OllamaLLMClient(client, model)


# ─────────────────────────────────────────────
# MLX-LM backend
# ─────────────────────────────────────────────

class MLXLLMClient:
    """
    Adapts an OpenAI-compatible client pointed at `mlx_lm.server` —
    MLX-LM's built-in server, which speaks the OpenAI
    /v1/chat/completions API — to the shared LLMClient interface.
    """

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model   = model

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        # mlx_lm.server's OpenAI-compatible endpoint does not support
        # response_format-enforced JSON mode, so json_mode is accepted
        # for interface parity with OllamaLLMClient but unused here —
        # the prompts already instruct the model to return JSON only,
        # and utils.extract_json_from_text tolerates surrounding prose.
        response = self._client.chat.completions.create(
            model      = self.model,
            messages   = messages,
            temperature= temperature,
            top_p      = top_p,
            max_tokens = max_tokens,
        )
        return response.choices[0].message.content or ""


def _init_mlx_client(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> MLXLLMClient:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "LLM_BACKEND=mlx requires the 'openai' package (used only "
            "as an HTTP client for mlx_lm.server's OpenAI-compatible "
            "API — no OpenAI account or API key is used). "
            "Run: pip install openai"
        )

    model    = model or MLX_CONFIG.model
    base_url = base_url or MLX_CONFIG.base_url
    log.info(
        f"Initialising MLX-LM client → model: {model} | "
        f"base_url: {base_url}"
    )
    client = OpenAI(
        base_url= base_url,
        api_key = "mlx-lm-local",  # unused by mlx_lm.server, required by the SDK
        timeout = MLX_CONFIG.timeout,
    )
    try:
        client.models.list()  # warm-up ping
        log.info("MLX-LM server connected successfully.")
    except Exception as err:
        raise RuntimeError(
            f"Cannot connect to MLX-LM server at {base_url}. "
            f"Ensure `mlx_lm.server --model {model}` is running. Error: {err}"
        ) from err
    return MLXLLMClient(client, model)


# ─────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────

def init_llm_client(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """
    Initialise and warm-up-ping a client for `model` on the backend
    selected via LLM_BACKEND. model / base_url default to the backend's
    Generator settings; base_url only applies to MLX-LM (Ollama serves
    every model from one server).
    Raises RuntimeError with a descriptive message if unreachable.
    """
    if LLM_BACKEND == "mlx":
        return _init_mlx_client(model=model, base_url=base_url)
    return _init_ollama_client(model=model)


def ping_backend(timeout: int = 5) -> Tuple[bool, str]:
    """
    Lightweight, non-raising reachability check for the configured
    backend — used for startup warnings, not the real connection
    (init_llm_client does that).

    Returns
    ───────
    (reachable, hint)
        hint is a human-readable "how to start it" message, populated
        only when reachable is False.
    """
    if LLM_BACKEND == "mlx":
        try:
            from openai import OpenAI
            OpenAI(
                base_url= MLX_CONFIG.base_url,
                api_key = "mlx-lm-local",
                timeout = timeout,
            ).models.list()
            return True, ""
        except Exception:
            return False, (
                f"MLX-LM server not reachable at {MLX_CONFIG.base_url}. "
                "Start it before running 'generate'.\n"
                f"  Command: mlx_lm.server --model {MLX_CONFIG.model}"
            )

    try:
        import ollama
        ollama.Client(host=OLLAMA_CONFIG.base_url, timeout=timeout).list()
        return True, ""
    except Exception:
        return False, (
            f"Ollama not reachable at {OLLAMA_CONFIG.base_url}. "
            "Start Ollama before running 'generate'.\n"
            "  Command: ollama serve"
        )
