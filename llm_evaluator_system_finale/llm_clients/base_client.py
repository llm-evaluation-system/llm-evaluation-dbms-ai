"""
llm_clients/base_client.py — Abstract base class for all LLM provider clients.

Each concrete client (OpenAI, Anthropic, Google, Groq) inherits from this base
and implements the `complete` method.  The base class handles retry logic,
latency tracking, cost accounting, and seed injection where supported.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from config import (
    API_MAX_RETRIES,
    API_RETRY_BACKOFF_BASE,
    API_TIMEOUT_SECONDS,
)


# ── Response Data Container ───────────────────────────────────────────────────
@dataclass
class LLMResponse:
    """Normalised response returned by every LLM client implementation."""
    model_id: str
    answer_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    ttft_ms: float = 0.0          # Time-to-first-token (ms)
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    retry_count: int = 0
    finish_reason: str = "stop"
    raw_response: Optional[dict] = field(default=None, repr=False)


# ── Abstract Base ─────────────────────────────────────────────────────────────
class BaseLLMClient(ABC):
    """
    Abstract base for all LLM provider clients.

    Concrete subclasses must implement:
        _call_api(messages, system_prompt, hyperparams) → LLMResponse

    The `complete` method wraps _call_api with retry logic and latency
    measurement.
    """

    def __init__(self, model_config: dict) -> None:
        self.model_id: str = model_config["model_id"] if "model_id" in model_config else list(model_config.keys())[0]
        self.api_model: str = model_config.get("api_model", "")
        self.cost_per_1k_input: float = model_config.get("cost_per_1k_input_tokens", 0.0)
        self.cost_per_1k_output: float = model_config.get("cost_per_1k_output_tokens", 0.0)
        self.supports_seed: bool = model_config.get("supports_seed", False)
        self._http_client: Optional[httpx.AsyncClient] = None

    # ── Public interface ──────────────────────────────────────────────────────
    async def complete(
        self,
        prompt: str,
        system_prompt: str,
        hyperparams: dict,
    ) -> LLMResponse:
        """
        Execute LLM completion with automatic retry and latency tracking.
        Returns an LLMResponse regardless of whether the request succeeded or
        raised after exhausting retries.
        """
        messages = [{"role": "user", "content": prompt}]
        retry_count = 0
        last_error: Optional[Exception] = None

        for attempt in range(API_MAX_RETRIES + 1):
            if attempt > 0:
                backoff = API_RETRY_BACKOFF_BASE ** attempt
                await asyncio.sleep(backoff)
                retry_count = attempt

            start_wall = time.perf_counter()
            try:
                response = await self._call_api(messages, system_prompt, hyperparams)
                response.retry_count = retry_count
                return response
            except Exception as exc:
                last_error = exc
                if attempt == API_MAX_RETRIES:
                    raise RuntimeError(
                        f"[{self.model_id}] API call failed after {API_MAX_RETRIES} retries: {exc}"
                    ) from exc

        raise RuntimeError(f"[{self.model_id}] Unreachable code")

    # ── To be implemented by each provider ───────────────────────────────────
    @abstractmethod
    async def _call_api(
        self,
        messages: list[dict],
        system_prompt: str,
        hyperparams: dict,
    ) -> LLMResponse:
        raise NotImplementedError

    # ── Cost helper ───────────────────────────────────────────────────────────
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000) * self.cost_per_1k_input
            + (output_tokens / 1000) * self.cost_per_1k_output
        )

    # ── HTTP client lifecycle ─────────────────────────────────────────────────
    async def __aenter__(self) -> "BaseLLMClient":
        self._http_client = httpx.AsyncClient(timeout=API_TIMEOUT_SECONDS)
        return self

    async def __aexit__(self, *_) -> None:
        if self._http_client:
            await self._http_client.aclose()

    @staticmethod
    def hyperparam_hash(hyperparams: dict) -> str:
        """Return a stable SHA-256 hash of the hyperparameter dict."""
        canonical = json.dumps(hyperparams, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
