"""
llm_clients/providers.py — Concrete LLM client implementations for all four
challenger providers: OpenAI, Anthropic, Google Gemini, and Groq.

Each client maps the normalised Hyperparams dict to the provider-specific
API payload, handling differences in parameter names, optional fields,
and streaming behaviour transparently.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import httpx

from config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY,
)
from .base_client import BaseLLMClient, LLMResponse


# ── OpenAI (GPT-4o) ───────────────────────────────────────────────────────────
class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI's Chat Completions API.
    Supports: temperature, top_p, max_tokens, presence_penalty,
              frequency_penalty, seed (GPT-4 turbo+).
    """

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)
        self.api_key = OPENAI_API_KEY

    async def _call_api(
        self,
        messages: list[dict],
        system_prompt: str,
        hyperparams: dict,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        payload: dict = {
            "model": self.api_model,
            "messages": full_messages,
            "temperature": hyperparams.get("temperature", 0.3),
            "top_p": hyperparams.get("top_p", 0.9),
            "max_tokens": hyperparams.get("max_tokens", 1024),
            "presence_penalty": hyperparams.get("presence_penalty", 0.0),
            "frequency_penalty": hyperparams.get("frequency_penalty", 0.0),
        }

        if self.supports_seed and hyperparams.get("seed") is not None:
            payload["seed"] = hyperparams["seed"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(self.BASE_URL, json=payload, headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        finish_reason = data["choices"][0].get("finish_reason", "stop")

        tps = (output_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0.0

        return LLMResponse(
            model_id=self.model_id,
            answer_text=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=self._calculate_cost(input_tokens, output_tokens),
            ttft_ms=latency_ms * 0.15,   # Approximate TTFT as 15% of total latency
            total_latency_ms=latency_ms,
            tokens_per_second=tps,
            finish_reason=finish_reason,
            raw_response=data,
        )


# ── Anthropic (Claude 3.5 Sonnet) ─────────────────────────────────────────────
class AnthropicClient(BaseLLMClient):
    """
    Client for Anthropic's Messages API.
    Supports: temperature, top_p, top_k, max_tokens.
    Note: presence_penalty and frequency_penalty are not supported by Anthropic.
    """

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)
        self.api_key = ANTHROPIC_API_KEY

    async def _call_api(
        self,
        messages: list[dict],
        system_prompt: str,
        hyperparams: dict,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        payload: dict = {
            "model": self.api_model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": hyperparams.get("max_tokens", 1024),
            "temperature": hyperparams.get("temperature", 0.3),
            "top_p": hyperparams.get("top_p", 0.9),
        }

        # top_k is supported by Anthropic
        top_k = hyperparams.get("top_k", -1)
        if top_k != -1 and top_k is not None:
            payload["top_k"] = int(top_k)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(self.BASE_URL, json=payload, headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

        answer = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        finish_reason = data.get("stop_reason", "end_turn")

        tps = (output_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0.0

        return LLMResponse(
            model_id=self.model_id,
            answer_text=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=self._calculate_cost(input_tokens, output_tokens),
            ttft_ms=latency_ms * 0.12,
            total_latency_ms=latency_ms,
            tokens_per_second=tps,
            finish_reason=finish_reason,
            raw_response=data,
        )


# ── Google Gemini ─────────────────────────────────────────────────────────────
class GeminiClient(BaseLLMClient):
    """
    Client for Google's Gemini API (v1beta generateContent endpoint).
    Supports: temperature, top_p, top_k, max_output_tokens.
    """

    BASE_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={api_key}"
    )

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)
        self.api_key = GOOGLE_API_KEY

    async def _call_api(
        self,
        messages: list[dict],
        system_prompt: str,
        hyperparams: dict,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        url = self.BASE_URL_TEMPLATE.format(
            model=self.api_model, api_key=self.api_key
        )

        # Gemini uses a different message format
        parts = [{"text": system_prompt}]
        for msg in messages:
            parts.append({"text": msg["content"]})

        generation_config = {
            "temperature": hyperparams.get("temperature", 0.3),
            "topP": hyperparams.get("top_p", 0.9),
            "maxOutputTokens": hyperparams.get("max_tokens", 1024),
        }
        top_k = hyperparams.get("top_k", -1)
        if top_k != -1 and top_k is not None:
            generation_config["topK"] = int(top_k)

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
        latency_ms = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        answer = ""
        if candidates:
            parts_out = candidates[0].get("content", {}).get("parts", [])
            answer = " ".join(p.get("text", "") for p in parts_out)

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        finish_reason = (
            candidates[0].get("finishReason", "STOP") if candidates else "STOP"
        )

        tps = (output_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0.0

        return LLMResponse(
            model_id=self.model_id,
            answer_text=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=self._calculate_cost(input_tokens, output_tokens),
            ttft_ms=latency_ms * 0.10,
            total_latency_ms=latency_ms,
            tokens_per_second=tps,
            finish_reason=finish_reason,
            raw_response=data,
        )


# ── Groq (Llama 3.1 70B) ──────────────────────────────────────────────────────
# class GroqClient(BaseLLMClient):
#     """
#     Client for the Groq inference API (OpenAI-compatible endpoint).
#     Groq is very fast (high tokens-per-second) so TTFT is typically low.
#     Supports: temperature, top_p, max_tokens, presence_penalty, frequency_penalty.
#     """

#     BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

#     def __init__(self, model_config: dict) -> None:
#         super().__init__(model_config)
#         self.api_key = GROQ_API_KEY

#     async def _call_api(
#         self,
#         messages: list[dict],
#         system_prompt: str,
#         hyperparams: dict,
#     ) -> LLMResponse:
#         if not self.api_key:
#             raise RuntimeError("GROQ_API_KEY is not set")

#         full_messages = [
#             {"role": "system", "content": system_prompt},
#             *messages,
#         ]

#         payload = {
#             "model": self.api_model,
#             "messages": full_messages,
#             "temperature": hyperparams.get("temperature", 0.3),
#             "top_p": hyperparams.get("top_p", 0.9),
#             "max_tokens": hyperparams.get("max_tokens", 1024),
#             "presence_penalty": hyperparams.get("presence_penalty", 0.0),
#             "frequency_penalty": hyperparams.get("frequency_penalty", 0.0),
#         }

#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }

#         start = time.perf_counter()
#         async with httpx.AsyncClient(timeout=120.0) as client:
#             resp = await client.post(self.BASE_URL, json=payload, headers=headers)
#         latency_ms = (time.perf_counter() - start) * 1000

#         resp.raise_for_status()
#         data = resp.json()

#         answer = data["choices"][0]["message"]["content"]
#         usage = data.get("usage", {})
#         input_tokens = usage.get("prompt_tokens", 0)
#         output_tokens = usage.get("completion_tokens", 0)
#         finish_reason = data["choices"][0].get("finish_reason", "stop")

#         # Groq typically achieves very low TTFT (~100-200ms)
#         tps = (output_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0.0

#         return LLMResponse(
#             model_id=self.model_id,
#             answer_text=answer,
#             input_tokens=input_tokens,
#             output_tokens=output_tokens,
#             total_tokens=input_tokens + output_tokens,
#             cost_usd=self._calculate_cost(input_tokens, output_tokens),
#             ttft_ms=min(latency_ms * 0.08, 200),
#             total_latency_ms=latency_ms,
#             tokens_per_second=tps,
#             finish_reason=finish_reason,
#             raw_response=data,
#         )

# ── Groq (Llama 3.1 70B) ──────────────────────────────────────────────────────

import asyncio
import time
import httpx
from config import GROQ_API_KEY

# ── Groq (Llama 3.1 70B) ──────────────────────────────────────────────────────
class GroqClient(BaseLLMClient):
    """
    Tank-Mode Client for Groq.
    Handles strict Tokens-Per-Minute limits by sleeping for 65 seconds on 429s.
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)
        self.api_key = GROQ_API_KEY

    async def _call_api(
        self,
        messages: list[dict],
        system_prompt: str,
        hyperparams: dict,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": self.api_model,
            "messages": full_messages,
            "temperature": hyperparams.get("temperature", 0.3),
            "top_p": hyperparams.get("top_p", 0.9),
            "max_tokens": hyperparams.get("max_tokens", 1024),
            "presence_penalty": hyperparams.get("presence_penalty", 0.0),
            "frequency_penalty": hyperparams.get("frequency_penalty", 0.0),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 10
        data = None
        latency_ms = 0

        for attempt in range(max_retries):
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(self.BASE_URL, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 429:
                print(f"\n⏳ [Groq Limit] Bucket empty. Sleeping 65s (Attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(65.0)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            break 

        if data is None:
            print("\n❌ [Groq Error] Failed after 10 retries. Prompt too large for TPM limit. Skipping.")
            return LLMResponse(
                model_id=self.model_id, answer_text="ERROR: Exceeded Max TPM.",
                input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0, ttft_ms=0, total_latency_ms=0, tokens_per_second=0, finish_reason="length", raw_response={}
            )

        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            model_id=self.model_id,
            answer_text=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=self._calculate_cost(input_tokens, output_tokens),
            ttft_ms=min(latency_ms * 0.08, 200),
            total_latency_ms=latency_ms,
            tokens_per_second=(output_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0.0,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            raw_response=data,
        )

# ── Client Factory ────────────────────────────────────────────────────────────
def get_llm_client(model_config: dict) -> BaseLLMClient:
    """
    Factory that returns the appropriate client implementation based on the
    provider field in the model configuration dict.
    """
    provider = model_config.get("provider", "openai").lower()
    dispatch = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GeminiClient,
        "groq": GroqClient,
    }
    cls = dispatch.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {list(dispatch.keys())}"
        )
    return cls(model_config)
