"""Real provider clients. SDKs are imported lazily, so none of them is required
unless that provider is selected.

Selection (see ``get_client``):

    LLM_PROVIDER = mock | anthropic | openai | gemini | deepseek   (default: mock)
    LLM_MODEL    = optional model override (default: from config/prices.json per stage)

API keys use each SDK's usual variable: ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY (or GOOGLE_API_KEY), DEEPSEEK_API_KEY.
"""
from __future__ import annotations

import os

from pydantic import BaseModel

from ..cost import load_prices
from .base import LLMClient, LLMResponse
from .mock import MockLLM


def _instructor():
    try:
        import instructor  # type: ignore

        return instructor
    except ImportError:
        return None


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str):
        import anthropic  # lazy

        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system, user, max_tokens=4096):
        r = self._client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                         messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in r.content)
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens, self.model, self.provider)

    def structured(self, system, user, schema: type[BaseModel], max_tokens=4096):
        inst = _instructor()
        if inst is None:
            return super().structured(system, user, schema, max_tokens)
        client = inst.from_anthropic(self._client)
        obj, raw = client.messages.create_with_completion(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], response_model=schema, max_retries=1)
        return obj, LLMResponse("", raw.usage.input_tokens, raw.usage.output_tokens, self.model, self.provider)


class OpenAIClient(LLMClient):
    """OpenAI, and any OpenAI-compatible endpoint (used for DeepSeek)."""

    provider = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 provider: str = "openai"):
        import openai  # lazy

        self.model, self.provider = model, provider
        self._client = openai.OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))

    def complete(self, system, user, max_tokens=4096):
        r = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return LLMResponse(r.choices[0].message.content or "", r.usage.prompt_tokens,
                           r.usage.completion_tokens, self.model, self.provider)

    def structured(self, system, user, schema: type[BaseModel], max_tokens=4096):
        inst = _instructor()
        if inst is None or self.provider != "openai":
            return super().structured(system, user, schema, max_tokens)
        client = inst.from_openai(self._client)
        obj, raw = client.chat.completions.create_with_completion(
            model=self.model, max_tokens=max_tokens, response_model=schema, max_retries=1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return obj, LLMResponse("", raw.usage.prompt_tokens, raw.usage.completion_tokens, self.model,
                                self.provider)


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str):
        from google import genai  # lazy (package: google-genai)

        self.model = model
        self._genai = genai
        self._client = genai.Client()

    def complete(self, system, user, max_tokens=4096):
        cfg = self._genai.types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens, response_mime_type="application/json")
        r = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        u = r.usage_metadata
        return LLMResponse(r.text or "", u.prompt_token_count or 0, u.candidates_token_count or 0,
                           self.model, self.provider)


def get_client(stage: str = "stage1", provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build the client for a pipeline stage from env vars / arguments."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "mock")).lower()
    prices = load_prices()
    if provider not in prices:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(prices)}")
    model = model or os.environ.get("LLM_MODEL") or prices[provider][stage]["model"]
    if provider == "mock":
        return MockLLM(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    if provider == "deepseek":
        return OpenAIClient(model, base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY",
                            provider="deepseek")
    if provider == "gemini":
        return GeminiClient(model)
    raise ValueError(provider)  # pragma: no cover
