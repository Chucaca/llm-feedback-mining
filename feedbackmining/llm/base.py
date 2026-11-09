"""Provider-agnostic client interface."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and prose around it."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in model output")
        text = text[start:end + 1]
    return json.loads(text)


class LLMClient:
    """Subclasses implement ``complete``; ``structured`` adds parsing + validation.

    Providers may override ``structured`` to use `instructor` when installed.
    """

    provider: str = "base"
    model: str = ""

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def structured(self, system: str, user: str, schema: type[BaseModel],
                   max_tokens: int = 4096) -> tuple[BaseModel, LLMResponse]:
        resp = self.complete(system, user, max_tokens=max_tokens)
        return schema.model_validate(parse_json(resp.text)), resp
