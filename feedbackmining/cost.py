"""Token accounting and cost estimation from the (illustrative) price table."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

PRICES_PATH = Path(__file__).resolve().parents[1] / "config" / "prices.json"


def load_prices(path: Path | str = PRICES_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["providers"]


def estimate_tokens(text: str) -> int:
    """Rough token count without a tokenizer: ~3 characters per token for mixed
    Vietnamese/English (diacritics and short syllables tokenise worse than English).
    Deliberately conservative so batches stay under limits."""
    return max(1, math.ceil(len(text) / 3.0))


def cost_usd(input_tokens: int, output_tokens: int, price: dict) -> float:
    """Cost given a price entry ``{"input": usd_per_1m, "output": usd_per_1m}``."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


@dataclass
class CostTracker:
    """Accumulates usage per (stage, model) and prices it."""

    prices: dict = field(default_factory=load_prices)
    usage: dict = field(default_factory=dict)  # (stage, provider, model) -> [calls, in, out]

    def add(self, stage: str, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        u = self.usage.setdefault((stage, provider, model), [0, 0, 0])
        u[0] += 1
        u[1] += input_tokens
        u[2] += output_tokens

    def cost(self, stage: str, provider: str, model: str) -> float:
        calls, tin, tout = self.usage.get((stage, provider, model), [0, 0, 0])
        price = self.prices.get(provider, {}).get(stage, {"input": 0.0, "output": 0.0})
        return cost_usd(tin, tout, price)

    @property
    def total_usd(self) -> float:
        return sum(self.cost(*k) for k in self.usage)

    def rows(self) -> list[dict]:
        return [
            dict(stage=s, provider=p, model=m, calls=c, input_tokens=i, output_tokens=o,
                 usd=self.cost(s, p, m))
            for (s, p, m), (c, i, o) in sorted(self.usage.items())
        ]

    def tokens(self, stage: str | None = None) -> tuple[int, int]:
        tin = sum(v[1] for k, v in self.usage.items() if stage in (None, k[0]))
        tout = sum(v[2] for k, v in self.usage.items() if stage in (None, k[0]))
        return tin, tout


def project_cost_per_1k(stage_tokens: dict, n_notes: int, prices: dict | None = None) -> list[dict]:
    """Price the token usage of a run on every provider in the table, per 1,000 notes.

    ``stage_tokens`` maps ``stage1``/``stage2``/``single_stage`` to (input, output)
    token totals measured over ``n_notes`` notes. ``single_stage`` (strong model on
    every note) is priced with the stage-2 model for comparison.
    """
    prices = prices or load_prices()
    scale = 1000 / max(n_notes, 1)
    out = []
    for provider, p in prices.items():
        if provider == "mock":
            continue
        s1 = cost_usd(*stage_tokens["stage1"], p["stage1"]) * scale
        s2 = cost_usd(*stage_tokens["stage2"], p["stage2"]) * scale
        single = cost_usd(*stage_tokens["single_stage"], p["stage2"]) * scale
        out.append(dict(provider=provider, stage1_model=p["stage1"]["model"], stage2_model=p["stage2"]["model"],
                        two_stage_usd=s1 + s2, stage1_usd=s1, stage2_usd=s2, single_stage_usd=single,
                        saving=1 - (s1 + s2) / single if single else 0.0))
    return out
