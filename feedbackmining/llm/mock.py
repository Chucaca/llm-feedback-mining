"""Deterministic offline stand-in for an LLM.

``MockLLM`` speaks the same protocol as the real clients (system + user prompt in,
JSON text and token usage out), so the whole pipeline - batching, parsing,
validation, checkpointing and cost accounting - runs unchanged without API keys.
Behind the protocol it is a keyword + regex heuristic over accent-stripped text,
plus fuzzy taxonomy matching for extraction. It is intentionally imperfect (it
misses paraphrases and is fooled by some phrasings), which makes the evaluation
meaningful.
"""
from __future__ import annotations

import json
import re

from ..cost import estimate_tokens
from ..taxonomy import FEATURES, clean, default_normalizer
from .base import LLMClient, LLMResponse

GAP_CUE = re.compile(
    r"\b(thieu tinh nang|thieu|chua co|khong co|ko co|chua ho tro|khong ho tro|chua lam|can|need|needs|"
    r"lack|lacks|missing|don t support|do not support|not available|we don t have|yeu cau|"
    r"bat buoc phai co|vi ho co|because they have|asked about|cung hoi)\b"
)
VAGUE = re.compile(
    r"(chua dap ung|chua phu hop|khong phu hop|doesn t fit|not fit|thieu nhieu tinh nang|"
    r"missing some features|chua du so voi)"
)
POSITIVE = re.compile(r"\b(thich|hai long|liked|demo)\b")
OTHER_REASON = re.compile(r"\b(gia|price|expensive|ngan sach|budget|doi thu|competitor|vendor|phi|giam gia)\b")

# Words that end a feature span after a cue ("thieu X ma ..." -> X).
STOP = {"ma", "nhung", "nen", "we", "which", "feature", "not", "sp", "ben", "chua", "khong", "ko",
        "minh", "they", "is", "are", "da", "follow", "tuy"}
SPLIT = re.compile(r"\b(?:va|and)\b")

_ALIASES = sorted(((clean(a), f.id) for f in FEATURES for a in f.aliases), key=lambda x: -len(x[0]))


def known_features(t: str) -> list[tuple[str, str]]:
    """Exact (accent-insensitive) alias hits as (alias, feature id)."""
    hits, seen = [], set()
    padded = f" {t} "
    for alias, fid in _ALIASES:
        if fid not in seen and f" {alias} " in padded:
            hits.append((alias, fid))
            seen.add(fid)
    return hits


def screen(text: str) -> tuple[bool, int]:
    t = clean(text)
    gap = GAP_CUE.search(t) is not None
    if POSITIVE.search(t) and not gap:
        return False, 0
    if VAGUE.search(t):
        return True, 1
    if gap and (known_features(t) or len(t.split()) >= 4):
        return True, 2 if OTHER_REASON.search(t) else 3
    return False, 0


def extract(text: str) -> tuple[list[str], str | None]:
    t = clean(text)
    if VAGUE.search(t) or (POSITIVE.search(t) and not GAP_CUE.search(t)):
        return [], None
    phrases = [alias for alias, _ in known_features(t)]
    norm = default_normalizer()
    for m in GAP_CUE.finditer(t):
        tail = t[m.end():].split()
        span = []
        for w in tail:
            if w in STOP:
                break
            span.append(w)
        for part in SPLIT.split(" ".join(span[:8])):
            part = part.strip()
            if part and norm.match(part)[0] and part not in phrases:
                phrases.append(part)
    evidence = text.strip()[:120] if phrases else None
    return phrases, evidence


class MockLLM(LLMClient):
    provider = "mock"

    def __init__(self, model: str = "mock-heuristic"):
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> LLMResponse:
        notes = json.loads(user)["notes"]
        if system.startswith("TASK: SCREEN"):
            results = []
            for n in notes:
                flag, strength = screen(n["text"])
                results.append({"id": n["id"], "has_feature_issue": flag, "signal_strength": strength})
        elif system.startswith("TASK: EXTRACT"):
            results = []
            for n in notes:
                phrases, ev = extract(n["text"])
                results.append({"id": n["id"], "missing_features": phrases, "evidence": ev})
        else:
            raise ValueError("MockLLM only understands the SCREEN and EXTRACT prompts")
        text = json.dumps({"results": results}, ensure_ascii=False)
        return LLMResponse(text=text, input_tokens=estimate_tokens(system + user),
                           output_tokens=estimate_tokens(text), model=self.model, provider=self.provider)
