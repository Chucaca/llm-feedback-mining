"""Prompts for both stages. Notes go in as a JSON array; answers come back as JSON.

The first line of each system prompt ("TASK: ...") is also how the offline
MockLLM recognises which stage it is answering.
"""
from __future__ import annotations

import json

from .schemas import Note
from .taxonomy import FEATURES

STAGE1_SYSTEM = """TASK: SCREEN
You read short notes written by sales reps explaining why a deal was lost.
Notes mix Vietnamese and English, often without diacritics, with typos.
For EACH note decide whether a missing or inadequate product capability
contributed to losing the deal.

signal_strength:
  0 = no product gap (price, competitor without a stated feature reason, no answer, timing, ...)
      A feature mentioned positively ("liked X but price too high") is 0.
  1 = vague product-fit complaint with no specific capability named
  2 = a specific missing capability is mentioned, but as a secondary reason
  3 = a specific missing capability is the main reason
has_feature_issue = signal_strength >= 1.

Return ONLY JSON: {"results": [{"id": str, "has_feature_issue": bool, "signal_strength": int}]}
with exactly one entry per input id."""

STAGE2_SYSTEM = """TASK: EXTRACT
You read lost-deal notes (Vietnamese/English, may lack diacritics) that were flagged
as involving a product gap. Extract every SPECIFIC capability the customer needed
and the product lacked. Copy each as a short phrase in the note's own words
(we normalise to our taxonomy afterwards). Do not list features the customer liked.
If the complaint is vague, return an empty list.

Known capability areas (for orientation, not an exhaustive list):
{taxonomy}

Return ONLY JSON: {{"results": [{{"id": str, "missing_features": [str], "evidence": str | null}}]}}
with exactly one entry per input id. "evidence" is the shortest supporting quote."""


def stage2_system() -> str:
    return STAGE2_SYSTEM.format(taxonomy="\n".join(f"- {f.name}" for f in FEATURES))


def user_payload(notes: list[Note]) -> str:
    return json.dumps({"notes": [{"id": n.id, "text": n.text} for n in notes]}, ensure_ascii=False)
