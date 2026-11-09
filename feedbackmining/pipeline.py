"""Batching, retry, checkpoint/resume and the two-stage run."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .cost import CostTracker, estimate_tokens
from .llm.base import LLMClient
from .prompts import STAGE1_SYSTEM, stage2_system, user_payload
from .schemas import Note, Stage1Batch, Stage1Result, Stage2Batch, Stage2Result
from .taxonomy import TaxonomyNormalizer, default_normalizer

# --- batching --------------------------------------------------------------------


def make_batches(notes: list[Note], max_input_tokens: int = 2500, max_items: int = 40,
                 overhead_tokens: int = 20) -> list[list[Note]]:
    """Greedy token-aware batching: a batch is closed when adding the next note would
    exceed ``max_input_tokens`` (prompt payload only) or ``max_items``. A single
    oversized note still gets its own batch."""
    batches, cur, cur_tok = [], [], 0
    for n in notes:
        t = estimate_tokens(n.text) + overhead_tokens
        if cur and (cur_tok + t > max_input_tokens or len(cur) >= max_items):
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(n)
        cur_tok += t
    if cur:
        batches.append(cur)
    return batches


# --- retry -------------------------------------------------------------------------


def with_retry(fn: Callable, max_retries: int = 4, base_delay: float = 1.0, max_delay: float = 30.0,
               sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None):
    """Call ``fn()``; on exception retry with exponential backoff + full jitter.
    Re-raises the last exception after ``max_retries`` retries."""
    rng = rng or random.Random(0)
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 - API, JSON and validation errors are all retryable here
            if attempt == max_retries:
                raise
            sleep(rng.uniform(0, min(max_delay, base_delay * 2**attempt)))


# --- checkpoint --------------------------------------------------------------------


class JsonlCheckpoint:
    """Append-only JSONL store keyed by ``id``. Each processed note is one line, so a
    crash loses at most the in-flight batch and a rerun skips everything already done."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not self.path.exists():
            return out
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # torn last line after a crash
                    continue
                out[rec["id"]] = rec
        return out

    def done_ids(self) -> set[str]:
        return set(self.load())

    def append(self, records: Iterable[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()


# --- stage runner --------------------------------------------------------------------


@dataclass
class StageStats:
    total: int = 0
    skipped: int = 0
    processed: int = 0
    failed: int = 0
    batches: int = 0


def run_stage(stage: str, notes: list[Note], client: LLMClient, checkpoint: JsonlCheckpoint,
              tracker: CostTracker | None = None, max_input_tokens: int = 2500, max_items: int = 40,
              max_retries: int = 4, sleep: Callable[[float], None] = time.sleep,
              normalizer: TaxonomyNormalizer | None = None) -> tuple[dict[str, dict], StageStats]:
    """Run one stage over ``notes`` with resume. Returns (all results by id, stats)."""
    if stage == "stage1":
        system, schema = STAGE1_SYSTEM, Stage1Batch
    elif stage == "stage2":
        system, schema = stage2_system(), Stage2Batch
        normalizer = normalizer or default_normalizer()
    else:
        raise ValueError(stage)

    done = checkpoint.done_ids()
    todo = [n for n in notes if n.id not in done]
    stats = StageStats(total=len(notes), skipped=len(notes) - len(todo))

    for batch in make_batches(todo, max_input_tokens, max_items):
        stats.batches += 1
        ids = {n.id for n in batch}
        user = user_payload(batch)

        def call():
            parsed, resp = client.structured(system, user, schema, max_tokens=60 * len(batch) + 200)
            if tracker is not None:
                tracker.add(stage, client.provider, client.model, resp.input_tokens, resp.output_tokens)
            got = {r.id for r in parsed.results}
            if not ids <= got:  # the model dropped some ids -> retry the batch
                raise ValueError(f"missing ids in response: {sorted(ids - got)[:5]}")
            return parsed

        try:
            parsed = with_retry(call, max_retries=max_retries, sleep=sleep)
        except Exception as exc:  # noqa: BLE001 - leave the batch for the next resume
            stats.failed += len(batch)
            print(f"[warn] {stage} batch of {len(batch)} failed after retries: {exc}")
            continue

        records = []
        for r in parsed.results:
            if r.id not in ids:
                continue  # ignore hallucinated ids
            if isinstance(r, Stage2Result):
                r.canonical_features = normalizer.normalize(r.missing_features)
            records.append(r.model_dump())
        checkpoint.append(records)
        stats.processed += len(records)

    return checkpoint.load(), stats


def run_pipeline(notes: list[Note], stage1_client: LLMClient, stage2_client: LLMClient, out_dir: Path,
                 tracker: CostTracker | None = None, min_strength: int = 1, **kw) -> dict:
    """Stage 1 on every note; Stage 2 only on notes with ``signal_strength >= min_strength``."""
    out_dir = Path(out_dir)
    s1, st1 = run_stage("stage1", notes, stage1_client, JsonlCheckpoint(out_dir / "stage1.jsonl"),
                        tracker, **kw)
    by_id = {n.id: n for n in notes}
    flagged = [by_id[i] for i, r in s1.items()
               if i in by_id and Stage1Result(**r).signal_strength >= min_strength]
    s2, st2 = run_stage("stage2", flagged, stage2_client, JsonlCheckpoint(out_dir / "stage2.jsonl"),
                        tracker, **kw)
    return {"stage1": s1, "stage2": {k: v for k, v in s2.items() if k in {n.id for n in flagged}},
            "stats": {"stage1": st1, "stage2": st2}, "flagged_ids": [n.id for n in flagged]}
