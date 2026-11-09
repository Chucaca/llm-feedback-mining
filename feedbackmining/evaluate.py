"""Evaluation against synthetic ground truth + cost projection per provider."""
from __future__ import annotations

import hashlib
from collections import Counter

from .baseline import train_baseline
from .cost import CostTracker, load_prices, project_cost_per_1k
from .llm.base import LLMClient
from .pipeline import make_batches
from .prompts import STAGE1_SYSTEM, stage2_system, user_payload
from .schemas import Note, Stage1Batch, Stage2Batch


def is_test(note_id: str, test_share: float = 0.3) -> bool:
    """Deterministic split by hashed id (stable across runs and machines)."""
    h = int(hashlib.md5(note_id.encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000 < test_share


def prf(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def micro_prf(true_sets: list[set], pred_sets: list[set]) -> dict:
    """Micro-averaged P/R/F1 over (note, feature) pairs."""
    tp = sum(len(t & p) for t, p in zip(true_sets, pred_sets))
    fp = sum(len(p - t) for t, p in zip(true_sets, pred_sets))
    fn = sum(len(t - p) for t, p in zip(true_sets, pred_sets))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def spearman(a: dict, b: dict) -> float:
    """Spearman rank correlation of two count dicts over the union of keys."""
    from scipy.stats import spearmanr

    keys = sorted(set(a) | set(b))
    if len(keys) < 3:
        return float("nan")
    return float(spearmanr([a.get(k, 0) for k in keys], [b.get(k, 0) for k in keys]).statistic)


def measure_tokens(stage: str, notes: list[Note], client: LLMClient, **batch_kw) -> tuple[int, int]:
    """Token usage of running ``stage`` on ``notes`` (no checkpointing)."""
    system, schema = (STAGE1_SYSTEM, Stage1Batch) if stage == "stage1" else (stage2_system(), Stage2Batch)
    tracker = CostTracker()
    for batch in make_batches(notes, **batch_kw):
        _, resp = client.structured(system, user_payload(batch), schema)
        tracker.add(stage, client.provider, client.model, resp.input_tokens, resp.output_tokens)
    return tracker.tokens(stage)


def evaluate(gold: list[dict], results: dict, client: LLMClient) -> dict:
    """``gold``: synthetic notes with labels; ``results``: output of ``run_pipeline``."""
    s1, s2 = results["stage1"], results["stage2"]
    train = [g for g in gold if not is_test(g["id"])]
    test = [g for g in gold if is_test(g["id"])]

    # Stage 1: LLM (mock) vs TF-IDF baseline on the held-out split.
    y_true = [g["has_feature_issue"] for g in test]
    y_llm = [bool(s1[g["id"]]["has_feature_issue"]) for g in test]
    baseline = train_baseline([g["text"] for g in train], [g["has_feature_issue"] for g in train])
    y_base = [bool(v) for v in baseline.predict([g["text"] for g in test])]
    strength_acc = sum(s1[g["id"]]["signal_strength"] == g["signal_strength"] for g in test) / len(test)

    # Stage 2: end-to-end micro-F1 (a note not flagged by stage 1 predicts no features).
    true_sets = [set(g["missing_features"]) for g in test]
    pred_sets = [set(s2.get(g["id"], {}).get("canonical_features", [])) for g in test]
    e2e = micro_prf(true_sets, pred_sets)
    gold_flag = [(t, p) for t, p, g in zip(true_sets, pred_sets, test)
                 if g["id"] in s2 and g["signal_strength"] >= 2]
    cond = micro_prf([t for t, _ in gold_flag], [p for _, p in gold_flag])

    # Ranking agreement on the full corpus.
    true_counts = Counter(f for g in gold for f in g["missing_features"])
    pred_counts = Counter(f for r in s2.values() for f in r.get("canonical_features", []))
    rank_corr = spearman(true_counts, pred_counts)
    top10_true = [k for k, _ in true_counts.most_common(10)]
    top10_pred = [k for k, _ in pred_counts.most_common(10)]

    # Cost: measured token usage on this corpus, priced for every provider.
    notes = [Note(id=g["id"], text=g["text"]) for g in gold]
    flagged = [n for n in notes if n.id in s2]
    tokens = {
        "stage1": measure_tokens("stage1", notes, client),
        "stage2": measure_tokens("stage2", flagged, client),
        "single_stage": measure_tokens("stage2", notes, client),
    }
    costs = project_cost_per_1k(tokens, len(notes), load_prices())

    return dict(
        n_train=len(train), n_test=len(test),
        stage1_llm=prf(y_true, y_llm), stage1_baseline=prf(y_true, y_base), strength_acc=strength_acc,
        stage2_e2e=e2e, stage2_on_true_positives=cond, n_cond=len(gold_flag),
        rank_corr=rank_corr, top10_overlap=len(set(top10_true) & set(top10_pred)),
        tokens=tokens, costs=costs, n_notes=len(notes), n_flagged=len(flagged),
    )
