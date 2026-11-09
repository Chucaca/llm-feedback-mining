# Evaluation (synthetic ground truth)

> **Synthetic data, offline MockLLM.** The mock is a keyword/regex heuristic; a real LLM would be evaluated the same way by setting `LLM_PROVIDER`. Numbers show the evaluation harness, not a claim about any production system.

Deterministic split by hashed note id: 3,560 train / 1,440 test notes. The TF-IDF baseline is trained on the train split; everything is scored on the test split.

## Stage 1 - has_feature_issue

| classifier | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| MockLLM heuristic (zero-shot, no training data) | 0.879 | 0.962 | 0.919 | 436 | 60 | 17 |
| TF-IDF + LogisticRegression (supervised) | 0.998 | 1.000 | 0.999 | 453 | 1 | 0 |

Exact signal_strength (0-3) accuracy of the mock: 94.4%.

## Stage 2 - missing_features (normalised to taxonomy)

| scope | micro-precision | micro-recall | micro-F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| end-to-end (Stage 1 misses count as misses) | 0.872 | 0.735 | 0.797 | 285 | 42 | 103 |
| notes with a specific gap that Stage 1 flagged (n=336) | 0.893 | 0.752 | 0.817 | 285 | 34 | 94 |

Ranking agreement on the full corpus: Spearman rho between extracted and true feature counts = 0.73; 9/10 of the true top-10 features are in the extracted top-10. Product decisions use the *ranking*, which is more robust than per-note extraction.

## Cost per 1,000 notes (ILLUSTRATIVE prices)

Token usage measured on this corpus (5,000 notes, 1,704 sent to Stage 2): Stage 1 148,600 in / 113,348 out; Stage 2 64,530 in / 61,287 out; single-stage alternative (strong model on every note) 167,185 in / 127,916 out. Token counts are character-based estimates.

| provider | stage-1 model | stage-2 model | stage 1 | stage 2 | two-stage total | single-stage | saving |
|---|---|---|---|---|---|---|---|
| anthropic | claude-haiku-4-5 | claude-sonnet-4-5 | $0.1431 | $0.2226 | **$0.3656** | $0.4841 | 24.5% |
| openai | gpt-4o-mini | gpt-4o | $0.0181 | $0.1548 | **$0.1729** | $0.3394 | 49.1% |
| gemini | gemini-2.0-flash | gemini-2.5-pro | $0.0120 | $0.1387 | **$0.1507** | $0.2976 | 49.4% |
| deepseek | deepseek-chat | deepseek-chat | $0.0330 | $0.0170 | **$0.0499** | $0.0372 | -34.3% |

**Prices in `config/prices.json` are illustrative examples and go stale; check each provider's pricing page.** The point is the structure: the cheap model reads everything, the strong model reads only the flagged minority.
