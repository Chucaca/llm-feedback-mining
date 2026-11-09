import json

import pytest

from feedbackmining.cost import CostTracker
from feedbackmining.llm.base import LLMClient, LLMResponse
from feedbackmining.llm.mock import MockLLM
from feedbackmining.pipeline import JsonlCheckpoint, make_batches, run_pipeline, run_stage, with_retry
from feedbackmining.schemas import Note


class CountingLLM(MockLLM):
    """Mock that records which ids it was asked about."""

    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    def complete(self, system, user, max_tokens=4096):
        self.seen += [n["id"] for n in json.loads(user)["notes"]]
        return super().complete(system, user, max_tokens)


class FlakyLLM(MockLLM):
    """Fails the first ``fail_times`` calls (bad JSON), then behaves."""

    def __init__(self, fail_times):
        super().__init__()
        self.fail_times, self.calls = fail_times, 0

    def complete(self, system, user, max_tokens=4096):
        self.calls += 1
        if self.calls <= self.fail_times:
            return LLMResponse("not json", 10, 2, self.model, self.provider)
        return super().complete(system, user, max_tokens)


NOTES = [Note(id=f"N{i}", text=t) for i, t in enumerate([
    "thieu tinh nang hoa don dien tu", "khach che gia cao", "Need QR ordering, we don't support it",
    "goi 3 lan ko bat may", "phan mem chua phu hop mo hinh kinh doanh", "liked our loyalty program but price too high",
])]


def test_checkpoint_resume_skips_done_ids(tmp_path):
    ck = JsonlCheckpoint(tmp_path / "s1.jsonl")
    llm = CountingLLM()
    run_stage("stage1", NOTES[:3], llm, ck, sleep=lambda s: None)
    assert sorted(llm.seen) == ["N0", "N1", "N2"]

    llm2 = CountingLLM()
    results, stats = run_stage("stage1", NOTES, llm2, ck, sleep=lambda s: None)
    assert sorted(llm2.seen) == ["N3", "N4", "N5"]  # only the new ids were sent
    assert stats.skipped == 3 and stats.processed == 3
    assert set(results) == {n.id for n in NOTES}


def test_checkpoint_tolerates_torn_last_line(tmp_path):
    p = tmp_path / "s1.jsonl"
    p.write_text('{"id": "N0", "has_feature_issue": false, "signal_strength": 0}\n{"id": "N1", "has_f',
                 encoding="utf-8")
    assert JsonlCheckpoint(p).done_ids() == {"N0"}


def test_retry_recovers_from_bad_output(tmp_path):
    llm = FlakyLLM(fail_times=2)
    sleeps = []
    results, stats = run_stage("stage1", NOTES, llm, JsonlCheckpoint(tmp_path / "a.jsonl"),
                               sleep=sleeps.append)
    assert stats.failed == 0 and len(results) == len(NOTES)
    assert len(sleeps) == 2


def test_failed_batch_is_left_for_resume(tmp_path):
    ck = JsonlCheckpoint(tmp_path / "b.jsonl")
    _, stats = run_stage("stage1", NOTES, FlakyLLM(fail_times=99), ck, max_retries=1, sleep=lambda s: None)
    assert stats.failed == len(NOTES)
    assert ck.done_ids() == set()
    _, stats = run_stage("stage1", NOTES, MockLLM(), ck, sleep=lambda s: None)
    assert stats.processed == len(NOTES)


def test_with_retry_backoff_and_reraise():
    calls, sleeps = [], []

    def boom():
        calls.append(1)
        raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError):
        with_retry(boom, max_retries=3, base_delay=1.0, sleep=sleeps.append)
    assert len(calls) == 4 and len(sleeps) == 3
    assert all(0 <= s <= 2**i for i, s in enumerate(sleeps))


def test_token_aware_batches():
    notes = [Note(id=str(i), text="x" * 300) for i in range(10)]  # ~100 tokens + overhead each
    batches = make_batches(notes, max_input_tokens=250, max_items=40, overhead_tokens=20)
    assert [len(b) for b in batches] == [2, 2, 2, 2, 2]
    assert sum(len(b) for b in batches) == 10
    big = [Note(id="big", text="y" * 30000)]
    assert len(make_batches(big, max_input_tokens=100)) == 1  # oversize note still processed
    assert [len(b) for b in make_batches(notes, max_input_tokens=10**6, max_items=4)] == [4, 4, 2]


def test_two_stage_pipeline_only_sends_flagged_to_stage2(tmp_path):
    s2 = CountingLLM()
    tracker = CostTracker()
    out = run_pipeline(NOTES, MockLLM(), s2, tmp_path, tracker=tracker, sleep=lambda s: None)
    assert set(s2.seen) == set(out["flagged_ids"])
    assert "N1" not in s2.seen and "N3" not in s2.seen  # price / no answer never reach stage 2
    assert out["stage2"]["N0"]["canonical_features"] == ["e_invoice"]
    assert out["stage2"]["N2"]["canonical_features"] == ["qr_ordering"]
    assert out["stage2"]["N4"]["canonical_features"] == []  # vague complaint
    assert tracker.tokens("stage1")[0] > 0 and tracker.tokens("stage2")[0] > 0


def test_custom_client_protocol():
    class EchoJSON(LLMClient):
        provider, model = "x", "y"

        def complete(self, system, user, max_tokens=4096):
            ids = [n["id"] for n in json.loads(user)["notes"]]
            body = {"results": [{"id": i, "has_feature_issue": False, "signal_strength": 0} for i in ids]}
            return LLMResponse(json.dumps(body), 1, 1, self.model, self.provider)

    from feedbackmining.schemas import Stage1Batch

    batch, resp = EchoJSON().structured("s", json.dumps({"notes": [{"id": "a", "text": "t"}]}), Stage1Batch)
    assert batch.results[0].id == "a" and resp.input_tokens == 1
