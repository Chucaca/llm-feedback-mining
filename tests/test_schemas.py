import pytest
from pydantic import ValidationError

from feedbackmining.llm.base import parse_json
from feedbackmining.schemas import Stage1Batch, Stage1Result, Stage2Result


def test_stage1_valid():
    r = Stage1Result(id="N1", has_feature_issue=True, signal_strength=3)
    assert r.signal_strength == 3


@pytest.mark.parametrize("strength", [-1, 4])
def test_stage1_strength_range(strength):
    with pytest.raises(ValidationError):
        Stage1Result(id="N1", has_feature_issue=True, signal_strength=strength)


def test_stage1_flag_must_match_strength():
    with pytest.raises(ValidationError):
        Stage1Result(id="N1", has_feature_issue=False, signal_strength=2)
    with pytest.raises(ValidationError):
        Stage1Result(id="N1", has_feature_issue=True, signal_strength=0)


def test_stage1_batch_from_llm_json_with_fence_and_extra_fields():
    text = 'Sure!\n```json\n{"results": [{"id": "a", "has_feature_issue": false, "signal_strength": 0, "why": "x"}]}\n```'
    batch = Stage1Batch.model_validate(parse_json(text))
    assert batch.results[0].id == "a"


def test_stage2_strips_and_dedupes():
    r = Stage2Result(id="N1", missing_features=["  e-invoice ", "e-invoice", "", "QR ordering"])
    assert r.missing_features == ["e-invoice", "QR ordering"]
    assert r.canonical_features == []


def test_stage2_requires_list():
    with pytest.raises(ValidationError):
        Stage2Result(id="N1", missing_features="e-invoice")


def test_parse_json_rejects_non_json():
    with pytest.raises(ValueError):
        parse_json("no json here")
