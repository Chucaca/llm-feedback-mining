import pytest

from feedbackmining.cost import CostTracker, cost_usd, estimate_tokens, project_cost_per_1k
from feedbackmining.synthetic import generate
from feedbackmining.taxonomy import FEATURES, TaxonomyNormalizer, clean, strip_accents


@pytest.fixture(scope="module")
def norm():
    return TaxonomyNormalizer()


def test_strip_accents():
    assert strip_accents("Hóa Đơn Điện Tử") == "hoa don dien tu"
    assert clean("  QR-ordering!! ") == "qr ordering"


@pytest.mark.parametrize("phrase,expected", [
    ("hóa đơn điện tử", "e_invoice"),
    ("hoa don dien tu", "e_invoice"),        # no diacritics
    ("hoa don dien tuu", "e_invoice"),       # typo
    ("E-Invoice", "e_invoice"),
    ("cham cong", "payroll"),
    ("tinh luong nhan vien", "payroll"),
    ("QR ordering", "qr_ordering"),
    ("multi branch", "multi_branch"),
    ("quan ly cong no", "customer_debt"),
    ("kitchen display", "kitchen_display"),
])
def test_normalize_variants(norm, phrase, expected):
    assert norm.match(phrase)[0] == expected


def test_unrelated_phrase_is_rejected(norm):
    assert norm.match("xyz qwerty")[0] is None
    assert norm.match("")[0] is None


def test_normalize_dedupes_and_keeps_order(norm):
    assert norm.normalize(["cham cong", "payroll", "hoa don dien tu", "zzzz"]) == ["payroll", "e_invoice"]


def test_every_alias_maps_to_its_feature(norm):
    for f in FEATURES:
        for a in f.aliases:
            assert norm.match(a)[0] == f.id, a


def test_cost_usd():
    price = {"input": 1.0, "output": 5.0}  # USD per 1M tokens
    assert cost_usd(1_000_000, 0, price) == pytest.approx(1.0)
    assert cost_usd(200_000, 100_000, price) == pytest.approx(0.2 + 0.5)
    with pytest.raises(ValueError):
        cost_usd(-1, 0, price)


def test_cost_tracker_accumulates():
    prices = {"p": {"stage1": {"input": 2.0, "output": 4.0}}}
    t = CostTracker(prices=prices)
    t.add("stage1", "p", "m", 500_000, 100_000)
    t.add("stage1", "p", "m", 500_000, 100_000)
    assert t.tokens("stage1") == (1_000_000, 200_000)
    assert t.total_usd == pytest.approx(2.0 + 0.8)
    assert t.rows()[0]["calls"] == 2
    # unknown provider is priced at zero, not an error
    t.add("stage1", "unknown", "m", 10, 10)
    assert t.total_usd == pytest.approx(2.8)


def test_project_cost_per_1k_two_stage_cheaper():
    prices = {"p": {"stage1": {"model": "small", "input": 0.1, "output": 0.4},
                    "stage2": {"model": "big", "input": 3.0, "output": 15.0}},
              "mock": {"stage1": {"model": "m", "input": 0, "output": 0},
                       "stage2": {"model": "m", "input": 0, "output": 0}}}
    tokens = {"stage1": (2_000_000, 200_000), "stage2": (300_000, 30_000), "single_stage": (2_200_000, 400_000)}
    rows = project_cost_per_1k(tokens, n_notes=2000, prices=prices)
    assert [r["provider"] for r in rows] == ["p"]  # mock excluded
    r = rows[0]
    assert r["stage1_usd"] == pytest.approx((2.0 * 0.1 + 0.2 * 0.4) / 2)
    assert r["two_stage_usd"] < r["single_stage_usd"]
    assert 0 < r["saving"] < 1


def test_estimate_tokens_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 30) == 10


def test_synthetic_labels_consistent():
    notes = generate(n=500, seed=1)
    assert len({n["id"] for n in notes}) == 500
    for n in notes:
        assert n["has_feature_issue"] == (n["signal_strength"] >= 1)
        assert bool(n["missing_features"]) == (n["signal_strength"] >= 2)
