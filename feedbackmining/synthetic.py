"""Synthetic "lost deal reason" notes with ground-truth labels.

Notes imitate what sales reps type into a CRM: short, mixed Vietnamese/English,
often without diacritics, with typos and abbreviations. Every note carries:

* ``has_feature_issue`` - True when the deal was (partly) lost to a product gap,
* ``signal_strength``   - 0 none, 1 vague product-fit complaint, 2 specific gap
                          mentioned as a secondary reason, 3 specific gap is the
                          primary reason,
* ``missing_features``  - canonical feature ids (empty for strength 0 and 1).

Some notes mention a feature *positively* ("liked the loyalty module but price
too high"): these are hard negatives for keyword heuristics.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .taxonomy import FEATURES, strip_accents

PRICE = [
    "Khách chê giá cao", "khach che gia dat qua", "Giá cao hơn ngân sách", "chua co ngan sach nam nay",
    "Customer said the price is too high", "price too high vs budget", "khach muon giam gia 50% moi mua",
    "phi hang nam cao qua", "too expensive for a small shop",
]
COMPETITOR = [
    "Khách đã chọn phần mềm khác", "chon ben doi thu vi re hon", "went with a competitor",
    "dang dung phan mem cu, ngai chuyen", "anh chủ quen người bán bên khác", "signed with another vendor",
    "doi thu tang them 6 thang mien phi",
]
NO_ANSWER = [
    "Không nghe máy", "goi 3 lan ko bat may", "no answer after several calls", "thue bao khong lien lac duoc",
    "Khách không phản hồi zalo", "kh tra loi tin nhan", "unreachable, wrong number", "hen goi lai nhung mat lien lac",
]
TIMING = [
    "Chưa mở cửa hàng, hẹn tháng sau", "dang sua quan, de sau tet tinh", "not opening until next quarter",
    "khach di cong tac, chua quyet", "chủ quán đang bận, để sau", "postponed, revisit in 3 months",
]
OTHER = [
    "Khách đóng cửa hàng", "dong cua quan roi", "lead ảo, không có nhu cầu", "test account only",
    "chi hoi tham khao", "duplicate lead",
]
VAGUE = [
    "Phần mềm chưa đáp ứng nhu cầu", "phan mem chua phu hop mo hinh kinh doanh", "product doesn't fit their workflow",
    "khach thay thieu nhieu tinh nang", "tính năng chưa đủ so với yêu cầu", "missing some features they need",
    "chua dap ung duoc nghiep vu cua khach",
]
GAP_PRIMARY = [
    "Khách cần {f} mà bên mình chưa có", "thieu tinh nang {f}", "Need {f}, we don't support it",
    "khach hoi {f} nhung chua ho tro nen thoi", "Không có {f} nên khách không mua", "kh yeu cau {f}, sp chua co",
    "Lost because we lack {f}", "bat buoc phai co {f} ma minh chua lam", "missing {f} feature",
    "Khách cần {f} và {g}, bên mình chưa hỗ trợ", "need {f} and {g}, not available",
]
GAP_SECONDARY = [
    "{p}, ngoài ra khách cũng hỏi {f}", "{p}; also asked about {f} which we don't have",
    "chon doi thu vi ho co {f}", "went with competitor because they have {f}", "{p}, them nua chua co {f}",
]
POSITIVE = [
    "Khách thích {f} nhưng {p}", "liked our {f} but {p}", "khach hai long voi {f}, tuy nhien {p}",
    "demo {f} ok, but {p}",
]

CATEGORY_P = {
    "price": 0.20, "competitor": 0.13, "no_answer": 0.16, "timing": 0.09, "other": 0.05,
    "gap_primary": 0.17, "gap_secondary": 0.08, "vague": 0.06, "positive_mention": 0.06,
}


def _feature_weights() -> list[float]:
    # Zipf-like popularity so that the "top missing features" ranking is meaningful.
    return [1.0 / (i + 1) ** 0.8 for i in range(len(FEATURES))]


def _surface(rng: random.Random, feat) -> str:
    pool = list(feat.aliases) + list(feat.paraphrases) * 2  # unseen paraphrases are common
    return rng.choice(pool)


def _typo(rng: random.Random, text: str) -> str:
    chars = list(text)
    for _ in range(rng.randint(1, 2)):
        if len(chars) < 6:
            break
        i = rng.randrange(1, len(chars) - 1)
        op = rng.random()
        if op < 0.4:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op < 0.7:
            del chars[i]
        else:
            chars.insert(i, chars[i])
    return "".join(chars)


def _noise(rng: random.Random, text: str) -> str:
    if rng.random() < 0.45:
        text = strip_accents(text)
    if rng.random() < 0.15:
        text = _typo(rng, text)
    r = rng.random()
    if r < 0.15:
        text = text.lower()
    elif r < 0.20:
        text = text.upper()
    if rng.random() < 0.2:
        text = rng.choice(["Note: ", "KH: ", "[lost] ", "Sale cmt: ", ""]) + text
    if rng.random() < 0.15:
        text += rng.choice([" ...", ".", "!!", " - da chot lost", " (follow up later)"])
    return text


def generate(n: int = 5000, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    cats, probs = zip(*CATEGORY_P.items())
    weights = _feature_weights()
    notes = []
    for i in range(n):
        cat = rng.choices(cats, probs)[0]
        feats: list[str] = []
        strength = 0
        if cat == "price":
            text = rng.choice(PRICE)
        elif cat == "competitor":
            text = rng.choice(COMPETITOR)
        elif cat == "no_answer":
            text = rng.choice(NO_ANSWER)
        elif cat == "timing":
            text = rng.choice(TIMING)
        elif cat == "other":
            text = rng.choice(OTHER)
        elif cat == "vague":
            text, strength = rng.choice(VAGUE), 1
        elif cat == "positive_mention":
            f = rng.choices(FEATURES, weights)[0]
            text = rng.choice(POSITIVE).format(f=_surface(rng, f), p=rng.choice(PRICE).lower())
        else:
            tmpl = rng.choice(GAP_PRIMARY if cat == "gap_primary" else GAP_SECONDARY)
            f = rng.choices(FEATURES, weights)[0]
            g = f
            while g.id == f.id:
                g = rng.choices(FEATURES, weights)[0]
            text = tmpl.format(f=_surface(rng, f), g=_surface(rng, g), p=rng.choice(PRICE))
            feats = [f.id] + ([g.id] if "{g}" in tmpl else [])
            strength = 3 if cat == "gap_primary" else 2
        notes.append(
            dict(
                id=f"N{i:05d}",
                text=_noise(rng, text),
                category=cat,
                has_feature_issue=strength >= 1,
                signal_strength=strength,
                missing_features=feats,
            )
        )
    return notes


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
