"""Canonical feature taxonomy and fuzzy normalisation.

The taxonomy is generic for a retail / F&B point-of-sale + CRM product and is
invented for this demo. Each feature has:

* ``aliases``      - surface forms the normaliser *knows* (Vietnamese with and
                     without diacritics, English). Used to fit the TF-IDF index.
* ``paraphrases``  - extra phrasings used only by the synthetic generator. They
                     are deliberately *not* given to the normaliser, so the
                     evaluation measures generalisation to unseen wording.

Normalisation maps any free-text feature phrase (from an LLM or the mock) to a
canonical id via cosine similarity of character n-gram TF-IDF vectors, after
lower-casing and stripping Vietnamese diacritics.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Feature:
    id: str
    name: str
    aliases: tuple[str, ...]
    paraphrases: tuple[str, ...] = field(default_factory=tuple)


FEATURES: tuple[Feature, ...] = (
    Feature("e_invoice", "E-invoice integration",
            ("hóa đơn điện tử", "xuất hóa đơn điện tử", "e-invoice", "electronic invoice", "hddt"),
            ("xuat hoa don VAT tu dong", "issue VAT invoices automatically")),
    Feature("batch_expiry", "Batch & expiry tracking",
            ("quản lý lô date", "hạn sử dụng", "batch tracking", "expiry date tracking", "lô hàng"),
            ("theo doi date tung lo", "track expiration per batch")),
    Feature("multi_branch", "Multi-branch management",
            ("quản lý chuỗi", "nhiều chi nhánh", "multi-branch", "multi store", "chuỗi cửa hàng"),
            ("quan ly 5 cua hang cung luc", "manage several outlets in one account")),
    Feature("qr_ordering", "QR table ordering",
            ("gọi món qua QR", "order tại bàn bằng QR", "QR ordering", "scan to order", "menu QR"),
            ("khach tu quet ma goi mon", "guests order from their phone at the table")),
    Feature("kitchen_display", "Kitchen display system",
            ("màn hình bếp", "kitchen display", "KDS", "báo bếp"),
            ("bep xem order tren tablet", "screen in the kitchen for tickets")),
    Feature("delivery_integration", "Food-delivery app integration",
            ("kết nối app giao đồ ăn", "tích hợp app giao hàng", "delivery app integration",
             "food delivery integration"),
            ("don tu app giao do an do ve may tinh tien", "orders from delivery platforms flow into POS")),
    Feature("loyalty", "Loyalty points & membership",
            ("tích điểm", "thẻ thành viên", "loyalty program", "membership points", "khách hàng thân thiết"),
            ("doi diem lay qua", "reward points for regulars")),
    Feature("marketplace_sync", "E-commerce marketplace sync",
            ("đồng bộ sàn thương mại điện tử", "đồng bộ tồn kho với sàn", "marketplace sync",
             "e-commerce sync"),
            ("ban online tren san va tru kho tu dong", "sync stock with online marketplaces")),
    Feature("accounting_export", "Accounting software export",
            ("xuất dữ liệu kế toán", "kết nối phần mềm kế toán", "accounting export", "accounting integration"),
            ("day so lieu sang ke toan", "send data to the accountant's system")),
    Feature("payroll", "Staff payroll & timekeeping",
            ("chấm công", "tính lương nhân viên", "payroll", "timekeeping", "staff attendance"),
            ("tinh luong theo ca", "pay staff by shift")),
    Feature("offline_mode", "Offline mode",
            ("bán hàng offline", "chế độ offline", "offline mode", "works without internet"),
            ("mat mang van ban duoc", "keep selling when wifi drops")),
    Feature("mobile_app", "Full-featured mobile app",
            ("app điện thoại", "ứng dụng di động", "mobile app", "phone app"),
            ("quan ly tren dien thoai day du", "do everything from my phone")),
    Feature("custom_reports", "Custom reports",
            ("báo cáo tùy chỉnh", "báo cáo chi tiết", "custom reports", "report builder"),
            ("tu tao mau bao cao rieng", "build their own reports")),
    Feature("barcode_scale", "Barcode scale integration",
            ("cân điện tử", "kết nối cân", "barcode scale", "weighing scale integration"),
            ("in tem can cho rau thit", "weigh items and print price labels")),
    Feature("serial_imei", "Serial / IMEI tracking",
            ("quản lý IMEI", "số serial", "serial number tracking", "IMEI tracking"),
            ("theo doi bao hanh theo so may", "warranty by device number")),
    Feature("recipe_bom", "Recipe / ingredient costing",
            ("định lượng nguyên liệu", "công thức món", "recipe management", "ingredient costing", "BOM"),
            ("tru kho nguyen lieu theo mon", "deduct ingredients per dish sold")),
    Feature("promotions", "Advanced promotions engine",
            ("khuyến mãi nâng cao", "chương trình khuyến mãi", "promotion engine", "combo discounts",
             "voucher"),
            ("mua 2 tang 1 theo khung gio", "happy hour discounts")),
    Feature("customer_debt", "Customer debt management",
            ("công nợ", "quản lý công nợ khách hàng", "customer debt", "accounts receivable"),
            ("ghi no khach si", "track what wholesale buyers owe")),
    Feature("open_api", "Open API / webhooks",
            ("API mở", "kết nối API", "open API", "webhooks", "API access"),
            ("cho dev ben em ket noi he thong", "let our developers integrate")),
    Feature("multi_currency", "Multi-currency",
            ("đa tiền tệ", "thanh toán ngoại tệ", "multi-currency", "foreign currency"),
            ("thu tien do va won", "accept payment in USD")),
    Feature("room_booking", "Room booking",
            ("đặt phòng", "quản lý phòng", "room booking", "hotel booking"),
            ("cho thue phong theo gio", "rent rooms by the hour")),
    Feature("appointment_booking", "Appointment scheduling",
            ("đặt lịch hẹn", "lịch hẹn spa", "appointment booking", "online booking"),
            ("khach dat lich cat toc truoc", "customers pre-book a slot")),
    Feature("bin_locations", "Warehouse bin locations",
            ("vị trí kho", "quản lý kệ hàng", "bin locations", "warehouse locations"),
            ("biet hang nam o ke nao", "know which shelf an item is on")),
    Feature("tax_reporting", "Tax reporting",
            ("báo cáo thuế", "kê khai thuế", "tax report", "tax declaration"),
            ("so lieu nop thue cuoi quy", "quarterly tax filing numbers")),
    Feature("sms_marketing", "SMS / chat marketing",
            ("gửi tin nhắn chăm sóc khách", "SMS marketing", "chat marketing", "tin nhắn quảng cáo"),
            ("nhan tin bao uu dai cho khach cu", "message past customers about offers")),
    Feature("label_printing", "Custom label printing",
            ("in tem nhãn", "in mã vạch", "label printing", "barcode labels"),
            ("tu thiet ke tem dan san pham", "design their own sticker labels")),
)

FEATURE_BY_ID = {f.id: f for f in FEATURES}


def strip_accents(text: str) -> str:
    """Lower-case and remove Vietnamese diacritics (đ -> d)."""
    text = text.lower().replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def clean(text: str) -> str:
    text = strip_accents(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TaxonomyNormalizer:
    """Map free-text feature phrases to canonical ids with char n-gram TF-IDF cosine."""

    def __init__(self, features=FEATURES, threshold: float = 0.45, ngram_range=(2, 4)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.threshold = threshold
        self._labels, docs = [], []
        for f in features:
            for a in (f.name, f.id.replace("_", " "), *f.aliases):
                self._labels.append(f.id)
                docs.append(clean(a))
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)
        self._matrix = self._vec.fit_transform(docs)

    def scores(self, phrases: list[str]) -> np.ndarray:
        """Cosine similarity (rows = phrases, cols = alias rows)."""
        q = self._vec.transform([clean(p) for p in phrases])
        return (q @ self._matrix.T).toarray()  # rows are L2-normalised -> dot == cosine

    def match(self, phrase: str) -> tuple[str | None, float]:
        if not clean(phrase):
            return None, 0.0
        s = self.scores([phrase])[0]
        j = int(s.argmax())
        return (self._labels[j], float(s[j])) if s[j] >= self.threshold else (None, float(s[j]))

    def normalize(self, phrases: list[str]) -> list[str]:
        """Canonical ids for a list of phrases, de-duplicated, order preserved."""
        out: list[str] = []
        for p in phrases:
            fid, _ = self.match(p)
            if fid and fid not in out:
                out.append(fid)
        return out


@lru_cache(maxsize=1)
def default_normalizer() -> TaxonomyNormalizer:
    return TaxonomyNormalizer()
