"""Supervised Stage-1 baseline: TF-IDF (char + word n-grams) + logistic regression.

Needs labelled data, which in practice means a few hundred hand- or LLM-labelled
notes; here it is trained on a split of the synthetic ground truth.
"""
from __future__ import annotations

from .taxonomy import clean


def build_baseline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                 preprocessor=clean)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 preprocessor=clean)),
    ])
    return Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])


def train_baseline(texts: list[str], labels: list[bool]):
    model = build_baseline()
    model.fit(texts, labels)
    return model
