import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.api.recommender_service import (
    get_user_candidates,
    score_candidates_with_ranker,
    build_recommendation_response,
    recommend_for_user,
)


def _make_tiny_model(feature_cols=("feat1", "feat2")):
    """Train a tiny logistic regression with known columns for testing."""
    X = pd.DataFrame({c: [0.1, 0.9, 0.2, 0.8] for c in feature_cols})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model


# --------------------------------------------------------------------------- #
# get_user_candidates
# --------------------------------------------------------------------------- #
def test_get_user_candidates_known_user():
    cand_cache = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 20]})
    pop_cache = pd.DataFrame({"item_id": [30, 40]})
    cands, fallback = get_user_candidates(1, cand_cache, pop_cache)
    assert not fallback
    assert len(cands) == 2
    assert list(cands["item_id"]) == [10, 20]


def test_get_user_candidates_unknown_user_uses_popularity():
    cand_cache = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 20]})
    pop_cache = pd.DataFrame({"item_id": [30, 40]})
    cands, fallback = get_user_candidates(2, cand_cache, pop_cache)
    assert fallback
    assert len(cands) == 2
    assert list(cands["item_id"]) == [30, 40]
    assert list(cands["user_id"]) == [2, 2]


# --------------------------------------------------------------------------- #
# score_candidates_with_ranker
# --------------------------------------------------------------------------- #
def test_score_candidates_with_ranker_success():
    model = _make_tiny_model(("feat1", "feat2"))
    cands = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 20]})
    user_feats = pd.DataFrame({"user_id": [1], "feat2": [0.5]})
    item_feats = pd.DataFrame({"item_id": [10, 20], "feat1": [0.3, 0.7]})

    scored, warnings = score_candidates_with_ranker(
        candidates=cands,
        ranker_model=model,
        user_features=user_feats,
        item_features=item_feats,
        feature_columns=["feat1", "feat2"],
        fill_values={"feat1": 0.0, "feat2": 0.0},
    )
    assert not scored.empty
    assert "ranking_score" in scored.columns
    assert all(0.0 <= s <= 1.0 for s in scored["ranking_score"])


def test_score_candidates_with_no_model_returns_empty_with_warning():
    cands = pd.DataFrame({"user_id": [1], "item_id": [10]})
    scored, warnings = score_candidates_with_ranker(
        candidates=cands,
        ranker_model=None,
        user_features=None,
        item_features=None,
        feature_columns=None,
        fill_values=None,
    )
    assert scored.empty
    assert any("not loaded" in w for w in warnings)


def test_score_candidates_ranker_fails_returns_warning():
    """Model that raises on predict_proba → empty df + warning."""
    class BrokenModel:
        def predict_proba(self, X):
            raise ValueError("intentional test error")

    cands = pd.DataFrame({"user_id": [1], "item_id": [10]})
    scored, warnings = score_candidates_with_ranker(
        candidates=cands,
        ranker_model=BrokenModel(),
        user_features=None,
        item_features=None,
        feature_columns=["feat1"],
        fill_values={"feat1": 0.0},
    )
    assert scored.empty
    assert any("failed" in w.lower() or "error" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# build_recommendation_response
# --------------------------------------------------------------------------- #
def test_build_recommendation_response_shape():
    df = pd.DataFrame({"item_id": [10], "ranking_score": [0.9]})
    res = build_recommendation_response(1, df, 10, "v1", False, scoring_mode="ranker")
    assert "request_id" in res
    assert res["user_id"] == 1
    assert len(res["items"]) == 1
    assert res["model_version"] == "v1"
    assert res["fallback_used"] is False
    assert res["scoring_mode"] == "ranker"
    assert "warnings" in res


# --------------------------------------------------------------------------- #
# recommend_for_user end-to-end
# --------------------------------------------------------------------------- #
def test_recommend_for_user_retrieval_only():
    config = {
        "recommendation": {
            "max_k": 50,
            "user_id_column": "user_id",
            "item_id_column": "item_id",
            "model_version": "v1",
        }
    }
    artifacts = {
        "candidate_cache": pd.DataFrame({
            "user_id": [1], "item_id": [10], "retrieval_score": [0.5]
        })
    }
    res = recommend_for_user(1, 10, artifacts, config)
    assert res["scoring_mode"] == "retrieval_only"
    # warning should explain why
    assert len(res["warnings"]) > 0


def test_recommend_for_user_respects_k():
    config = {"recommendation": {"max_k": 50}}
    artifacts = {"popularity_cache": pd.DataFrame({"item_id": list(range(100))})}
    res = recommend_for_user(1, 5, artifacts, config)
    assert len(res["items"]) == 5


def test_recommend_for_user_uses_ranker_when_available():
    model = _make_tiny_model(("feat1", "feat2"))
    config = {
        "recommendation": {
            "max_k": 50,
            "user_id_column": "user_id",
            "item_id_column": "item_id",
            "score_column": "ranking_score",
            "fallback_score_column": "retrieval_score",
            "model_version": "test_v1",
        }
    }
    artifacts = {
        "candidate_cache": pd.DataFrame({
            "user_id": [1, 1],
            "item_id": [10, 20],
            "retrieval_score": [0.8, 0.7],
            "rank": [1, 2],
        }),
        "user_features": pd.DataFrame({"user_id": [1], "feat2": [0.5]}),
        "item_features": pd.DataFrame({"item_id": [10, 20], "feat1": [0.3, 0.7]}),
        "ranker_model": model,
        "ranker_feature_columns": ["feat1", "feat2"],
        "ranker_fill_values": {"feat1": 0.5, "feat2": 0.5},
    }
    res = recommend_for_user(1, 10, artifacts, config)
    assert res["scoring_mode"] == "ranker"
    assert res["fallback_used"] is False
    assert len(res["items"]) == 2


def test_recommend_for_user_falls_back_with_warning_when_ranker_fails():
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("deliberate failure")

    config = {
        "recommendation": {
            "max_k": 50,
            "user_id_column": "user_id",
            "item_id_column": "item_id",
            "score_column": "ranking_score",
            "fallback_score_column": "retrieval_score",
            "model_version": "v1",
        }
    }
    artifacts = {
        "candidate_cache": pd.DataFrame({
            "user_id": [1], "item_id": [10], "retrieval_score": [0.5]
        }),
        "ranker_model": BrokenModel(),
        "ranker_feature_columns": ["feat1"],
        "ranker_fill_values": {"feat1": 0.0},
    }
    res = recommend_for_user(1, 10, artifacts, config)
    assert res["scoring_mode"] == "retrieval_only"
    assert len(res["warnings"]) > 0
