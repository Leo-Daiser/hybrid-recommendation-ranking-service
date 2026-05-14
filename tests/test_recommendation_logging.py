"""Tests verifying /recommend endpoint calls DB logging and survives failures."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from src.api.main import app
import pandas as pd


def _make_tiny_model():
    X = pd.DataFrame({"feat1": [0.1, 0.9, 0.2, 0.8], "feat2": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model


@pytest.fixture
def client_with_artifacts():
    with TestClient(app) as client:
        app.state.api_config = {
            "recommendation": {
                "default_k": 10,
                "max_k": 50,
                "user_id_column": "user_id",
                "item_id_column": "item_id",
                "score_column": "ranking_score",
                "fallback_score_column": "retrieval_score",
                "model_version": "test_v1",
            }
        }
        app.state.recommender_artifacts = {
            "candidate_cache": pd.DataFrame({
                "user_id": [1, 1],
                "item_id": [10, 20],
                "retrieval_score": [0.8, 0.7],
                "rank": [1, 2],
            }),
            "popularity_cache": pd.DataFrame({
                "item_id": [30, 40],
                "retrieval_score": [0.9, 0.8],
            }),
            "user_features": pd.DataFrame({"user_id": [1], "feat2": [0.5]}),
            "item_features": pd.DataFrame({"item_id": [10, 20], "feat1": [0.3, 0.7]}),
            "ranker_model": _make_tiny_model(),
            "ranker_feature_columns": ["feat1", "feat2"],
            "ranker_fill_values": {"feat1": 0.5, "feat2": 0.5},
        }
        yield client


def test_recommend_endpoint_calls_logging_repository(client_with_artifacts):
    with patch("src.api.routes.session_scope") as mock_scope, \
         patch("src.api.routes.log_recommendation_request") as mock_req, \
         patch("src.api.routes.log_ranked_recommendations") as mock_recs:

        session_mock = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=session_mock)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        resp = client_with_artifacts.get("/recommend?user_id=1&k=2")
        assert resp.status_code == 200

        mock_req.assert_called_once()
        mock_recs.assert_called_once()

        # Verify the request_id passed to both is consistent
        req_call_kwargs = mock_req.call_args.kwargs
        rec_call_kwargs = mock_recs.call_args.kwargs
        assert req_call_kwargs["request_id"] == rec_call_kwargs["request_id"]
        assert req_call_kwargs["user_id"] == 1
        assert req_call_kwargs["k"] == 2


def test_recommend_endpoint_survives_logging_failure(client_with_artifacts):
    with patch("src.api.routes.session_scope") as mock_scope:
        mock_scope.return_value.__enter__ = MagicMock(
            side_effect=Exception("DB connection refused")
        )
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        resp = client_with_artifacts.get("/recommend?user_id=1&k=2")
        # Recommendation must still succeed
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) > 0

        # Warning must be present
        warnings = data.get("warnings", [])
        assert any("db_logging_failed" in w for w in warnings)


def test_recommend_endpoint_returns_warnings_field(client_with_artifacts):
    resp = client_with_artifacts.get("/recommend?user_id=1&k=2")
    assert resp.status_code == 200
    data = resp.json()
    assert "warnings" in data
    assert isinstance(data["warnings"], list)
