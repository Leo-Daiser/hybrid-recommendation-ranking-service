import pytest
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
def test_client():
    with TestClient(app) as client:
        # Inject synthetic state AFTER lifespan (which loads real files if they exist)
        app.state.api_config = {
            "api": {"service_name": "test", "version": "test"},
            "recommendation": {
                "default_k": 10,
                "max_k": 50,
                "user_id_column": "user_id",
                "item_id_column": "item_id",
                "score_column": "ranking_score",
                "fallback_score_column": "retrieval_score",
                "model_version": "test_v1",
            },
        }
        app.state.recommender_artifacts = {
            "candidate_cache": pd.DataFrame({
                "user_id": [1, 1],
                "item_id": [10, 20],
                "retrieval_score": [0.8, 0.7],
                "rank": [1, 2],
                "retrieval_model": ["itemknn", "itemknn"],
            }),
            "popularity_cache": pd.DataFrame({
                "user_id": [0, 0],
                "item_id": [30, 40],
                "retrieval_score": [0.9, 0.8],
                "rank": [1, 2],
                "retrieval_model": ["pop", "pop"],
            }),
            "user_features": pd.DataFrame({
                "user_id": [1],
                "feat2": [0.5],
            }),
            "item_features": pd.DataFrame({
                "item_id": [10, 20, 30, 40],
                "feat1": [0.3, 0.7, 0.5, 0.4],
            }),
            "ranker_model": _make_tiny_model(),
            "ranker_feature_columns": ["feat1", "feat2"],
            "ranker_fill_values": {"feat1": 0.5, "feat2": 0.5},
        }
        yield client


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_endpoint_known_user(test_client):
    response = test_client.get("/recommend?user_id=1&k=2")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["fallback_used"] is False
    assert len(data["items"]) == 2
    assert data["items"][0]["item_id"] in [10, 20]


def test_recommend_endpoint_known_user_ranker_mode(test_client):
    response = test_client.get("/recommend?user_id=1&k=2")
    assert response.status_code == 200
    data = response.json()
    assert data["scoring_mode"] == "ranker", (
        f"Expected 'ranker' but got '{data['scoring_mode']}'. "
        f"Warnings: {data.get('warnings')}"
    )


def test_recommend_endpoint_invalid_k(test_client):
    response = test_client.get("/recommend?user_id=1&k=0")
    assert response.status_code == 422


def test_recommend_endpoint_unknown_user_fallback(test_client):
    response = test_client.get("/recommend?user_id=999&k=2")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 999
    assert data["fallback_used"] is True
    assert len(data["items"]) == 2
    assert data["items"][0]["item_id"] in [30, 40]

def test_recommend_response_contains_warnings_field(test_client):
    response = test_client.get("/recommend?user_id=1&k=2")
    assert response.status_code == 200
    data = response.json()
    assert "warnings" in data
    assert isinstance(data["warnings"], list)
