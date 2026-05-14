"""Tests for POST /feedback endpoint — DB calls are mocked."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _feedback_payload(**kwargs):
    base = {
        "user_id": 1,
        "item_id": 10,
        "event_type": "click",
    }
    base.update(kwargs)
    return base


def test_feedback_endpoint_success_with_mocked_db(client):
    with patch("src.api.routes.session_scope") as mock_scope, \
         patch("src.api.routes.log_feedback_event") as mock_log:
        mock_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.post("/feedback", json=_feedback_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert "feedback_id" in data
        assert data["status"] == "logged"


def test_feedback_endpoint_invalid_event_type(client):
    resp = client.post("/feedback", json=_feedback_payload(event_type="purchase"))
    assert resp.status_code == 422


def test_feedback_endpoint_with_optional_fields(client):
    with patch("src.api.routes.session_scope") as mock_scope, \
         patch("src.api.routes.log_feedback_event"):
        mock_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        payload = _feedback_payload(
            request_id="some-request-id",
            event_value=1.0,
            metadata={"source": "banner"},
        )
        resp = client.post("/feedback", json=payload)
        assert resp.status_code == 200


def test_feedback_endpoint_db_failure_returns_503(client):
    with patch("src.api.routes.session_scope") as mock_scope:
        mock_scope.return_value.__enter__ = MagicMock(side_effect=Exception("DB down"))
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.post("/feedback", json=_feedback_payload())
        assert resp.status_code == 503
        assert "DB unavailable" in resp.json()["detail"]


def test_feedback_all_valid_event_types(client):
    for et in ("impression", "click", "like", "dislike", "skip"):
        with patch("src.api.routes.session_scope") as mock_scope, \
             patch("src.api.routes.log_feedback_event"):
            mock_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)
            resp = client.post("/feedback", json=_feedback_payload(event_type=et))
            assert resp.status_code == 200, f"Failed for event_type={et}"
