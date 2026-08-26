"""Tests for the optional API token gate."""

import pytest
import requests

from src.client import ExperimentTrackerClient

API_TOKEN_ENV_VAR = "MLTRACKER_API_TOKEN"


def test_api_allows_requests_without_configuration(api, monkeypatch):
    monkeypatch.delenv(API_TOKEN_ENV_VAR, raising=False)
    assert api.get("/experiments/").status_code == 200
    assert api.get("/health").status_code == 200


def test_api_rejects_missing_and_wrong_tokens(api, monkeypatch):
    monkeypatch.setenv(API_TOKEN_ENV_VAR, "secret-token")
    assert api.get("/experiments/").status_code == 401
    wrong = api.get("/experiments/", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Missing or invalid API token"
    malformed = api.get("/experiments/", headers={"Authorization": "secret-token"})
    assert malformed.status_code == 401


def test_api_exempts_health_but_protects_writes(api, monkeypatch):
    monkeypatch.setenv(API_TOKEN_ENV_VAR, "secret-token")
    assert api.get("/health").status_code == 200
    blocked_post = api.post("/experiments/", json={"name": "blocked"})
    assert blocked_post.status_code == 401
    allowed = api.post(
        "/experiments/",
        json={"name": "allowed"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert allowed.status_code == 200


def test_client_token_is_accepted_by_gated_server(live_server, monkeypatch):
    monkeypatch.setenv(API_TOKEN_ENV_VAR, "secret-token")
    authorized = ExperimentTrackerClient(base_url=live_server, api_key="secret-token")
    experiment = authorized.create_experiment("gated", tags=["secure"])
    assert experiment["name"] == "gated"
    assert authorized.list_experiments()[0]["id"] == experiment["id"]

    anonymous = ExperimentTrackerClient(base_url=live_server)
    with pytest.raises(requests.HTTPError) as excinfo:
        anonymous.list_experiments()
    assert excinfo.value.response.status_code == 401