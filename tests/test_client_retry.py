"""Tests for client retry/backoff on transient server failures."""

import json
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
import requests

from src.client import ExperimentTrackerClient


def fake_response(status_code, payload=None, headers=None, content=None):
    response = requests.Response()
    response.status_code = status_code
    response._content = content if content is not None else json.dumps(payload or {}).encode()
    response.headers.update(headers or {})
    return response


def attach_fake_responses(client, responses, monkeypatch):
    state = {"index": 0}

    def fake_request(method, url, **kwargs):
        response = responses[state["index"]]
        state["index"] += 1
        return response

    monkeypatch.setattr(client.session, "request", fake_request)
    return state


def record_sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(time, "sleep", lambda delay: recorded.append(delay))
    return recorded


def test_client_retries_transient_5xx_with_exponential_backoff(monkeypatch):
    client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.5)
    state = attach_fake_responses(
        client,
        [fake_response(503), fake_response(502), fake_response(200, {"id": "r1"})],
        monkeypatch,
    )
    recorded = record_sleeps(monkeypatch)

    assert client.get_run("r1") == {"id": "r1"}
    assert state["index"] == 3
    assert recorded == [0.5, 1.0]


def test_client_honors_retry_after_seconds(monkeypatch):
    client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.5)
    attach_fake_responses(
        client,
        [
            fake_response(503, headers={"Retry-After": "2"}),
            fake_response(200, {"ok": True}),
        ],
        monkeypatch,
    )
    recorded = record_sleeps(monkeypatch)

    assert client.get_run("r1") == {"ok": True}
    assert recorded == [2.0]


def test_client_honors_retry_after_http_date(monkeypatch):
    past = format_datetime(datetime.now(timezone.utc) - timedelta(seconds=30))
    client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.5)
    attach_fake_responses(
        client,
        [
            fake_response(503, headers={"Retry-After": past}),
            fake_response(200, {"ok": True}),
        ],
        monkeypatch,
    )
    recorded = record_sleeps(monkeypatch)

    assert client.get_run("r1") == {"ok": True}
    assert recorded == [0.0]


def test_client_caps_retry_attempts(monkeypatch):
    client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.0)
    state = attach_fake_responses(
        client,
        [fake_response(503) for _ in range(4)],
        monkeypatch,
    )
    record_sleeps(monkeypatch)

    with pytest.raises(requests.HTTPError):
        client.get_run("r1")
    assert state["index"] == 4


def test_client_does_not_retry_non_transient_responses(monkeypatch):
    client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.5)
    state = attach_fake_responses(client, [fake_response(404)], monkeypatch)
    with pytest.raises(requests.HTTPError):
        client.get_run("missing")
    assert state["index"] == 1

    ok_client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.5)
    ok_state = attach_fake_responses(ok_client, [fake_response(200, {"id": "r"})], monkeypatch)
    assert ok_client.get_run("r") == {"id": "r"}
    assert ok_state["index"] == 1


def test_client_zero_retries_disables_backoff(monkeypatch):
    client = ExperimentTrackerClient(max_retries=0, backoff_factor=0.5)
    state = attach_fake_responses(client, [fake_response(503)], monkeypatch)
    with pytest.raises(requests.HTTPError):
        client.get_run("r1")
    assert state["index"] == 1


def test_client_retry_parameter_validation():
    with pytest.raises(ValueError, match="max_retries"):
        ExperimentTrackerClient(max_retries=11)
    with pytest.raises(ValueError, match="max_retries"):
        ExperimentTrackerClient(max_retries=-1)
    with pytest.raises(ValueError, match="max_retries"):
        ExperimentTrackerClient(max_retries=True)
    with pytest.raises(ValueError, match="backoff_factor"):
        ExperimentTrackerClient(backoff_factor=-0.1)


def test_client_retries_downloads(monkeypatch):
    client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.0)
    state = attach_fake_responses(
        client,
        [
            fake_response(500),
            fake_response(200, content=b"model-bytes"),
        ],
        monkeypatch,
    )
    record_sleeps(monkeypatch)

    assert client.download_artifact("r1", "weights.bin") == b"model-bytes"
    assert state["index"] == 2


def test_client_retries_csv_fetch(monkeypatch):
    csv_client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.0)
    attach_fake_responses(
        csv_client,
        [
            fake_response(502),
            fake_response(200, content=b"rank,run_id\n1,run-a\n"),
        ],
        monkeypatch,
    )
    record_sleeps(monkeypatch)
    assert "run-a" in csv_client.run_leaderboard_csv("exp", "accuracy")

    search_client = ExperimentTrackerClient(max_retries=3, backoff_factor=0.0)
    attach_fake_responses(
        search_client,
        [
            fake_response(502),
            fake_response(200, content=b"experiment_id,run_id\n"),
        ],
        monkeypatch,
    )
    assert search_client.search_runs_csv() == "experiment_id,run_id\n"


def test_client_retry_defaults_are_harmless_on_live_server(tracker):
    experiment = tracker.create_experiment("retry-smoke")
    run = tracker.create_run(experiment["id"], "runner")
    tracker.log_metric(run["id"], "accuracy", 0.9, step=1)

    assert tracker.list_runs(experiment["id"]).total == 1
