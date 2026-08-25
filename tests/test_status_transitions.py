"""Tests for run status transition validation."""

import pytest
import requests

from src.models import RunStatus, parse_run_status, validate_status_transition


def test_declared_transitions_allow_expected_moves():
    assert parse_run_status("running") is RunStatus.RUNNING
    validate_status_transition(RunStatus.RUNNING, RunStatus.COMPLETED)
    validate_status_transition(RunStatus.RUNNING, RunStatus.FAILED)
    validate_status_transition(RunStatus.RUNNING, RunStatus.ABORTED)
    validate_status_transition(RunStatus.FAILED, RunStatus.RUNNING)
    validate_status_transition(RunStatus.ABORTED, RunStatus.RUNNING)
    # restating the current status stays legal
    validate_status_transition(RunStatus.RUNNING, RunStatus.RUNNING)


def test_validate_status_transition_rejects_illegal_moves():
    with pytest.raises(ValueError, match="cannot move from 'completed'"):
        validate_status_transition(RunStatus.COMPLETED, RunStatus.RUNNING)
    with pytest.raises(ValueError, match="cannot move from 'aborted'"):
        validate_status_transition(RunStatus.ABORTED, RunStatus.COMPLETED)
    with pytest.raises(ValueError, match="unknown run status"):
        parse_run_status("finshed")


def test_api_rejects_illegal_status_transition_with_conflict(api):
    experiment = api.post("/experiments/", json={"name": "lifecycle"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()

    finished = api.patch(f"/runs/{run['id']}", json={"status": "completed"})
    assert finished.status_code == 200
    body = finished.json()
    assert body["status"] == "completed"
    assert body["finished_at"]

    conflict = api.patch(f"/runs/{run['id']}", json={"status": "failed"})
    assert conflict.status_code == 409
    assert "cannot move from 'completed'" in conflict.json()["detail"]


def test_api_patch_rejects_unknown_status_values(api):
    experiment = api.post("/experiments/", json={"name": "lifecycle"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()

    response = api.patch(f"/runs/{run['id']}", json={"status": "finshed"})
    assert response.status_code == 400
    assert "unknown run status" in response.json()["detail"]


def test_api_allows_failed_runs_to_be_retried(api):
    experiment = api.post("/experiments/", json={"name": "lifecycle"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()

    failed = api.patch(
        f"/runs/{run['id']}", json={"status": "failed", "error": "OOM on epoch 3"}
    )
    assert failed.status_code == 200
    assert failed.json()["finished_at"]

    retried = api.patch(f"/runs/{run['id']}", json={"status": "running"})
    assert retried.status_code == 200
    assert retried.json()["status"] == "running"


def test_client_set_run_status_surfaces_conflicts(tracker):
    experiment = tracker.create_experiment("lifecycle")
    run = tracker.create_run(experiment["id"], "trainer")

    aborted = tracker.set_run_status(run["id"], "aborted")
    assert aborted["status"] == "aborted"

    with pytest.raises(requests.HTTPError) as excinfo:
        tracker.set_run_status(run["id"], "completed")
    assert excinfo.value.response.status_code == 409

    restarted = tracker.set_run_status(run["id"], "running")
    assert restarted["status"] == "running"
