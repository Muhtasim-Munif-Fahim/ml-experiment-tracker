"""Tests for the run status-transition audit log (status_history)."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run, RunStatus
from src.storage import LocalStorageBackend


def _create_run(api):
    experiment = api.post("/experiments/", json={"name": "lifecycle"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()
    return experiment, run


# running -> failed -> running -> completed is a fully legal lifecycle
_VALID_CHAIN = ["failed", "running", "completed"]


def test_model_run_round_trips_status_history():
    run = Run(experiment_id="exp", name="run")
    run.status_history = [
        {"from": "running", "to": "failed", "error": "OOM", "timestamp": "2026-01-01T00:00:00+00:00"}
    ]
    restored = Run.from_dict(run.to_dict())
    assert restored.status_history == run.status_history
    assert restored.status_history is not run.status_history


def test_new_run_has_no_status_history(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="fresh")
    storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r")
    storage.save_run(run.to_dict())
    loaded = storage.load_run(run.id)
    assert loaded.get("status_history") == []


def test_patch_records_history_on_real_transitions(api):
    _, run = _create_run(api)
    history = []
    prev = None
    for status, error in zip(_VALID_CHAIN, ["OOM on epoch 3", None, None]):
        payload = {"status": status}
        if error:
            payload["error"] = error
        response = api.patch(f"/runs/{run['id']}", json=payload)
        assert response.status_code == 200
        history = response.json().get("status_history", [])
    assert [entry["to"] for entry in history] == _VALID_CHAIN
    assert [entry["from"] for entry in history] == ["running", "failed", "running"]
    assert history[0]["error"] == "OOM on epoch 3"
    assert history[0]["timestamp"]
    assert history[1]["error"] is None


def test_patch_no_op_transition_does_not_record_history(api):
    _, run = _create_run(api)
    before = api.patch(f"/runs/{run['id']}", json={"status": "running"})
    assert before.status_code == 200
    assert before.json().get("status_history") == []


def test_patch_illegal_transition_rejects_and_appends_nothing(api):
    _, run = _create_run(api)
    api.patch(f"/runs/{run['id']}", json={"status": "completed"})
    conflict = api.patch(f"/runs/{run['id']}", json={"status": "failed"})
    assert conflict.status_code == 409
    after = api.get(f"/runs/{run['id']}").json()
    assert [entry["to"] for entry in after.get("status_history", [])] == ["completed"]


def test_patch_unknown_status_rejects_with_400(api):
    _, run = _create_run(api)
    response = api.patch(f"/runs/{run['id']}", json={"status": "finshed"})
    assert response.status_code == 400
    assert api.get(f"/runs/{run['id']}").json().get("status_history") == []


def test_status_history_endpoint_returns_audit_log(api):
    _, run = _create_run(api)
    for status in ("failed", "running"):
        api.patch(f"/runs/{run['id']}", json={"status": status})

    response = api.get(f"/runs/{run['id']}/status-history")
    assert response.status_code == 200
    history = response.json()
    assert [entry["to"] for entry in history] == ["failed", "running"]
    assert [entry["from"] for entry in history] == ["running", "failed"]


def test_status_history_endpoint_404_for_missing_run(api):
    response = api.get("/runs/does-not-exist/status-history")
    assert response.status_code == 404


def test_status_history_endpoint_defaults_to_empty_for_legacy_run(api):
    experiment = api.post("/experiments/", json={"name": "legacy"}).json()
    raw_run = {
        "id": "legacy-run-id",
        "experiment_id": experiment["id"],
        "name": "legacy",
        "status": "completed",
        "params": {},
        "metrics": [],
        "artifacts": [],
        "tags": {},
        "alerts": [],
        "notes": [],
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:00+00:00",
        "error": None,
    }
    import src.api as api_module

    api_module.storage.save_run(raw_run)

    response = api.get("/runs/legacy-run-id/status-history")
    assert response.status_code == 200
    assert response.json() == []


def test_client_status_history(tracker):
    experiment = tracker.create_experiment("client-history")
    run = tracker.create_run(experiment["id"], "trainer")
    for status, error in zip(_VALID_CHAIN, ["OOM on epoch 3", None, None]):
        tracker.set_run_status(run["id"], status, error=error if error else None)

    history = tracker.run_status_history(run["id"])
    assert [entry["to"] for entry in history] == _VALID_CHAIN
    assert [entry["from"] for entry in history] == ["running", "failed", "running"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
