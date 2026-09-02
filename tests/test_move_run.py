"""Tests for the move-run feature."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def _seed() -> tuple[LocalStorageBackend, Experiment, Experiment, Run]:
    storage = LocalStorageBackend("/tmp/move-run-fixture")
    source = Experiment(name="source")
    target = Experiment(name="target")
    storage.save_experiment(source.to_dict())
    storage.save_experiment(target.to_dict())

    run = Run(experiment_id=source.id, name="original")
    run.log_metric("accuracy", 0.8)
    storage.save_run(run.to_dict())
    source.add_run(run)
    return storage, source, target, run


def test_move_run_updates_experiment_id() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.storage.Path", lambda *a, **k: None)  # placeholder
    storage, source, target, run = _seed()
    updated = storage.move_run(run.id, target.id)
    assert updated["experiment_id"] == target.id


def test_move_run_drops_source_run_id_and_adds_to_target() -> None:
    storage, source, target, run = _seed()
    storage.move_run(run.id, target.id)
    reloaded_source = storage.load_experiment(source.id)
    reloaded_target = storage.load_experiment(target.id)
    assert all(r["id"] != run.id for r in reloaded_source.get("runs", []))
    assert any(r["id"] == run.id for r in reloaded_target.get("runs", []))


def test_move_run_preserves_metrics_and_params() -> None:
    storage, source, target, run = _seed()
    storage.move_run(run.id, target.id)
    reloaded = storage.load_run(run.id)
    assert reloaded["experiment_id"] == target.id
    assert reloaded["metrics"][0]["name"] == "accuracy"
    assert reloaded["metrics"][0]["value"] == 0.8


def test_move_run_unknown_run_raises() -> None:
    storage, source, target, run = _seed()
    with pytest.raises(KeyError, match="run not found"):
        storage.move_run("does-not-exist", target.id)


def test_move_run_unknown_target_experiment_raises() -> None:
    storage, source, target, run = _seed()
    with pytest.raises(KeyError, match="experiment not found"):
        storage.move_run(run.id, "missing-target")


def test_move_run_same_experiment_raises_value_error() -> None:
    storage, source, target, run = _seed()
    with pytest.raises(ValueError, match="already belongs"):
        storage.move_run(run.id, source.id)


def test_api_move_run_updates_run_and_experiments(api, temp_storage) -> None:
    source = Experiment(name="src-api")
    target = Experiment(name="tgt-api")
    temp_storage.save_experiment(source.to_dict())
    temp_storage.save_experiment(target.to_dict())
    run = Run(experiment_id=source.id, name="runner")
    run.log_metric("loss", 0.42)
    temp_storage.save_run(run.to_dict())
    source.add_run(run)

    response = api.post(
        f"/runs/{run.id}/move",
        json={"target_experiment_id": target.id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"] == target.id

    reloaded_run = temp_storage.load_run(run.id)
    assert reloaded_run["experiment_id"] == target.id

    reloaded_source = temp_storage.load_experiment(source.id)
    reloaded_target = temp_storage.load_experiment(target.id)
    assert all(r["id"] != run.id for r in reloaded_source["runs"])
    assert any(r["id"] == run.id for r in reloaded_target["runs"])


def test_api_move_run_404_for_missing_run(api, temp_storage) -> None:
    target = Experiment(name="tgt-404")
    temp_storage.save_experiment(target.to_dict())
    response = api.post(
        "/runs/missing-run-id/move",
        json={"target_experiment_id": target.id},
    )
    assert response.status_code == 404


def test_api_move_run_404_for_missing_target(api, temp_storage) -> None:
    source = Experiment(name="src-404")
    temp_storage.save_experiment(source.to_dict())
    run = Run(experiment_id=source.id, name="runner")
    temp_storage.save_run(run.to_dict())
    response = api.post(
        f"/runs/{run.id}/move",
        json={"target_experiment_id": "no-such-target"},
    )
    assert response.status_code == 404


def test_api_move_run_409_when_target_is_current_experiment(api, temp_storage) -> None:
    source = Experiment(name="src-409")
    temp_storage.save_experiment(source.to_dict())
    run = Run(experiment_id=source.id, name="runner")
    temp_storage.save_run(run.to_dict())
    response = api.post(
        f"/runs/{run.id}/move",
        json={"target_experiment_id": source.id},
    )
    assert response.status_code == 409


def test_client_move_run_updates_run(live_server, temp_storage) -> None:
    from src.client import ExperimentTrackerClient

    source = Experiment(name="src-client")
    target = Experiment(name="tgt-client")
    temp_storage.save_experiment(source.to_dict())
    temp_storage.save_experiment(target.to_dict())
    run = Run(experiment_id=source.id, name="runner")
    temp_storage.save_run(run.to_dict())

    client = ExperimentTrackerClient(base_url=live_server)
    updated = client.move_run(run.id, target.id)
    assert updated["experiment_id"] == target.id