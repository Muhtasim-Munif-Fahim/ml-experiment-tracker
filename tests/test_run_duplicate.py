"""Tests for duplicating a run into a fresh copy with a new id."""

import tempfile

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def seed_storage(storage):
    experiment = Experiment(name="dupe")
    source = Run(
        experiment_id=experiment.id,
        name="baseline",
        params={"lr": 0.01},
        tags={"stage": "base"},
    )
    source.log_metric("loss", 0.5, step=1)
    source.log_metric("loss", 0.3, step=2)
    storage.save_experiment(experiment.to_dict())
    storage.save_run(source.to_dict())
    return experiment, source


def test_storage_duplicate_copies_configuration_into_fresh_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment, source = seed_storage(storage)

        copy = storage.duplicate_run(source.id)

        assert copy["id"] != source.id
        assert copy["experiment_id"] == experiment.id
        assert copy["parent_run_id"] == source.id
        assert copy["name"] == "baseline (copy)"
        assert copy["params"] == {"lr": 0.01}
        assert copy["tags"] == {"stage": "base"}
        assert copy["status"] == "running"
        assert copy["metrics"] == []
        assert storage.load_run(copy["id"]) == copy
        assert storage.load_run(source.id)["id"] == source.id


def test_storage_duplicate_optionally_copies_metric_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        _, source = seed_storage(storage)

        copy = storage.duplicate_run(source.id, name="rerun", include_metrics=True)

        assert copy["name"] == "rerun"
        assert [(metric["name"], metric["value"], metric["step"]) for metric in copy["metrics"]] == [
            ("loss", 0.5, 1),
            ("loss", 0.3, 2),
        ]


def test_storage_duplicate_rejects_missing_run_and_non_boolean_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        with pytest.raises(KeyError, match="run not found"):
            storage.duplicate_run("missing")
        _, source = seed_storage(storage)
        with pytest.raises(ValueError, match="boolean"):
            storage.duplicate_run(source.id, include_metrics="yes")


def test_api_duplicate_run_creates_fresh_copy(api):
    experiment = api.post("/experiments/", json={"name": "dupe"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/",
        json={"name": "baseline", "params": {"lr": 0.01}, "tags": {"stage": "base"}},
    ).json()
    api.post(f"/runs/{run['id']}/metrics", json={"name": "loss", "value": 0.5, "step": 1})

    response = api.post(f"/runs/{run['id']}/duplicate", json={"include_metrics": True})
    assert response.status_code == 200
    copy = response.json()

    assert copy["id"] != run["id"]
    assert copy["parent_run_id"] == run["id"]
    assert copy["params"] == {"lr": 0.01}
    assert copy["tags"] == {"stage": "base"}
    assert [metric["value"] for metric in copy["metrics"]] == [0.5]
    assert api.get(f"/runs/{run['id']}").json()["id"] == run["id"]


def test_api_duplicate_defaults_to_config_only_and_rejects_missing(api):
    experiment = api.post("/experiments/", json={"name": "dupe"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "baseline"}
    ).json()
    api.post(f"/runs/{run['id']}/metrics", json={"name": "loss", "value": 0.5})

    copy = api.post(f"/runs/{run['id']}/duplicate", json={}).json()
    assert copy["metrics"] == []

    assert api.post("/runs/missing/duplicate", json={}).status_code == 404


def test_client_duplicates_run_with_optional_metric_history(tracker):
    experiment = tracker.create_experiment("dupe")
    run = tracker.create_run(
        experiment["id"], "baseline", params={"lr": 0.01}, tags={"stage": "base"}
    )
    tracker.log_metric(run["id"], "loss", 0.5, step=1)
    tracker.log_metric(run["id"], "loss", 0.3, step=2)

    copy = tracker.duplicate_run(run["id"], name="rerun", include_metrics=True)
    assert copy["id"] != run["id"]
    assert copy["name"] == "rerun"
    assert copy["params"] == {"lr": 0.01}
    assert copy["tags"] == {"stage": "base"}
    assert [(metric["value"], metric["step"]) for metric in copy["metrics"]] == [
        (0.5, 1),
        (0.3, 2),
    ]

    config_only = tracker.duplicate_run(run["id"])
    assert config_only["metrics"] == []
    assert config_only["params"] == {"lr": 0.01}
