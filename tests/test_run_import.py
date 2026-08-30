"""Tests for importing several runs from explicit specs in one call."""

import tempfile

import pytest

from src.models import Experiment
from src.storage import LocalStorageBackend


def seed_storage(storage):
    experiment = Experiment(name="imported")
    storage.save_experiment(experiment.to_dict())
    return experiment


def test_storage_imports_runs_with_fresh_ids_and_configuration():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_storage(storage)

        created = storage.import_runs(
            experiment.id,
            [
                {"name": "alpha", "params": {"lr": 0.01}, "tags": {"split": "cv"}},
                {"name": "beta", "params": {"lr": 0.1}},
            ],
        )

        assert [run["name"] for run in created] == ["alpha", "beta"]
        assert len({run["id"] for run in created}) == 2
        assert created[0]["experiment_id"] == experiment.id
        assert created[0]["params"] == {"lr": 0.01}
        assert created[0]["tags"] == {"split": "cv"}
        assert created[1]["params"] == {"lr": 0.1}
        assert created[1]["tags"] == {}
        for run in created:
            assert run["status"] == "running"
            assert run["metrics"] == []
            assert storage.load_run(run["id"]) == run


def test_storage_import_rejects_missing_experiment_and_bad_specs():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_storage(storage)

        with pytest.raises(KeyError, match="experiment not found"):
            storage.import_runs("missing", [{"name": "alpha"}])
        with pytest.raises(ValueError, match="non-empty"):
            storage.import_runs(experiment.id, [])
        with pytest.raises(ValueError, match="non-empty name"):
            storage.import_runs(experiment.id, [{"name": "  "}])
        with pytest.raises(ValueError, match="mapping"):
            storage.import_runs(experiment.id, [{"name": "alpha", "params": "oops"}])


def test_storage_import_caps_the_batch_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_storage(storage)

        with pytest.raises(ValueError, match="1000"):
            storage.import_runs(experiment.id, [{"name": f"r{i}"} for i in range(1001)])


def test_api_imports_runs_in_one_request(api):
    experiment = api.post("/experiments/", json={"name": "imported"}).json()
    response = api.post(
        f"/experiments/{experiment['id']}/runs/import",
        json={"runs": [{"name": "alpha", "tags": {"split": "cv"}}, {"name": "beta"}]},
    )
    assert response.status_code == 200
    created = response.json()
    assert [run["name"] for run in created] == ["alpha", "beta"]
    assert created[0]["tags"] == {"split": "cv"}

    listed = api.get(f"/experiments/{experiment['id']}/runs/").json()
    assert {run["name"] for run in listed} == {"alpha", "beta"}


def test_api_import_rejects_missing_experiment_and_empty_payload(api):
    missing = api.post("/experiments/x/runs/import", json={"runs": [{"name": "alpha"}]})
    assert missing.status_code == 404
    experiment = api.post("/experiments/", json={"name": "imported"}).json()
    empty = api.post(f"/experiments/{experiment['id']}/runs/import", json={"runs": []})
    assert empty.status_code == 422
    bad = api.post(
        f"/experiments/{experiment['id']}/runs/import",
        json={"runs": [{"name": "alpha", "tags": "nope"}]},
    )
    assert bad.status_code == 400 and "mapping" in bad.json()["detail"]


def test_client_imports_multiple_runs(tracker):
    experiment = tracker.create_experiment("imported")
    created = tracker.import_runs(
        experiment["id"],
        [
            {"name": "alpha", "params": {"lr": 0.01}, "tags": {"split": "cv"}},
            {"name": "beta"},
        ],
    )

    assert [run["name"] for run in created] == ["alpha", "beta"]
    assert created[0]["params"] == {"lr": 0.01}
    assert created[0]["tags"] == {"split": "cv"}
    assert len({run["id"] for run in created}) == 2

    listed = tracker.list_runs(experiment["id"])
    assert {run["name"] for run in listed} == {"alpha", "beta"}
