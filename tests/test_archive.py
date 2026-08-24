"""Tests for the experiment archive lifecycle."""

import tempfile

import pytest

from src.models import Experiment
from src.storage import LocalStorageBackend


def seed_two_experiments(storage):
    first = Experiment(name="active")
    second = Experiment(name="finished")
    storage.save_experiment(first.to_dict())
    storage.save_experiment(second.to_dict())
    return first, second


def test_archived_experiments_are_hidden_from_default_listings():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        active, finished = seed_two_experiments(storage)

        archived = storage.set_experiment_archived(finished.id, archived=True)
        assert archived["archived"] is True

        names = [exp["name"] for exp in storage.list_experiments()]
        assert names == ["active"]
        with_archived = {
            exp["name"]: exp["archived"]
            for exp in storage.list_experiments(include_archived=True)
        }
        assert with_archived == {"active": False, "finished": True}


def test_unarchive_restores_experiment_to_default_listing():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        active, finished = seed_two_experiments(storage)

        storage.set_experiment_archived(active.id, archived=True)
        storage.set_experiment_archived(finished.id, archived=True)
        assert storage.list_experiments() == []
        restored = storage.set_experiment_archived(finished.id, archived=False)
        assert restored["archived"] is False
        assert [exp["id"] for exp in storage.list_experiments()] == [finished.id]


def test_set_experiment_archived_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        assert storage.set_experiment_archived("missing") is None


def test_experiment_model_round_trips_archived_flag():
    experiment = Experiment(name="finished", archived=True)
    data = experiment.to_dict()
    assert data["archived"] is True
    assert Experiment.from_dict(data).archived is True
    assert Experiment.from_dict({**data, "archived": False}).archived is False


def test_api_archive_lifecycle_filters_listings(api):
    first = api.post("/experiments/", json={"name": "active"}).json()
    second = api.post("/experiments/", json={"name": "finished"}).json()

    archived = api.post(f"/experiments/{second['id']}/archive")
    assert archived.status_code == 200
    assert archived.json()["archived"] is True

    listed = [exp["id"] for exp in api.get("/experiments/").json()]
    assert listed == [first["id"]]
    with_archived = {
        exp["id"]: exp["archived"]
        for exp in api.get("/experiments/", params={"include_archived": True}).json()
    }
    assert with_archived == {first["id"]: False, second["id"]: True}

    missing = api.post("/experiments/missing/archive")
    assert missing.status_code == 404

    unarchived = api.post(f"/experiments/{second['id']}/unarchive")
    assert unarchived.json()["archived"] is False
    assert {
        exp["id"] for exp in api.get("/experiments/").json()
    } == {first["id"], second["id"]}
    assert api.post("/experiments/missing/unarchive").status_code == 404


def test_client_archive_and_unarchive(tracker):
    experiment = tracker.create_experiment("finished")
    other = tracker.create_experiment("active")

    archived = tracker.archive_experiment(experiment["id"])
    assert archived["archived"] is True
    assert [exp["id"] for exp in tracker.list_experiments()] == [other["id"]]
    assert len(tracker.list_experiments(include_archived=True)) == 2

    tracker.unarchive_experiment(experiment["id"])
    assert {exp["id"] for exp in tracker.list_experiments()} == {
        experiment["id"],
        other["id"],
    }
