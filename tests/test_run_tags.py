"""Tests for granular key/value tag management on runs."""

import tempfile

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def seed_run(storage) -> Run:
    experiment = Experiment(name="tagged")
    run = Run(experiment_id=experiment.id, name="trainer", tags={"seed": "42"})
    storage.save_experiment(experiment.to_dict())
    storage.save_run(run.to_dict())
    return run


def test_set_list_and_delete_run_tags():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run(storage)

        assert storage.list_run_tags(run.id) == {"seed": "42"}

        created = storage.set_run_tag(run.id, "framework", "pytorch")
        assert created == {"name": "framework", "value": "pytorch"}
        assert storage.list_run_tags(run.id) == {"seed": "42", "framework": "pytorch"}

        replaced = storage.set_run_tag(run.id, "seed", "7")
        assert replaced == {"name": "seed", "value": "7"}
        assert storage.list_run_tags(run.id) == {"seed": "7", "framework": "pytorch"}

        assert storage.delete_run_tag(run.id, "framework") is True
        assert storage.delete_run_tag(run.id, "framework") is False
        assert storage.list_run_tags(run.id) == {"seed": "7"}


def test_run_tag_validation_and_missing_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run(storage)

        for name, value in (("", "x"), (" ", "x")):
            with pytest.raises(ValueError, match="tag name"):
                storage.set_run_tag(run.id, name, value)
        for value in ("", "   "):
            with pytest.raises(ValueError, match="tag value"):
                storage.set_run_tag(run.id, "key", value)
        with pytest.raises(KeyError, match="run not found"):
            storage.set_run_tag("missing", "key", "value")
        with pytest.raises(KeyError, match="run not found"):
            storage.list_run_tags("missing")
        with pytest.raises(KeyError, match="run not found"):
            storage.delete_run_tag("missing", "key")


def test_run_tags_drive_storage_query_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = Experiment(name="tagged")
        baseline = Run(experiment_id=experiment.id, name="baseline")
        tuned = Run(experiment_id=experiment.id, name="tuned")
        storage.save_experiment(experiment.to_dict())
        storage.save_run(baseline.to_dict())
        storage.save_run(tuned.to_dict())

        storage.set_run_tag(tuned.id, "split", "cv")

        matches = storage.query_runs(
            experiment.id, tags={"split": "cv"}
        )
        assert [run["id"] for run in matches] == [tuned.id]
        assert storage.query_runs(experiment.id, tags={"split": "holdout"}) == []


def test_api_run_tag_crud(api):
    experiment = api.post("/experiments/", json={"name": "tagged"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/",
        json={"name": "trainer", "tags": {"seed": "42"}},
    ).json()

    assert api.get(f"/runs/{run['id']}/tags").json() == {"seed": "42"}

    created = api.put(f"/runs/{run['id']}/tags/framework", json={"value": "pytorch"})
    assert created.status_code == 200
    assert created.json() == {"name": "framework", "value": "pytorch"}

    replaced = api.put(f"/runs/{run['id']}/tags/seed", json={"value": "7"})
    assert replaced.json() == {"name": "seed", "value": "7"}

    detail = api.get(f"/runs/{run['id']}").json()
    assert detail["tags"] == {"seed": "7", "framework": "pytorch"}

    assert api.delete(f"/runs/{run['id']}/tags/framework").json()["message"] == (
        "Tag deleted"
    )
    assert api.get(f"/runs/{run['id']}/tags").json() == {"seed": "7"}


def test_api_run_tag_endpoints_reject_unknown_and_invalid_input(api):
    experiment = api.post("/experiments/", json={"name": "tagged"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()

    assert api.get("/runs/missing/tags").status_code == 404
    assert api.put("/runs/missing/tags/key", json={"value": "v"}).status_code == 404
    assert (
        api.delete("/runs/missing/tags/key").status_code == 404
    )
    assert (
        api.delete(f"/runs/{run['id']}/tags/nope").status_code == 404
    )
    empty = api.put(f"/runs/{run['id']}/tags/key", json={"value": "  "})
    assert empty.status_code == 400 and "non-empty" in empty.json()["detail"]


def test_client_manages_run_tags(tracker):
    experiment = tracker.create_experiment("tagged")
    run = tracker.create_run(experiment["id"], "trainer", tags={"seed": "42"})

    assert tracker.list_run_tags(run["id"]) == {"seed": "42"}

    tracker.set_run_tag(run["id"], "framework", "pytorch")
    tracker.set_run_tag(run["id"], "seed", "7")
    assert tracker.list_run_tags(run["id"]) == {"seed": "7", "framework": "pytorch"}

    detail = tracker.get_run(run["id"])
    assert detail["tags"] == {"seed": "7", "framework": "pytorch"}

    tracker.delete_run_tag(run["id"], "framework")
    assert tracker.list_run_tags(run["id"]) == {"seed": "7"}
