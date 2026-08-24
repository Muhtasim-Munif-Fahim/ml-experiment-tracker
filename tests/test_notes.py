"""Tests for run annotation notes."""

import tempfile

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def seed_run(storage) -> Run:
    experiment = Experiment(name="noted")
    run = Run(experiment_id=experiment.id, name="trainer")
    storage.save_experiment(experiment.to_dict())
    storage.save_run(run.to_dict())
    return run


def test_add_and_list_run_notes_preserves_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run(storage)

        first = storage.add_note(run.id, "# baseline\nlow lr")
        second = storage.add_note(run.id, "candidate looks better")

        assert first["body"] == "# baseline\nlow lr"
        assert first["id"] and first["created_at"]
        assert [note["id"] for note in storage.list_notes(run.id)] == [
            first["id"],
            second["id"],
        ]
        # notes ride along in the stored run record
        stored = storage.load_run(run.id)["notes"]
        assert [note["body"] for note in stored] == [
            "# baseline\nlow lr",
            "candidate looks better",
        ]


def test_update_and_delete_run_notes():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run(storage)
        note = storage.add_note(run.id, "draft")

        updated = storage.update_note(run.id, note["id"], "reviewed: keep")
        assert updated["body"] == "reviewed: keep"
        assert storage.load_run(run.id)["notes"][0]["body"] == "reviewed: keep"

        assert storage.delete_note(run.id, note["id"]) is True
        assert storage.list_notes(run.id) == []
        assert storage.delete_note(run.id, note["id"]) is False
        assert storage.update_note(run.id, note["id"], "gone") is None


def test_notes_validate_body_and_run_existence():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run(storage)

        for body in ("", "   ", None):
            with pytest.raises(ValueError, match="non-empty"):
                storage.add_note(run.id, body)
        with pytest.raises(KeyError, match="run not found"):
            storage.add_note("missing", "hello")


def test_run_model_round_trips_notes():
    run = Run(experiment_id="exp", name="trainer")
    run.notes.append({"id": "note_1", "body": "keep", "created_at": "2026-08-24T00:00:00"})
    restored = Run.from_dict(run.to_dict())
    assert restored.notes[0]["body"] == "keep"
    bare = Run(experiment_id="exp", name="bare")
    assert Run.from_dict(bare.to_dict()).notes == []


def test_api_note_crud_included_in_run_detail(api):
    experiment = api.post("/experiments/", json={"name": "noted"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()

    created = api.post(f"/runs/{run['id']}/notes", json={"body": "# baseline"})
    assert created.status_code == 200
    note = created.json()

    assert api.get(f"/runs/{run['id']}/notes").json() == [note]

    updated = api.put(f"/runs/{run['id']}/notes/{note['id']}", json={"body": "**final**"})
    assert updated.status_code == 200
    detail = api.get(f"/runs/{run['id']}").json()
    assert detail["notes"] == [{**note, "body": "**final**"}]

    assert api.delete(f"/runs/{run['id']}/notes/{note['id']}").json()["message"] == (
        "Note deleted"
    )
    assert api.get(f"/runs/{run['id']}/notes").json() == []


def test_api_note_endpoints_reject_unknown_and_invalid_input(api):
    experiment = api.post("/experiments/", json={"name": "noted"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()

    assert (
        api.post("/runs/missing/notes", json={"body": "hi"}).status_code == 404
    )
    assert api.get("/runs/missing/notes").status_code == 404
    empty = api.post(f"/runs/{run['id']}/notes", json={"body": "  "})
    assert empty.status_code == 400 and "non-empty" in empty.json()["detail"]
    assert api.put(f"/runs/{run['id']}/notes/nope", json={"body": "x"}).status_code == 404
    assert api.delete(f"/runs/{run['id']}/notes/nope").status_code == 404
    assert (
        api.put("/runs/missing/notes/nope", json={"body": "x"}).status_code == 404
    )


def test_client_note_crud(tracker):
    experiment = tracker.create_experiment("noted")
    run = tracker.create_run(experiment["id"], "trainer")

    note = tracker.create_note(run["id"], "# baseline\nlr=0.1")
    assert [item["id"] for item in tracker.list_notes(run["id"])] == [note["id"]]

    tracker.update_note(run["id"], note["id"], "**tuned**")
    detail = tracker.get_run(run["id"])
    assert detail["notes"][0]["body"] == "**tuned**"

    tracker.delete_note(run["id"], note["id"])
    assert tracker.list_notes(run["id"]) == []
