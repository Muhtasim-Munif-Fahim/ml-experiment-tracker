"""Tests for export_run_notes_csv."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def _seed_run(tmp_path: Path, *, note_body: str):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="notes-exp")
    storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    storage.save_run(run.to_dict())
    storage.add_note(run.id, note_body)
    return storage, run


def test_export_run_notes_csv_writes_one_row_per_note(tmp_path: Path) -> None:
    storage, run = _seed_run(tmp_path, note_body="first observation")
    storage.add_note(run.id, "second observation")
    destination = tmp_path / "out.csv"
    path = storage.export_run_notes_csv(run.id, str(destination))
    assert path == str(destination)
    rows = list(csv.DictReader(open(destination, encoding="utf-8")))
    assert len(rows) == 2
    bodies = {row["body"] for row in rows}
    assert bodies == {"first observation", "second observation"}
    for row in rows:
        assert row["author"] == ""
        assert row["run_id"] == run.id


def test_export_run_notes_csv_handles_no_notes(tmp_path: Path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="empty")
    storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    storage.save_run(run.to_dict())
    destination = tmp_path / "empty.csv"
    path = storage.export_run_notes_csv(run.id, str(destination))
    rows = list(csv.DictReader(open(destination, encoding="utf-8")))
    assert rows == []


def test_export_run_notes_csv_rejects_missing_run(tmp_path: Path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    with pytest.raises(KeyError, match="run not found"):
        storage.export_run_notes_csv("missing", str(tmp_path / "x.csv"))


def test_export_run_notes_csv_creates_parent(tmp_path: Path) -> None:
    storage, run = _seed_run(tmp_path, note_body="x")
    destination = tmp_path / "subdir" / "x.csv"
    storage.export_run_notes_csv(run.id, str(destination))
    assert destination.exists()


def test_export_run_notes_csv_skips_non_dict_notes(tmp_path: Path) -> None:
    storage, run = _seed_run(tmp_path, note_body="real")
    raw_run = storage.load_run(run.id)
    raw_run["notes"].append("not a dict")
    storage.save_run(raw_run)
    destination = tmp_path / "out.csv"
    storage.export_run_notes_csv(run.id, str(destination))
    rows = list(csv.DictReader(open(destination, encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["body"] == "real"


def test_api_export_run_notes_returns_csv(tmp_path, api, temp_storage) -> None:
    from src.models import Experiment, Run
    experiment = Experiment(name="api-exp")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    temp_storage.save_run(run.to_dict())
    temp_storage.add_note(run.id, "first note")
    response = api.get(f"/runs/{run.id}/notes.csv")
    assert response.status_code == 200
    text = response.text
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["body"] == "first note"
    assert rows[0]["author"] == ""
    assert rows[0]["run_id"] == run.id


def test_api_export_run_notes_404_for_missing_run(api) -> None:
    response = api.get("/runs/missing-run-id/notes.csv")
    assert response.status_code == 404


def test_client_export_run_notes_csv_to_destination(live_server, temp_storage, tmp_path: Path) -> None:
    from src.client import ExperimentTrackerClient
    from src.models import Experiment, Run

    experiment = Experiment(name="cli-exp")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    temp_storage.save_run(run.to_dict())
    temp_storage.add_note(run.id, "client note")

    client = ExperimentTrackerClient(base_url=live_server)
    destination = tmp_path / "client.csv"
    client.export_run_notes_csv(run.id, destination=str(destination))
    assert destination.exists()
    rows = list(csv.DictReader(open(destination, encoding="utf-8")))
    assert any(row["body"] == "client note" for row in rows)