"""Tests for experiment_snapshot."""

from __future__ import annotations

import csv
import io

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def _seed(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="snap-exp")
    storage.save_experiment(experiment.to_dict())
    run_a = Run(experiment_id=experiment.id, name="train-a")
    run_a.log_metric("accuracy", 0.7, step=1)
    run_a.log_metric("accuracy", 0.9, step=2)
    run_a.log_metric("loss", 0.5, step=1)
    run_b = Run(experiment_id=experiment.id, name="train-b")
    run_b.log_metric("accuracy", 0.6, step=1)
    run_b.log_param("lr", 0.01)
    storage.save_run(run_a.to_dict())
    storage.save_run(run_b.to_dict())
    experiment.add_run(run_a)
    experiment.add_run(run_b)
    return storage, experiment


def test_snapshot_returns_one_row_per_run(tmp_path) -> None:
    storage, experiment = _seed(tmp_path)
    rows = storage.experiment_snapshot(experiment.id)
    assert len(rows) == 2
    columns = set(rows[0].keys())
    assert "run_id" in columns
    assert "run_name" in columns
    assert "metric_count" in columns
    assert "artifact_count" in columns


def test_snapshot_includes_latest_metric_values(tmp_path) -> None:
    storage, experiment = _seed(tmp_path)
    rows = storage.experiment_snapshot(experiment.id)
    by_name = {row["run_name"]: row for row in rows}
    assert by_name["train-a"]["accuracy"] == 0.9
    assert by_name["train-b"]["accuracy"] == 0.6
    assert by_name["train-a"]["loss"] == 0.5
    assert "loss" not in by_name["train-b"] or by_name["train-b"]["loss"] == ""


def test_snapshot_restricts_metric_names(tmp_path) -> None:
    storage, experiment = _seed(tmp_path)
    rows = storage.experiment_snapshot(experiment.id, metric_names=["accuracy"])
    assert "accuracy" in rows[0]
    assert "loss" not in rows[0]


def test_snapshot_rejects_unknown_experiment(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    import pytest
    with pytest.raises(KeyError, match="experiment not found"):
        storage.experiment_snapshot("missing")


def test_snapshot_handles_no_runs(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="empty")
    storage.save_experiment(experiment.to_dict())
    rows = storage.experiment_snapshot(experiment.id)
    assert rows == []


def test_snapshot_records_run_counts(tmp_path) -> None:
    storage, experiment = _seed(tmp_path)
    rows = storage.experiment_snapshot(experiment.id)
    by_name = {row["run_name"]: row for row in rows}
    assert by_name["train-a"]["metric_count"] == 3
    assert by_name["train-b"]["metric_count"] == 1
    assert by_name["train-b"]["tag_count"] == 0
    assert by_name["train-b"]["note_count"] == 0


def test_snapshot_csv_endpoint_serialises_rows(tmp_path, api, temp_storage) -> None:
    from src.models import Experiment, Run
    experiment = Experiment(name="csv-exp")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    run.log_metric("accuracy", 0.8)
    temp_storage.save_run(run.to_dict())
    response = api.get(f"/experiments/{experiment.id}/snapshot.csv")
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["run_name"] == "r1"
    assert float(rows[0]["accuracy"]) == 0.8


def test_snapshot_csv_endpoint_filters_metric_names(api, temp_storage) -> None:
    from src.models import Experiment, Run
    experiment = Experiment(name="csv-exp")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    run.log_metric("accuracy", 0.8)
    run.log_metric("loss", 0.4)
    temp_storage.save_run(run.to_dict())
    response = api.get(
        f"/experiments/{experiment.id}/snapshot.csv",
        params={"metric_names": "accuracy"},
    )
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert "accuracy" in rows[0]
    assert "loss" not in rows[0]


def test_snapshot_csv_endpoint_404_for_missing_experiment(api) -> None:
    response = api.get("/experiments/missing/snapshot.csv")
    assert response.status_code == 404