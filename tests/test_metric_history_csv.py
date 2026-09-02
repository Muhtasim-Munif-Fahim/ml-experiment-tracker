"""Tests for the experiment metric long-form history CSV export."""

from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime as dt

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def seed_experiment(storage) -> Experiment:
    experiment = Experiment(name="history-exp")
    storage.save_experiment(experiment.to_dict())

    run_a = Run(experiment_id=experiment.id, name="train-a")
    run_a.log_metric("accuracy", 0.7, step=1)
    run_a.log_metric("accuracy", 0.9, step=2)
    run_a.log_metric("loss", 0.5, step=1)
    run_a.log_metric("loss", 0.3, step=2)

    run_b = Run(experiment_id=experiment.id, name="train-b")
    run_b.log_metric("accuracy", 0.6, step=1)
    run_b.log_metric("loss", 0.4, step=1)

    storage.save_run(run_a.to_dict())
    storage.save_run(run_b.to_dict())
    experiment.add_run(run_a)
    experiment.add_run(run_b)
    return experiment


def test_experiment_metric_long_returns_every_observation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment(storage)
        rows = storage.experiment_metric_long(experiment.id)
        # 4 from run_a + 2 from run_b
        assert len(rows) == 6
        # Every row carries the expected keys
        for row in rows:
            assert set(row) == {
                "run_id", "run_name", "metric_name", "step", "value", "timestamp",
            }
            assert row["run_id"] in (experiment.runs[0].id, experiment.runs[1].id)
            assert row["run_name"] in ("train-a", "train-b")
            assert row["metric_name"] in ("accuracy", "loss")


def test_experiment_metric_long_filters_metric_names() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment(storage)
        rows = storage.experiment_metric_long(experiment.id, metric_names=["accuracy"])
        assert all(row["metric_name"] == "accuracy" for row in rows)
        assert len(rows) == 3


def test_experiment_metric_long_filters_step_range() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment(storage)
        rows = storage.experiment_metric_long(experiment.id, start_step=1, end_step=1)
        assert all(row["step"] == 1 for row in rows)


def test_experiment_metric_long_rejects_invalid_step_range() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment(storage)
        with pytest.raises(ValueError, match="start_step must not exceed end_step"):
            storage.experiment_metric_long(experiment.id, start_step=10, end_step=2)


def test_experiment_metric_long_rejects_unknown_experiment() -> None:
    with pytest.raises(KeyError, match="experiment not found"):
        LocalStorageBackend("/tmp").experiment_metric_long("missing")


def test_experiment_metric_long_is_sorted_by_run_then_step() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment(storage)
        rows = storage.experiment_metric_long(experiment.id)
        # Sort key is (run_name, metric_name, step)
        seen = [(row["run_name"], row["metric_name"], row["step"]) for row in rows]
        assert seen == sorted(seen)


def test_api_metrics_history_csv_returns_long_form_rows(api, temp_storage) -> None:
    experiment = seed_experiment(temp_storage)
    response = api.get(f"/experiments/{experiment.id}/metrics.history.csv")
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert len(rows) == 6
    assert set(rows[0]) == {
        "run_id", "run_name", "metric_name", "step", "value", "timestamp",
    }
    assert {row["metric_name"] for row in rows} == {"accuracy", "loss"}


def test_api_metrics_history_csv_accepts_metric_filter(api, temp_storage) -> None:
    experiment = seed_experiment(temp_storage)
    response = api.get(
        f"/experiments/{experiment.id}/metrics.history.csv",
        params={"metric_names": "accuracy"},
    )
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert all(row["metric_name"] == "accuracy" for row in rows)
    assert len(rows) == 3


def test_api_metrics_history_csv_accepts_step_range(api, temp_storage) -> None:
    experiment = seed_experiment(temp_storage)
    response = api.get(
        f"/experiments/{experiment.id}/metrics.history.csv",
        params={"start_step": 2, "end_step": 2},
    )
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert all(int(row["step"]) == 2 for row in rows)


def test_api_metrics_history_csv_404_for_unknown_experiment(api) -> None:
    response = api.get("/experiments/missing/metrics.history.csv")
    assert response.status_code == 404


def test_api_metrics_history_csv_400_for_invalid_range(api, temp_storage) -> None:
    experiment = seed_experiment(temp_storage)
    response = api.get(
        f"/experiments/{experiment.id}/metrics.history.csv",
        params={"start_step": 5, "end_step": 1},
    )
    assert response.status_code == 400


def test_client_can_fetch_metrics_history_csv(live_server, temp_storage) -> None:
    from src.client import ExperimentTrackerClient

    experiment = seed_experiment(temp_storage)
    client = ExperimentTrackerClient(base_url=live_server)
    csv_text = client.experiment_metric_history_csv(experiment.id)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 6


def test_client_can_save_metrics_history_csv_to_destination(
    live_server, temp_storage, tmp_path,
) -> None:
    from src.client import ExperimentTrackerClient

    experiment = seed_experiment(temp_storage)
    client = ExperimentTrackerClient(base_url=live_server)
    destination = tmp_path / "history.csv"
    client.experiment_metric_history_csv(
        experiment.id, destination=str(destination)
    )
    assert destination.exists()
    # Round-trip the on-disk file through the CSV reader; that is the
    # contract callers actually rely on.
    on_disk = destination.read_bytes()
    assert b"run_id,run_name,metric_name,step,value,timestamp" in on_disk
    reader = csv.DictReader(io.StringIO(on_disk.decode("utf-8")))
    rows = list(reader)
    assert len(rows) == 6