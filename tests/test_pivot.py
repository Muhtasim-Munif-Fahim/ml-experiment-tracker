"""Tests for per-experiment metric pivot CSV export."""

import csv
import io
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.models import Experiment, Run, Metric
from src.storage import LocalStorageBackend


def seed_experiment_with_metrics(storage) -> Experiment:
    experiment = Experiment(name="pivot-exp")
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


def test_experiment_metric_pivot_returns_wide_table():
    with pytest.raises(KeyError, match="experiment not found"):
        LocalStorageBackend("/tmp").experiment_metric_pivot("missing")

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment_with_metrics(storage)

        table = storage.experiment_metric_pivot(experiment.id)
        assert table["columns"] == ["run_id", "run_name", "accuracy", "loss"]
        assert len(table["rows"]) == 2

        row_by_id = {row[0]: row for row in table["rows"]}
        a_row = row_by_id[experiment.runs[0].id]
        assert a_row[1] == "train-a"
        assert a_row[2] == 0.9
        assert a_row[3] == 0.3

        b_row = row_by_id[experiment.runs[1].id]
        assert b_row[1] == "train-b"
        assert b_row[2] == 0.6
        assert b_row[3] == 0.4


def test_experiment_metric_pivot_filters_requested_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment_with_metrics(storage)

        table = storage.experiment_metric_pivot(experiment.id, ["accuracy"])
        assert table["columns"] == ["run_id", "run_name", "accuracy"]
        assert len(table["rows"]) == 2
        for row in table["rows"]:
            assert len(row) == 3


def test_experiment_metric_pivot_handles_empty_experiment():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = Experiment(name="empty")
        storage.save_experiment(experiment.to_dict())

        table = storage.experiment_metric_pivot(experiment.id)
        assert table["columns"] == ["run_id", "run_name"]
        assert table["rows"] == []


def test_api_experiment_pivot_returns_csv(api):
    experiment = api.post("/experiments/", json={"name": "pivot-api"}).json()
    run_a = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "train-a"}
    ).json()
    run_b = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "train-b"}
    ).json()

    api.post(
        f"/runs/{run_a['id']}/metrics",
        json={"name": "accuracy", "value": 0.9, "step": 1},
    )
    api.post(
        f"/runs/{run_a['id']}/metrics",
        json={"name": "loss", "value": 0.3, "step": 1},
    )
    api.post(
        f"/runs/{run_b['id']}/metrics",
        json={"name": "accuracy", "value": 0.6, "step": 1},
    )

    response = api.get(f"/experiments/{experiment['id']}/pivot.csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment" in response.headers["content-disposition"]

    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert len(rows) == 2
    assert {row["run_name"] for row in rows} == {"train-a", "train-b"}
    acc_values = {float(row["accuracy"]) for row in rows}
    assert acc_values == {0.9, 0.6}


def test_api_experiment_pivot_filters_metrics(api):
    experiment = api.post("/experiments/", json={"name": "pivot-filter"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()

    api.post(
        f"/runs/{run['id']}/metrics",
        json={"name": "accuracy", "value": 0.8, "step": 1},
    )
    api.post(
        f"/runs/{run['id']}/metrics",
        json={"name": "loss", "value": 0.2, "step": 1},
    )

    response = api.get(
        f"/experiments/{experiment['id']}/pivot.csv", params={"metric_names": "accuracy"}
    )
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert rows[0]["accuracy"] == "0.8"
    assert "loss" not in rows[0]


def test_api_experiment_pivot_404_for_missing(api):
    assert api.get("/experiments/missing/pivot.csv").status_code == 404


def test_client_fetches_experiment_metric_pivot(tracker, tmp_path):
    experiment = tracker.create_experiment("client-pivot")
    run_a = tracker.create_run(experiment["id"], "run-a")
    run_b = tracker.create_run(experiment["id"], "run-b")

    tracker.log_metric(run_a["id"], "accuracy", 0.9, step=1)
    tracker.log_metric(run_a["id"], "loss", 0.3, step=1)
    tracker.log_metric(run_b["id"], "accuracy", 0.6, step=1)

    text = tracker.experiment_metric_pivot_csv(experiment["id"])
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    assert len(rows) == 2
    names = {row["run_name"] for row in rows}
    assert names == {"run-a", "run-b"}

    destination = tmp_path / "pivot.csv"
    tracker.experiment_metric_pivot_csv(
        experiment["id"], destination=str(destination)
    )
    assert destination.exists()
    saved_reader = csv.DictReader(io.StringIO(destination.read_text(encoding="utf-8")))
    saved_rows = list(saved_reader)
    assert saved_rows == rows
