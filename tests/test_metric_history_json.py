"""Tests for the experiment per-step metric history JSON export."""

from __future__ import annotations

import csv
import io
import json

from tests.test_metric_history_csv import seed_experiment


HISTORY_COLUMNS = [
    "run_id",
    "run_name",
    "metric_name",
    "step",
    "value",
    "timestamp",
]


def test_api_metrics_history_json_matches_csv_rows(api, temp_storage) -> None:
    experiment = seed_experiment(temp_storage)
    response = api.get(f"/experiments/{experiment.id}/metrics.history.json")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith(
        f'filename="experiment-{experiment.id}-metrics-history.json"'
    )

    payload = response.json()
    assert payload["experiment_id"] == experiment.id
    assert payload["columns"] == HISTORY_COLUMNS
    assert len(payload["rows"]) == 6
    assert {row["metric_name"] for row in payload["rows"]} == {"accuracy", "loss"}
    for row in payload["rows"]:
        assert list(row) == HISTORY_COLUMNS

    csv_response = api.get(f"/experiments/{experiment.id}/metrics.history.csv")
    csv_rows = list(csv.DictReader(io.StringIO(csv_response.text)))
    assert [row["metric_name"] for row in payload["rows"]] == [
        row["metric_name"] for row in csv_rows
    ]
    assert [row["step"] for row in payload["rows"]] == [
        int(row["step"]) for row in csv_rows
    ]
    assert [row["value"] for row in payload["rows"]] == [
        float(row["value"]) for row in csv_rows
    ]


def test_api_metrics_history_json_accepts_metric_and_step_filters(
    api, temp_storage
) -> None:
    experiment = seed_experiment(temp_storage)
    response = api.get(
        f"/experiments/{experiment.id}/metrics.history.json",
        params={"metric_names": "accuracy", "start_step": 2, "end_step": 2},
    )
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["metric_name"] == "accuracy"
    assert rows[0]["step"] == 2
    assert rows[0]["run_name"] == "train-a"
    assert rows[0]["value"] == 0.9


def test_api_metrics_history_json_404_for_unknown_experiment(api) -> None:
    response = api.get("/experiments/missing/metrics.history.json")
    assert response.status_code == 404


def test_api_metrics_history_json_400_for_invalid_range(api, temp_storage) -> None:
    experiment = seed_experiment(temp_storage)
    response = api.get(
        f"/experiments/{experiment.id}/metrics.history.json",
        params={"start_step": 5, "end_step": 1},
    )
    assert response.status_code == 400


def test_client_can_fetch_and_save_metrics_history_json(
    live_server, temp_storage, tmp_path
) -> None:
    from src.client import ExperimentTrackerClient

    experiment = seed_experiment(temp_storage)
    client = ExperimentTrackerClient(base_url=live_server)
    destination = tmp_path / "history.json"
    payload = client.experiment_metric_history_json(
        experiment.id,
        metric_names=["loss"],
        destination=str(destination),
    )
    assert payload["columns"] == HISTORY_COLUMNS
    assert len(payload["rows"]) == 3
    assert all(row["metric_name"] == "loss" for row in payload["rows"])
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk == payload
