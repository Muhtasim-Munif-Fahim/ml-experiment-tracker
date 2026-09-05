"""Tests for z-score standardization of run metrics against experiment baselines."""

import pytest

from src.models import Experiment, Run, RunStatus, standardize_series
from src.storage import LocalStorageBackend


def _seed_baseline(storage, exp_name="standardize"):
    experiment = Experiment(name=exp_name)
    storage.save_experiment(experiment.to_dict())
    for value in (4.0, 5.0, 6.0):
        run = Run(experiment_id=experiment.id, name="runner")
        run.log_metric("loss", value, step=1)
        run.finish(RunStatus.COMPLETED)
        storage.save_run(run.to_dict())
    return experiment


def test_standardize_series_returns_zscores():
    assert standardize_series([1.0, 2.0, 3.0], mean=2.0, std=1.0) == [
        pytest.approx(-1.0),
        pytest.approx(0.0),
        pytest.approx(1.0),
    ]


def test_standardize_series_collapses_when_std_is_zero():
    assert standardize_series([5.0, 5.0, 5.0], mean=5.0, std=0.0) == [0.0, 0.0, 0.0]
    assert standardize_series([], mean=1.0, std=0.0) == []


def test_experiment_metric_baseline_reports_descriptive_stats(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed_baseline(storage)
    stats = storage.experiment_metric_baseline(experiment.id, "loss")
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["std"] == pytest.approx(1.0)
    assert stats["min"] == 4.0
    assert stats["max"] == 6.0
    assert stats["median"] == 5.0


def test_experiment_metric_baseline_for_unknown_experiment(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    with pytest.raises(KeyError, match="experiment not found"):
        storage.experiment_metric_baseline("missing", "loss")


def test_experiment_metric_baseline_when_metric_has_no_observations(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed_baseline(storage)
    stats = storage.experiment_metric_baseline(experiment.id, "missing_metric")
    assert stats == {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "median": None,
    }


def test_standardize_run_metric_compute_zscores_and_flags_outliers(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="standardize")
    storage.save_experiment(experiment.to_dict())
    for value in (5.0, 5.0):
        run = Run(experiment_id=experiment.id, name="runner")
        run.log_metric("loss", value, step=1)
        run.finish(RunStatus.COMPLETED)
        storage.save_run(run.to_dict())
    target = Run(experiment_id=experiment.id, name="target")
    target.log_metric("loss", 5.0, step=1)
    target.log_metric("loss", 7.0, step=2)
    storage.save_run(target.to_dict())

    result = storage.standardize_run_metric(experiment.id, target.id, "loss", outlier_threshold=1.0)

    baseline = result["baseline"]
    assert baseline["count"] == 4
    assert baseline["mean"] == pytest.approx(5.5)
    assert baseline["std"] == pytest.approx(1.0)
    points = result["points"]
    assert [point["value"] for point in points] == [5.0, 7.0]
    assert [point["zscore"] for point in points] == pytest.approx([-0.5, 1.5])
    assert [point["is_outlier"] for point in points] == [False, True]


def test_standardize_run_metric_default_threshold_does_not_flag(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="standardize")
    storage.save_experiment(experiment.to_dict())
    for value in (5.0, 5.0):
        run = Run(experiment_id=experiment.id, name="runner")
        run.log_metric("loss", value, step=1)
        run.finish(RunStatus.COMPLETED)
        storage.save_run(run.to_dict())
    target = Run(experiment_id=experiment.id, name="target")
    target.log_metric("loss", 5.0, step=1)
    target.log_metric("loss", 7.0, step=2)
    storage.save_run(target.to_dict())

    result = storage.standardize_run_metric(experiment.id, target.id, "loss")
    assert [point["is_outlier"] for point in result["points"]] == [False, False]


def test_standardize_run_metric_zero_variance_baseline_yields_zero_zscores(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="flat")
    storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="target")
    run.log_metric("loss", 5.0, step=1)
    storage.save_run(run.to_dict())
    second = Run(experiment_id=experiment.id, name="runner")
    second.log_metric("loss", 5.0, step=1)
    storage.save_run(second.to_dict())

    result = storage.standardize_run_metric(experiment.id, run.id, "loss")
    assert result["baseline"]["std"] == 0.0
    assert all(point["zscore"] == 0.0 for point in result["points"])
    assert all(point["is_outlier"] is False for point in result["points"])


def test_standardize_run_metric_missing_run_or_experiment(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="standardize")
    storage.save_experiment(experiment.to_dict())
    with pytest.raises(KeyError, match="experiment not found"):
        storage.standardize_run_metric("missing", "any", "loss")
    with pytest.raises(KeyError, match="run not found"):
        storage.standardize_run_metric(experiment.id, "missing", "loss")


def test_standardize_run_metric_empty_points_when_metric_absent_on_run(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed_baseline(storage)
    target = Run(experiment_id=experiment.id, name="target")
    target.log_metric("accuracy", 0.9, step=1)
    storage.save_run(target.to_dict())

    result = storage.standardize_run_metric(experiment.id, target.id, "loss")
    assert result["points"] == []
    assert result["baseline"]["count"] == 3


def test_api_metric_stats(api, temp_storage) -> None:
    experiment = _seed_baseline(temp_storage)
    response = api.get(f"/experiments/{experiment.id}/metrics/loss/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["std"] == pytest.approx(1.0)


def test_api_metric_stats_404_for_missing_experiment(api) -> None:
    response = api.get("/experiments/missing/metrics/loss/stats")
    assert response.status_code == 404


def test_api_standardized_metric(api, temp_storage) -> None:
    experiment = Experiment(name="standardize")
    temp_storage.save_experiment(experiment.to_dict())
    for value in (5.0, 5.0):
        run = Run(experiment_id=experiment.id, name="runner")
        run.log_metric("loss", value, step=1)
        temp_storage.save_run(run.to_dict())
    target = Run(experiment_id=experiment.id, name="target")
    target.log_metric("loss", 5.0, step=1)
    target.log_metric("loss", 7.0, step=2)
    temp_storage.save_run(target.to_dict())

    response = api.get(
        f"/experiments/{experiment.id}/runs/{target.id}/metrics/loss/standardized",
        params={"outlier_threshold": 1.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert [point["zscore"] for point in body["points"]] == pytest.approx([-0.5, 1.5])
    assert [point["is_outlier"] for point in body["points"]] == [False, True]


def test_api_standardized_metric_404_for_missing_experiment_or_run(api) -> None:
    missing_exp = api.get(
        "/experiments/missing/runs/any/metrics/loss/standardized"
    )
    assert missing_exp.status_code == 404


def test_client_metric_baseline_and_standardize(tracker) -> None:
    experiment = tracker.create_experiment("client-standardize")
    for value in (5.0, 5.0):
        run = tracker.create_run(experiment["id"], "runner")
        tracker.log_metric(run["id"], "loss", value, step=1)
    target = tracker.create_run(experiment["id"], "target")
    tracker.log_metric(target["id"], "loss", 5.0, step=1)
    tracker.log_metric(target["id"], "loss", 7.0, step=2)

    stats = tracker.metric_baseline(experiment["id"], "loss")
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(5.5)
    assert stats["std"] == pytest.approx(1.0)

    result = tracker.standardize_metric(
        experiment["id"], target["id"], "loss", outlier_threshold=1.0
    )
    assert [point["zscore"] for point in result["points"]] == pytest.approx([-0.5, 1.5])
    assert [point["is_outlier"] for point in result["points"]] == [False, True]
