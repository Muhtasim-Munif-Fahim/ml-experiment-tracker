"""Tests for ordinary least-squares metric trend regression over steps."""

import pytest

from src.models import Experiment, Run, metric_trend
from src.storage import LocalStorageBackend


def _raw(run):
    return [
        {"name": "loss", "value": metric.value, "step": metric.step}
        for metric in run.metrics
    ]


def build_run(*pairs):
    run = Run(experiment_id="exp", name="r")
    for step, value in pairs:
        run.log_metric("loss", value, step=step)
    return run


def test_metric_trend_detects_a_perfect_linear_increase():
    series = _raw(build_run((0, 0.0), (1, 2.0), (2, 4.0), (3, 6.0), (4, 8.0)))
    result = metric_trend(series)
    assert result is not None
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(0.0)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["std_err"] == 0.0
    assert result["t_statistic"] is None
    assert result["p_value"] == 0.0
    assert result["direction"] == "increasing"
    assert result["n_points"] == 5
    assert result["significant"] is True


def test_metric_trend_perfect_decrease():
    series = _raw(build_run((0, 10.0), (1, 8.0), (2, 6.0), (3, 4.0)))
    result = metric_trend(series)
    assert result["slope"] == pytest.approx(-2.0)
    assert result["direction"] == "decreasing"
    assert result["significant"] is True


def test_metric_trend_flat_series_is_not_a_significant_trend():
    series = _raw(build_run((0, 5.0), (1, 5.0), (2, 5.0), (3, 5.0)))
    result = metric_trend(series)
    assert result["slope"] == pytest.approx(0.0)
    assert result["direction"] == "flat"
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_metric_trend_two_points_yield_a_perfect_line():
    series = _raw(build_run((0, 1.0), (1, 4.0)))
    result = metric_trend(series)
    assert result["slope"] == pytest.approx(3.0)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["t_statistic"] is None
    assert result["n_points"] == 2
    assert result["significant"] is True


def test_metric_trend_returns_none_for_insufficient_points():
    assert metric_trend([]) is None
    assert metric_trend(_raw(build_run((0, 1.0)))) is None


def test_metric_trend_skips_points_without_a_step():
    series = [
        {"name": "loss", "value": 1.0, "step": None},
        {"name": "loss", "value": 2.0, "step": 1},
        {"name": "loss", "value": 4.0, "step": 2},
    ]
    result = metric_trend(series)
    assert result is not None
    assert result["n_points"] == 2
    assert result["slope"] == pytest.approx(2.0)


def test_metric_trend_matches_scipy_for_noisy_data():
    series = _raw(
        build_run(
            (0, 1.0), (1, 3.0), (2, 2.5), (3, 5.0), (4, 6.0),
            (5, 5.5), (6, 8.0), (7, 9.5), (8, 8.5), (9, 11.0),
        )
    )
    result = metric_trend(series)
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        pytest.skip("numpy/scipy not available for cross-check")
    xs = np.array([point["step"] for point in series], dtype=float)
    ys = np.array([point["value"] for point in series], dtype=float)
    slope, intercept, r_value, p_value, _std_err = stats.linregress(xs, ys)
    assert result["slope"] == pytest.approx(slope)
    assert result["intercept"] == pytest.approx(intercept)
    assert result["r_squared"] == pytest.approx(r_value ** 2)
    assert result["p_value"] == pytest.approx(p_value, abs=1e-9)


def test_metric_trend_rejects_out_of_range_alpha():
    series = _raw(build_run((0, 0.0), (1, 1.0)))
    with pytest.raises(ValueError, match="alpha"):
        metric_trend(series, alpha=0.0)


def _seed_trend_run(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    exp = Experiment(name="trend")
    storage.save_experiment(exp.to_dict())
    run = Run(experiment_id=exp.id, name="trainer")
    for step, value in ((0, 0.0), (1, 2.0), (2, 4.0)):
        run.log_metric("loss", value, step=step)
    storage.save_run(run.to_dict())
    return storage, exp, run


def test_run_metric_trend_storage_method(tmp_path):
    storage, _exp, run = _seed_trend_run(tmp_path)

    result = storage.run_metric_trend(run.id, "loss")
    assert result["metric_name"] == "loss"
    assert result["trend"]["slope"] == pytest.approx(2.0)
    assert result["trend"]["significant"] is True

    empty = storage.run_metric_trend(run.id, "missing")
    assert empty["metric_name"] == "missing"
    assert empty["trend"] is None


def test_run_metric_trend_missing_run(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    with pytest.raises(KeyError, match="run not found"):
        storage.run_metric_trend("missing", "loss")


def test_api_metric_trend(api, temp_storage):
    storage, _exp, run = _seed_trend_run_with_storage(temp_storage)

    response = api.get(f"/runs/{run.id}/metrics/loss/trend")
    assert response.status_code == 200
    body = response.json()
    assert body["metric_name"] == "loss"
    assert body["trend"]["slope"] == pytest.approx(2.0)
    assert body["trend"]["direction"] == "increasing"
    assert body["trend"]["significant"] is True

    missing_run = api.get("/runs/missing/metrics/loss/trend")
    assert missing_run.status_code == 404

    absent_metric = api.get(f"/runs/{run.id}/metrics/missing/trend")
    assert absent_metric.status_code == 200
    assert absent_metric.json()["trend"] is None


def _seed_trend_run_with_storage(storage):
    exp = Experiment(name="api-trend")
    storage.save_experiment(exp.to_dict())
    run = Run(experiment_id=exp.id, name="trainer")
    for step, value in ((0, 0.0), (1, 2.0), (2, 4.0), (3, 6.0)):
        run.log_metric("loss", value, step=step)
    storage.save_run(run.to_dict())
    return storage, exp, run


def test_api_metric_trend_alpha_is_rejected_when_out_of_range(api, temp_storage):
    storage, _exp, run = _seed_trend_run_with_storage(temp_storage)

    assert api.get(f"/runs/{run.id}/metrics/loss/trend", params={"alpha": 0.0}).status_code == 422
    assert api.get(f"/runs/{run.id}/metrics/loss/trend", params={"alpha": 1.5}).status_code == 422


def test_client_metric_trend(tracker):
    experiment = tracker.create_experiment("client-trend")
    run = tracker.create_run(experiment["id"], "trainer")
    for step, value in ((0, 0.0), (1, 2.0), (2, 4.0), (3, 6.0)):
        tracker.log_metric(run["id"], "loss", value, step=step)

    result = tracker.metric_trend(run["id"], "loss")
    assert result["metric_name"] == "loss"
    assert result["trend"]["slope"] == pytest.approx(2.0)
    assert result["trend"]["p_value"] == 0.0

    result = tracker.metric_trend(run["id"], "loss", alpha=1.0)
    assert result["trend"]["alpha"] == 1.0
