"""Tests for metric series smoothing (EMA and simple moving average)."""

import pytest

from src.models import smooth_metric_series


def build_series(values, step_shift=0):
    return [
        {
            "name": "loss",
            "value": value,
            "step": index + step_shift,
            "timestamp": f"2026-09-05T00:00:{index:02d}",
        }
        for index, value in enumerate(values)
    ]


def test_ema_reduces_noise_on_a_constant_series():
    series = build_series([1.0] * 10)
    smoothed = smooth_metric_series(series, window=3, method="ema")
    assert len(smoothed) == 10
    assert all(point["value"] == 1.0 for point in smoothed)


def test_ema_first_point_is_unchanged_and_subsequent_are_blended():
    series = build_series([0.0, 10.0, 10.0])
    smoothed = smooth_metric_series(series, window=3, method="ema")
    # alpha = 2 / (3 + 1) = 0.5
    assert smoothed[0]["value"] == pytest.approx(0.0)
    assert smoothed[1]["value"] == pytest.approx(5.0)
    assert smoothed[2]["value"] == pytest.approx(7.5)


def test_sma_averages_the_trailing_window_and_expands_at_the_start():
    series = build_series([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = smooth_metric_series(series, window=2, method="sma")
    assert [point["value"] for point in smoothed] == [1.0, 1.5, 2.5, 3.5, 4.5]


def test_smooth_keeps_step_timestamp_and_name_and_returns_copies():
    series = build_series([2.0, 4.0])
    smoothed = smooth_metric_series(series, window=2, method="sma")
    assert smoothed[0]["name"] == "loss"
    assert smoothed[0]["step"] == 0
    assert smoothed[1]["step"] == 1
    assert smoothed[0] is not series[0]
    assert series[0]["value"] == 2.0
    assert series[1]["value"] == 4.0


def test_smooth_handles_single_point_and_empty_series():
    single = build_series([7.0])
    assert smooth_metric_series(single, window=3) == [
        {"name": "loss", "value": 7.0, "step": 0, "timestamp": "2026-09-05T00:00:00"}
    ]
    assert smooth_metric_series([], window=3) == []


def test_smooth_rejects_invalid_window_and_method():
    series = build_series([1.0, 2.0])
    with pytest.raises(ValueError, match="positive integer"):
        smooth_metric_series(series, window=0)
    with pytest.raises(ValueError, match="positive integer"):
        smooth_metric_series(series, window=1.5)
    with pytest.raises(ValueError, match="method"):
        smooth_metric_series(series, window=2, method="bogus")


def test_api_smooths_logged_metric(api):
    experiment = api.post("/experiments/", json={"name": "noisy"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()
    for index in range(10):
        api.post(
            f"/runs/{run['id']}/metrics",
            json={"name": "loss", "value": float(index), "step": index},
        )

    response = api.get(f"/runs/{run['id']}/metrics/loss/smooth", params={"window": 3, "method": "sma"})
    assert response.status_code == 200
    smoothed = response.json()
    assert len(smoothed) == 10
    assert smoothed[0]["name"] == "loss"
    expected = [0.0, 0.5, 1.0, 2.0, 3.0]
    assert [point["value"] for point in smoothed[:5]] == pytest.approx(expected)


def test_api_smooth_rejects_invalid_window_and_method(api):
    experiment = api.post("/experiments/", json={"name": "edge"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()

    missing_run = api.get("/runs/missing/metrics/loss/smooth")
    assert missing_run.status_code == 404

    bad_method = api.get(f"/runs/{run['id']}/metrics/loss/smooth", params={"method": "rms"})
    assert bad_method.status_code == 422

    bad_window = api.get(f"/runs/{run['id']}/metrics/loss/smooth", params={"window": 0})
    assert bad_window.status_code == 422

    unknown_metric = api.get(f"/runs/{run['id']}/metrics/missing/smooth", params={"window": 2})
    assert unknown_metric.status_code == 200
    assert unknown_metric.json() == []


def test_client_fetches_smoothed_metric(tracker):
    experiment = tracker.create_experiment("client-smooth")
    run = tracker.create_run(experiment["id"], "trainer")
    for index in range(8):
        tracker.log_metric(run["id"], "loss", float(index), step=index)

    smoothed = tracker.smooth_metric(run["id"], "loss", window=3, method="ema")
    assert len(smoothed) == 8
    # alpha = 2 / (3 + 1) = 0.5
    assert smoothed[0]["value"] == pytest.approx(0.0)
    assert smoothed[1]["value"] == pytest.approx(0.5)
    assert smoothed[2]["value"] == pytest.approx(1.25)
