"""Tests for linear interpolation of missing metric steps."""

import pytest

from src.models import interpolate_metric_series


def build_series(values, step_shift=0):
    return [
        {
            "name": "loss",
            "value": value,
            "step": index + step_shift,
            "timestamp": f"2026-09-06T00:00:{index:02d}",
        }
        for index, value in enumerate(values)
    ]


def test_interpolate_fills_a_single_missing_step():
    series = [
        {"name": "loss", "value": 0.0, "step": 0, "timestamp": "t0"},
        {"name": "loss", "value": 10.0, "step": 2, "timestamp": "t2"},
    ]
    result = interpolate_metric_series(series)
    assert [point["step"] for point in result] == [0, 1, 2]
    assert pytest.approx(result[1]["value"]) == 5.0


def test_interpolate_preserves_known_values_and_order():
    series = build_series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = interpolate_metric_series(series)
    assert len(result) == 5
    assert [point["value"] for point in result] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [point["step"] for point in result] == [0, 1, 2, 3, 4]


def test_interpolate_fills_multiple_gaps_and_copies_inputs():
    series = [
        {"name": "acc", "value": 0.0, "step": 0, "timestamp": "t0"},
        {"name": "acc", "value": 0.4, "step": 2, "timestamp": "t2"},
        {"name": "acc", "value": 0.8, "step": 5, "timestamp": "t5"},
    ]
    result = interpolate_metric_series(series)
    assert [point["step"] for point in result] == [0, 1, 2, 3, 4, 5]
    assert [point["value"] for point in result] == [
        pytest.approx(0.0),
        pytest.approx(0.2),
        pytest.approx(0.4),
        pytest.approx(0.4 + 0.4 * (1 / 3)),
        pytest.approx(0.4 + 0.4 * (2 / 3)),
        pytest.approx(0.8),
    ]
    assert all(point is not original for point, original in zip(result, series))
    assert series[0]["value"] == 0.0
    assert series[1]["value"] == 0.4


def test_interpolate_inherits_name_and_timestamp_from_predecessor():
    series = [
        {"name": "loss", "value": 0.0, "step": 0, "timestamp": "pred-t0"},
        {"name": "loss", "value": 10.0, "step": 2, "timestamp": "pred-t2"},
    ]
    result = interpolate_metric_series(series)
    inserted = [point for point in result if point["step"] == 1][0]
    assert inserted["name"] == "loss"
    assert inserted["timestamp"] == "pred-t0"


def test_interpolate_sorts_unordered_steps():
    series = [
        {"name": "acc", "value": 10.0, "step": 2, "timestamp": "t2"},
        {"name": "acc", "value": 0.0, "step": 0, "timestamp": "t0"},
    ]
    result = interpolate_metric_series(series)
    assert [point["step"] for point in result] == [0, 1, 2]
    assert [point["value"] for point in result] == [pytest.approx(0.0), pytest.approx(5.0), pytest.approx(10.0)]


def test_interpolate_respects_max_gap():
    series = [
        {"name": "acc", "value": 0.0, "step": 0, "timestamp": "t0"},
        {"name": "acc", "value": 10.0, "step": 5, "timestamp": "t5"},
    ]
    result = interpolate_metric_series(series, max_gap=2)
    assert [point["step"] for point in result] == [0, 5]


def test_interpolate_max_gap_none_fills_every_gap():
    series = [
        {"name": "acc", "value": 0.0, "step": 0, "timestamp": "t0"},
        {"name": "acc", "value": 10.0, "step": 5, "timestamp": "t5"},
    ]
    result = interpolate_metric_series(series, max_gap=None)
    assert [point["step"] for point in result] == [0, 1, 2, 3, 4, 5]


def test_interpolate_max_gap_equal_to_gap_fills():
    series = [
        {"name": "acc", "value": 0.0, "step": 0, "timestamp": "t0"},
        {"name": "acc", "value": 10.0, "step": 3, "timestamp": "t3"},
    ]
    result = interpolate_metric_series(series, max_gap=2)
    assert [point["step"] for point in result] == [0, 1, 2, 3]
    assert pytest.approx(result[1]["value"]) == 10.0 / 3


def test_interpolate_handles_empty_and_single_point():
    assert interpolate_metric_series([]) == []
    single = build_series([7.0])
    assert interpolate_metric_series(single) == single


def test_interpolate_rejects_non_integer_max_gap():
    series = build_series([0.0, 1.0])
    with pytest.raises(ValueError, match="max_gap"):
        interpolate_metric_series(series, max_gap=0)
    with pytest.raises(ValueError, match="max_gap"):
        interpolate_metric_series(series, max_gap=1.5)


def test_interpolate_requires_steps():
    series = [{"name": "acc", "value": 1.0, "step": None, "timestamp": "t0"}]
    with pytest.raises(ValueError, match="step"):
        interpolate_metric_series(series)


def test_api_interpolates_logged_metric(api):
    experiment = api.post("/experiments/", json={"name": "gappy"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()
    for index in range(0, 13, 2):
        api.post(
            f"/runs/{run['id']}/metrics",
            json={"name": "loss", "value": float(index), "step": index},
        )
    response = api.get(f"/runs/{run['id']}/metrics/loss/interpolated")
    assert response.status_code == 200
    result = response.json()
    assert [point["step"] for point in result] == list(range(13))
    assert [point["value"] for point in result] == [float(index) for index in range(13)]


def test_api_interpolate_validates_requests(api):
    experiment = api.post("/experiments/", json={"name": "edge"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()

    missing_run = api.get("/runs/missing/metrics/loss/interpolated")
    assert missing_run.status_code == 404

    bad_gap = api.get(f"/runs/{run['id']}/metrics/loss/interpolated", params={"max_gap": 0})
    assert bad_gap.status_code == 422

    unknown_metric = api.get(f"/runs/{run['id']}/metrics/missing/interpolated")
    assert unknown_metric.status_code == 200
    assert unknown_metric.json() == []


def test_client_fetches_interpolated_metric(tracker):
    experiment = tracker.create_experiment("client-interp")
    run = tracker.create_run(experiment["id"], "trainer")
    for index in range(0, 13, 2):
        tracker.log_metric(run["id"], "loss", float(index), step=index)

    result = tracker.interpolate_metric(run["id"], "loss")
    assert [point["step"] for point in result] == list(range(13))
    assert [point["value"] for point in result] == [float(index) for index in range(13)]

    for index in range(0, 13, 3):
        tracker.log_metric(run["id"], "gap", float(index), step=index)
    sparse = tracker.interpolate_metric(run["id"], "gap")
    assert [point["step"] for point in sparse] == list(range(13))
    limited = tracker.interpolate_metric(run["id"], "gap", max_gap=1)
    assert [point["step"] for point in limited] == [0, 3, 6, 9, 12]
