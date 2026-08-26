"""Tests for shape-preserving LTTB downsampling of metric series."""

import pytest

from src.models import lttb_downsample


def build_series(values):
    return [
        {"name": "loss", "value": value, "step": index, "timestamp": f"2026-08-26T00:00:{index:02d}"}
        for index, value in enumerate(values)
    ]


def test_lttb_keeps_endpoints_and_requested_count():
    series = build_series([float(index) for index in range(100)])
    reduced = lttb_downsample(series, 10)
    assert len(reduced) == 10
    assert reduced[0] == series[0]
    assert reduced[-1] == series[-1]
    assert all(set(sample) == {"name", "value", "step", "timestamp"} for sample in reduced)


def test_lttb_preserves_spike_shape():
    values = [float(index) for index in range(100)]
    values[42] = 999.0
    reduced = lttb_downsample(build_series(values), 8)
    assert 999.0 in [sample["value"] for sample in reduced]


def test_lttb_returns_copies_when_series_is_short():
    series = build_series([1.0, 2.0, 3.0])
    reduced = lttb_downsample(series, 10)
    assert reduced == series
    assert reduced[0] is not series[0]

    exact = lttb_downsample(series, 3)
    assert exact == series

    empty = lttb_downsample([], 5)
    assert empty == []


def test_lttb_rejects_invalid_point_counts():
    series = build_series([1.0, 2.0])
    with pytest.raises(ValueError, match="points"):
        lttb_downsample(series, 1)
    with pytest.raises(ValueError, match="points"):
        lttb_downsample(series, 0)


def test_api_downsamples_logged_metric(api):
    experiment = api.post("/experiments/", json={"name": "curves"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()
    for index in range(30):
        logged = api.post(
            f"/runs/{run['id']}/metrics",
            json={"name": "loss", "value": float(index), "step": index},
        )
        assert logged.status_code == 200
    api.post(f"/runs/{run['id']}/metrics/batch", json={"metrics": {"loss": 40.0}, "step": 30})

    response = api.get(f"/runs/{run['id']}/metrics/loss/downsample", params={"points": 6})
    assert response.status_code == 200
    reduced = response.json()
    assert len(reduced) == 6
    assert reduced[0]["value"] == 0.0
    assert reduced[-1]["value"] == 40.0
    assert all(sample["name"] == "loss" for sample in reduced)


def test_api_downsample_validates_requests(api):
    experiment = api.post("/experiments/", json={"name": "edge"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()

    missing_run = api.get("/runs/missing/metrics/loss/downsample", params={"points": 4})
    assert missing_run.status_code == 404

    unknown_metric = api.get(
        f"/runs/{run['id']}/metrics/nope/downsample", params={"points": 4}
    )
    assert unknown_metric.status_code == 200
    assert unknown_metric.json() == []

    too_few = api.get(f"/runs/{run['id']}/metrics/loss/downsample", params={"points": 1})
    assert too_few.status_code == 422


def test_client_fetches_downsampled_metric(tracker):
    experiment = tracker.create_experiment("client-curves")
    run = tracker.create_run(experiment["id"], "trainer")
    for index in range(24):
        tracker.log_metric(run["id"], "loss", float(index), step=index)
    tracker.log_metrics(run["id"], {"loss": 99.0}, step=24)

    reduced = tracker.downsample_metric(run["id"], "loss", points=5)
    assert len(reduced) == 5
    assert reduced[0]["value"] == 0.0
    assert reduced[-1]["value"] == 99.0