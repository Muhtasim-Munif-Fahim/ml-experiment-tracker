"""Tests for learning-curve extrapolation and terminate hints."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from src.models import (
    Experiment,
    Run,
    learning_curve_extrapolation,
    learning_curve_forecast,
)
from src.storage import LocalStorageBackend


def _run(name, fn, steps):
    run = Run(experiment_id="e", name=name, params={})
    for t in steps:
        run.log_metric("acc", fn(t), step=t)
    return run


def test_recovers_power_law_and_extrapolates():
    run = _run("r", lambda t: 0.9 - 0.5 * t ** -0.7, range(1, 21))
    out = learning_curve_extrapolation(run, "acc", 100)
    assert out["status"] == "ok"
    assert out["model"] == "pow3"
    assert out["asymptote"] == pytest.approx(0.9, abs=0.01)
    assert out["predicted_value"] == pytest.approx(0.9 - 0.5 * 100 ** -0.7, abs=0.005)
    assert out["predicted_change"] > 0
    assert out["r2"] > 0.999


def test_exponential_curve_prefers_exp3():
    run = _run("r", lambda t: 2.0 * math.exp(-0.3 * t) + 0.25, range(1, 16))
    out = learning_curve_extrapolation(run, "acc", 40)
    assert out["model"] == "exp3"
    assert out["predicted_value"] == pytest.approx(0.25 + 2.0 * math.exp(-12), abs=0.02)


def test_insufficient_data_and_validation():
    run = _run("r", lambda t: t, [1, 2])
    assert learning_curve_extrapolation(run, "acc", 10)["status"] == "insufficient_data"
    with pytest.raises(ValueError):
        learning_curve_extrapolation(run, "", 10)
    with pytest.raises(ValueError):
        learning_curve_extrapolation(run, "acc", 0)
    with pytest.raises(ValueError):
        learning_curve_extrapolation(run, "acc", 10, model="spline")
    with pytest.raises(ValueError):
        learning_curve_extrapolation(run, "acc", 10, min_points=2)


def test_steps_default_to_index_and_duplicates_keep_last():
    run = Run(experiment_id="e", name="r", params={})
    for v in [0.1, 0.3, 0.4, 0.45, 0.47]:
        run.log_metric("acc", v)
    run.log_metric("acc", 0.2, step=1)  # overrides step 1
    out = learning_curve_extrapolation(run, "acc", 10)
    assert out["n_points"] == 5
    assert out["last_step"] == 5.0


def test_forecast_flags_runs_unlikely_to_win():
    good = _run("good", lambda t: 0.92 - 0.6 * t ** -0.5, range(1, 11))
    slow_but_better = _run("late", lambda t: 1.05 - 0.9 * t ** -0.4, range(1, 11))
    bad = _run("bad", lambda t: 0.70 - 0.3 * t ** -0.5, range(1, 11))
    short = _run("short", lambda t: 0.5, [1, 2])
    # "late" trails "good" at step 10 but overtakes it well before step 200
    assert slow_but_better.metrics[-1].value < good.metrics[-1].value
    out = learning_curve_forecast([good, slow_but_better, bad, short], "acc", 200, margin=0.01)
    assert out["incumbent_run_id"] == slow_but_better.id
    hints = {r["name"]: r["terminate_hint"] for r in out["runs"]}
    assert hints == {"late": False, "good": True, "bad": True, "short": False}
    assert out["fitted_count"] == 3 and out["terminate_count"] == 2
    assert out["runs"][0]["name"] == "late"
    assert out["runs"][-1]["status"] == "insufficient_data"


def test_forecast_minimize_mode():
    a = _run("a", lambda t: 0.2 + 1.0 * t ** -0.8, range(1, 12))
    b = _run("b", lambda t: 0.5 + 1.0 * t ** -0.8, range(1, 12))
    out = learning_curve_forecast([a, b], "acc", 100, maximize=False)
    assert out["incumbent_run_id"] == a.id
    assert {r["name"]: r["terminate_hint"] for r in out["runs"]} == {"a": False, "b": True}
    with pytest.raises(ValueError):
        learning_curve_forecast([a], "acc", 100, margin=-1)


def test_storage_and_api(tmp_path, monkeypatch):
    import src.api as api

    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="sweep")
    storage.save_experiment(experiment.to_dict())
    for name, ceiling in (("x", 0.9), ("y", 0.7)):
        run = Run(experiment_id=experiment.id, name=name, params={})
        for t in range(1, 9):
            run.log_metric("acc", ceiling - 0.4 * t ** -0.6, step=t)
        storage.save_run(run.to_dict())
    out = storage.experiment_learning_curve_forecast(experiment.id, "acc", 50)
    assert out["run_count"] == 2 and out["terminate_count"] == 1
    with pytest.raises(KeyError):
        storage.experiment_learning_curve_forecast("missing", "acc", 50)

    monkeypatch.setattr(api, "storage", storage)
    client = TestClient(api.app)
    resp = client.get(
        f"/experiments/{experiment.id}/learning-curve-forecast",
        params={"metric": "acc", "target_step": 50},
    )
    assert resp.status_code == 200
    assert resp.json()["terminate_count"] == 1
    assert client.get(
        "/experiments/missing/learning-curve-forecast",
        params={"metric": "acc", "target_step": 50},
    ).status_code == 404
