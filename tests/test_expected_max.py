"""Tests for expected max performance vs. tuning budget (Dodge et al. 2019)."""

from __future__ import annotations

import itertools

import pytest

from src.models import Experiment, Run, expected_max_performance


def _runs(spec, exp_id="e"):
    runs = []
    for model, values in spec.items():
        for i, v in enumerate(values):
            run = Run(experiment_id=exp_id, name=f"{model}-{i}", params={"model": {"name": model}})
            run.log_metric("acc", v, step=1)
            runs.append(run)
    return runs


def _brute_force(values, n, maximize=True):
    agg = max if maximize else min
    combos = list(itertools.product(values, repeat=n))
    return sum(agg(c) for c in combos) / len(combos)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("maximize", [True, False])
def test_matches_brute_force_enumeration(n, maximize):
    values = [0.3, 0.9, 0.5, 0.5]
    r = expected_max_performance(_runs({"m": values}), "acc", maximize=maximize, max_budget=3)
    point = r["groups"][0]["curve"][n - 1]
    assert point["expected_best"] == pytest.approx(_brute_force(values, n, maximize))
    assert point["lower"] <= point["expected_best"] <= point["upper"]


def test_curve_is_monotone_and_starts_at_mean():
    values = [0.1, 0.4, 0.2, 0.8, 0.6]
    r = expected_max_performance(_runs({"m": values}), "acc")
    curve = r["groups"][0]["curve"]
    assert curve[0]["expected_best"] == pytest.approx(sum(values) / len(values))
    exp = [p["expected_best"] for p in curve]
    assert all(b >= a - 1e-12 for a, b in zip(exp, exp[1:]))
    assert exp[-1] < max(values)
    assert len(curve) == 5 and not any(p["extrapolated"] for p in curve)


def test_group_crossover():
    # "steady" is reliably decent; "spiky" is usually bad but sometimes great.
    spec = {"steady": [0.70, 0.71, 0.72, 0.70], "spiky": [0.2, 0.3, 0.25, 0.95]}
    r = expected_max_performance(_runs(spec), "acc", group_by="model.name", max_budget=10)
    assert [g["group"] for g in r["groups"]] == ["spiky", "steady"]
    assert r["leader_by_budget"][0]["group"] == "steady"
    assert r["leader_by_budget"][-1]["group"] == "spiky"
    assert len(r["crossovers"]) == 1
    assert r["crossovers"][0]["from"] == "steady"
    assert r["groups"][1]["curve"][-1]["extrapolated"]


def test_empty_and_validation():
    assert expected_max_performance([], "acc")["run_count"] == 0
    with pytest.raises(ValueError, match="metric_name"):
        expected_max_performance([], "")
    with pytest.raises(ValueError, match="max_budget"):
        expected_max_performance([], "acc", max_budget=0)


def _seed(storage):
    exp = Experiment(name="sweep")
    storage.save_experiment(exp.to_dict())
    for run in _runs({"a": [0.5, 0.6], "b": [0.4, 0.9]}, exp_id=exp.id):
        storage.save_run(run.to_dict())
    return exp


def test_storage_and_api(api, temp_storage):
    exp = _seed(temp_storage)
    r = temp_storage.experiment_expected_max(exp.id, "acc", group_by="model.name")
    assert r["run_count"] == 4 and len(r["groups"]) == 2
    with pytest.raises(KeyError):
        temp_storage.experiment_expected_max("missing", "acc")
    resp = api.get(f"/experiments/{exp.id}/expected-max", params={"metric": "acc", "max_budget": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["groups"][0]["curve"][0]["expected_best"] == pytest.approx(0.6)
    assert api.get("/experiments/missing/expected-max", params={"metric": "acc"}).status_code == 404
    assert api.get(f"/experiments/{exp.id}/expected-max", params={"metric": "acc", "max_budget": 0}).status_code == 422
