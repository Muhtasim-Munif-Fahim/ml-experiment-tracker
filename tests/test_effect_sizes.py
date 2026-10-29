"""Tests for Vargha-Delaney A12 / Cliff's delta / Mann-Whitney U effect-size comparison."""

import pytest

from src.models import (
    Experiment,
    Run,
    RunStatus,
    effect_size_comparison,
    mann_whitney_effect_size,
)

A = [0.91, 0.93, 0.90, 0.94, 0.92, 0.95]
B = [0.88, 0.899, 0.89, 0.87, 0.909, 0.86, 0.885]


def _run(model, seed, value, metric="accuracy", **extra):
    run = Run(experiment_id="exp", name=f"{model}-{seed}", params={"model": model, "seed": seed, **extra})
    run.log_metric(metric, value - 0.4, step=1)
    run.log_metric(metric, value, step=2)  # latest value is used
    return run


def test_exact_and_asymptotic_match_scipy_reference():
    # scipy.stats.mannwhitneyu(A, B, method="exact" / "asymptotic")
    res = mann_whitney_effect_size(A, B, n_bootstrap=0)
    assert res["method"] == "exact" and not res["ties"]
    assert res["u_statistic"] == 41.0
    assert res["p_value"] == pytest.approx(0.002331002331002331, rel=1e-12)
    assert res["a12"] == pytest.approx(41 / 42)
    assert res["cliffs_delta"] == pytest.approx(2 * 41 / 42 - 1)
    assert res["magnitude"] == "large"
    asym = mann_whitney_effect_size(A, B, n_bootstrap=0, method="asymptotic")
    assert asym["p_value"] == pytest.approx(0.005340992525067206, rel=1e-9)
    assert asym["z"] > 0


def test_ties_use_tie_corrected_normal_approximation():
    a = [1, 2, 2, 3, 4, 4, 4, 5]
    b = [2, 3, 3, 3, 5, 6, 6]
    res = mann_whitney_effect_size(a, b, n_bootstrap=0)
    assert res["method"] == "asymptotic" and res["ties"]
    assert res["u_statistic"] == 20.0
    assert res["p_value"] == pytest.approx(0.376759117811582, rel=1e-9)  # scipy reference
    # A12 by direct counting: wins + half ties over all pairs
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    assert res["a12"] == pytest.approx(wins / (len(a) * len(b)))
    assert res["hodges_lehmann"] == -1.0
    with pytest.raises(ValueError):
        mann_whitney_effect_size(a, b, method="exact")


def test_symmetry_magnitudes_and_degenerate_samples():
    fwd = mann_whitney_effect_size(A, B, n_bootstrap=0)
    rev = mann_whitney_effect_size(B, A, n_bootstrap=0)
    assert rev["a12"] == pytest.approx(1 - fwd["a12"])
    assert rev["p_value"] == pytest.approx(fwd["p_value"])
    assert rev["hodges_lehmann"] == pytest.approx(-fwd["hodges_lehmann"])
    same = mann_whitney_effect_size([0.5] * 4, [0.5] * 5, n_bootstrap=100)
    assert same["a12"] == 0.5 and same["p_value"] == 1.0 and same["magnitude"] == "negligible"
    assert same["a12_ci_low"] == same["a12_ci_high"] == 0.5
    small = mann_whitney_effect_size([1, 2, 3, 4, 5, 6], [1.5, 2.5, 3.5, 4.5, 5.5, 0.5], n_bootstrap=0)
    assert small["magnitude"] in {"negligible", "small"}


def test_bootstrap_interval_covers_estimate_and_is_reproducible():
    res = mann_whitney_effect_size(A, B, n_bootstrap=1000, seed=7)
    assert res["a12_ci_low"] <= res["a12"] <= res["a12_ci_high"] <= 1.0
    assert res["a12_ci_low"] > 0.5
    again = mann_whitney_effect_size(A, B, n_bootstrap=1000, seed=7)
    assert again["a12_ci_low"] == res["a12_ci_low"]
    wider = mann_whitney_effect_size(A, B, n_bootstrap=1000, seed=7, confidence=0.99)
    assert wider["a12_ci_low"] <= res["a12_ci_low"]


def test_validation():
    for kwargs in ({"n_bootstrap": -1}, {"confidence": 1.0}, {"method": "x"}, {"n_bootstrap": True}):
        with pytest.raises(ValueError):
            mann_whitney_effect_size(A, B, **kwargs)
    with pytest.raises(ValueError):
        mann_whitney_effect_size([], B)
    with pytest.raises(ValueError):
        mann_whitney_effect_size([float("nan")], B)
    with pytest.raises(ValueError):
        effect_size_comparison([], "")
    with pytest.raises(ValueError):
        effect_size_comparison([], "accuracy", alpha=0.0)


def test_comparison_groups_seeds_and_orients_against_best():
    runs = [_run("a", s, v) for s, v in enumerate(A)]
    runs += [_run("b", s, v) for s, v in enumerate(B)]
    runs += [_run("c", s, v) for s, v in enumerate([0.905, 0.925, 0.915, 0.935])]
    runs.append(Run(experiment_id="exp", name="no-metric", params={"model": "d", "seed": 0}))
    res = effect_size_comparison(runs, "accuracy", n_bootstrap=200)
    assert res["best"]["params"] == {"model": "a"} and res["best"]["n"] == 6
    assert res["best"]["median"] == pytest.approx(0.925)
    by_model = {c["params"]["model"]: c for c in res["comparisons"]}
    assert set(by_model) == {"b", "c"}
    assert by_model["b"]["a12"] == pytest.approx(41 / 42)
    assert by_model["b"]["significant"] and by_model["b"]["test_method"] == "exact"
    assert by_model["b"]["hodges_lehmann"] > 0
    assert not by_model["c"]["significant"]
    assert [c["params"]["model"] for c in res["comparisons"]] == ["b", "c"]  # sorted by A12


def test_minimize_flips_orientation_and_group_by():
    runs = [_run("sgd", s, v, metric="loss", lr=0.1) for s, v in enumerate([0.30, 0.32, 0.31, 0.29])]
    runs += [_run("adam", s, v, metric="loss", lr=0.1) for s, v in enumerate([0.20, 0.22, 0.21, 0.19])]
    res = effect_size_comparison(runs, "loss", maximize=False, n_bootstrap=0)
    assert res["best"]["params"] == {"lr": 0.1, "model": "adam"}
    comp = res["comparisons"][0]
    assert comp["a12"] == 1.0 and comp["cliffs_delta"] == 1.0
    assert comp["hodges_lehmann"] == pytest.approx(0.1)
    assert comp["a12_ci_low"] is None
    pooled = effect_size_comparison(runs, "loss", group_by=["lr"], n_bootstrap=0)
    assert pooled is None  # one configuration only
    assert effect_size_comparison(runs[:4], "loss") is None


def test_storage_api_and_client(api, tracker, temp_storage):
    experiment = Experiment(name="effect-sizes")
    temp_storage.save_experiment(experiment.to_dict())
    for model, vals in {"a": A, "b": B}.items():
        for seed, v in enumerate(vals):
            run = Run(experiment_id=experiment.id, name=f"{model}-{seed}", params={"model": model, "seed": seed})
            run.log_metric("accuracy", v, step=1)
            run.finish(RunStatus.COMPLETED)
            temp_storage.save_run(run.to_dict())
    stored = temp_storage.experiment_effect_size_comparison(experiment.id, "accuracy", n_bootstrap=100)
    assert stored["comparisons"][0]["p_value"] == pytest.approx(0.002331002331002331)
    response = api.get(
        f"/experiments/{experiment.id}/effect-sizes", params={"metric": "accuracy", "n_bootstrap": 100}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["comparisons"][0]["a12"] == pytest.approx(41 / 42)
    assert body["comparisons"][0]["a12_ci_low"] == pytest.approx(stored["comparisons"][0]["a12_ci_low"])
    assert api.get("/experiments/nope/effect-sizes", params={"metric": "accuracy"}).status_code == 404
    bad = api.get(f"/experiments/{experiment.id}/effect-sizes", params={"metric": "accuracy", "alpha": 2})
    assert bad.status_code == 422
    via_client = tracker.effect_size_comparison(experiment.id, "accuracy", n_bootstrap=0, alpha=0.001)
    assert via_client["comparisons"][0]["significant"] is False
    with pytest.raises(KeyError):
        temp_storage.experiment_effect_size_comparison("nope", "accuracy")
