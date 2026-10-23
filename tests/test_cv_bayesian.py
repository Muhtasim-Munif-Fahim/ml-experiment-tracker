"""Tests for the corrected resampled t-test and Bayesian correlated t-test."""

import pytest

from src.models import (
    Experiment,
    Run,
    RunStatus,
    correlated_t_test,
    cv_bayesian_comparison,
)

DIFFS = [0.012, 0.020, -0.004, 0.015, 0.009, 0.018, 0.003, 0.011, 0.007, 0.016]


def _run(model, fold, value, metric="accuracy", **extra):
    run = Run(experiment_id="exp", name=f"{model}-{fold}", params={"model": model, "fold": fold, **extra})
    run.log_metric(metric, value - 0.3, step=1)
    run.log_metric(metric, value, step=2)  # latest value is used
    return run


def test_matches_scipy_reference():
    # Reference: scipy.stats.t(df=9, loc=mean, scale=sqrt((1/n + rho/(1-rho)) s^2)), rho = 0.1.
    res = correlated_t_test(DIFFS, test_fraction=0.1, rope=0.005)
    assert res["df"] == 9
    assert res["t_statistic"] == pytest.approx(3.1884795336461447, rel=1e-9)
    assert res["p_value"] == pytest.approx(0.011033476798824276, rel=1e-6)
    assert res["naive_p_value"] == pytest.approx(0.001231853631954539, rel=1e-6)
    assert res["p_a_better"] == pytest.approx(0.9381848431909778, rel=1e-6)
    assert res["p_rope"] == pytest.approx(0.06123777051964814, rel=1e-5)
    assert res["p_b_better"] == pytest.approx(0.000577386289374005, rel=1e-5)
    assert res["ci_low"] == pytest.approx(0.0031085817749435633, rel=1e-6)
    assert res["ci_high"] == pytest.approx(0.018291418225056436, rel=1e-6)
    assert res["p_a_better"] + res["p_rope"] + res["p_b_better"] == pytest.approx(1.0)


def test_correction_is_more_conservative_and_rope_zero():
    res = correlated_t_test(DIFFS, test_fraction=0.2)
    assert res["p_value"] > res["naive_p_value"]
    assert res["p_rope"] == pytest.approx(0.0, abs=1e-12)
    # larger test fraction -> more overlap correction -> wider posterior
    assert correlated_t_test(DIFFS, test_fraction=0.5)["scale"] > res["scale"]
    flipped = correlated_t_test([-d for d in DIFFS], test_fraction=0.2)
    assert flipped["p_b_better"] == pytest.approx(res["p_a_better"])


def test_constant_differences_and_validation():
    res = correlated_t_test([0.02, 0.02, 0.02], test_fraction=0.2, rope=0.01)
    assert res["p_a_better"] == 1.0 and res["p_rope"] == 0.0 and res["t_statistic"] is None
    same = correlated_t_test([0.0, 0.0], test_fraction=0.5, rope=0.01)
    assert same["p_rope"] == 1.0 and same["p_value"] == 1.0
    with pytest.raises(ValueError):
        correlated_t_test([0.1], test_fraction=0.2)
    with pytest.raises(ValueError):
        correlated_t_test(DIFFS, test_fraction=1.0)
    with pytest.raises(ValueError):
        correlated_t_test(DIFFS, test_fraction=0.2, rope=-1)
    with pytest.raises(ValueError):
        correlated_t_test([0.1, float("nan")], test_fraction=0.2)


def test_comparison_against_best_config():
    base = [0.80, 0.82, 0.79, 0.81, 0.83]
    runs = []
    for fold, b in enumerate(base):
        runs.append(_run("gbm", fold, b + 0.05 + 0.002 * (fold % 2)))
        runs.append(_run("rf", fold, b + 0.001 * (-1) ** fold))
        runs.append(_run("lr", fold, b))
    runs.append(_run("lr", 99, 0.5))  # fold missing for others -> dropped
    res = cv_bayesian_comparison(runs, "accuracy", rope=0.01)
    assert res["n_blocks"] == 5 and res["dropped_blocks"] == [99]
    assert res["test_fraction"] == pytest.approx(0.2)
    assert res["best"]["params"] == {"model": "gbm"}
    assert len(res["comparisons"]) == 2
    for c in res["comparisons"]:
        assert c["mean_difference"] > 0.04
        assert c["decision"] == "best_better"
    # rf vs lr are practically equivalent
    sub = cv_bayesian_comparison([r for r in runs if r.params["model"] != "gbm"], "accuracy", rope=0.01)
    assert sub["comparisons"][0]["decision"] == "equivalent"
    # minimising flips which configuration is best
    low = cv_bayesian_comparison(runs, "accuracy", maximize=False)
    assert low["best"]["params"] in ({"model": "lr"}, {"model": "rf"})
    assert all(c["mean_difference"] >= 0 for c in low["comparisons"])


def test_not_enough_data_and_validation():
    assert cv_bayesian_comparison([_run("a", 0, 0.9), _run("a", 1, 0.8)], "accuracy") is None
    assert cv_bayesian_comparison([_run("a", 0, 0.9), _run("b", 0, 0.8)], "accuracy") is None
    with pytest.raises(ValueError):
        cv_bayesian_comparison([], "")
    with pytest.raises(ValueError):
        cv_bayesian_comparison([], "accuracy", test_fraction=1.5)
    with pytest.raises(ValueError):
        cv_bayesian_comparison([], "accuracy", decision_threshold=0.4)


def test_storage_api_and_client(api, tracker, temp_storage):
    experiment = Experiment(name="cv-bayes")
    temp_storage.save_experiment(experiment.to_dict())
    table = {"a": [0.91, 0.92, 0.90, 0.93, 0.92], "b": [0.85, 0.86, 0.84, 0.88, 0.85]}
    for model, vals in table.items():
        for fold, v in enumerate(vals):
            run = Run(experiment_id=experiment.id, name=f"{model}-{fold}", params={"model": model, "fold": fold})
            run.log_metric("accuracy", v, step=1)
            run.finish(RunStatus.COMPLETED)
            temp_storage.save_run(run.to_dict())
    stored = temp_storage.experiment_cv_bayesian_comparison(experiment.id, "accuracy")
    diffs = [x - y for x, y in zip(table["a"], table["b"])]
    ref = correlated_t_test(diffs, test_fraction=0.2, rope=0.01)
    assert stored["comparisons"][0]["p_value"] == pytest.approx(ref["p_value"])
    response = api.get(f"/experiments/{experiment.id}/cv-bayesian", params={"metric": "accuracy"})
    assert response.status_code == 200
    assert response.json()["comparisons"][0]["p_best_better"] == pytest.approx(ref["p_a_better"])
    assert api.get("/experiments/nope/cv-bayesian", params={"metric": "accuracy"}).status_code == 404
    bad = api.get(f"/experiments/{experiment.id}/cv-bayesian", params={"metric": "accuracy", "rope": -1})
    assert bad.status_code == 422
    via_client = tracker.cv_bayesian_comparison(experiment.id, "accuracy", test_fraction=0.1, rope=0.0)
    ref2 = correlated_t_test(diffs, test_fraction=0.1)
    assert via_client["comparisons"][0]["p_value"] == pytest.approx(ref2["p_value"])
