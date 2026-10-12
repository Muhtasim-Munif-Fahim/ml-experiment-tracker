"""Tests for ANOVA-based hyperparameter importance (eta- / omega-squared)."""

import math
import random

import pytest

from src.models import (
    Experiment,
    Run,
    RunStatus,
    one_way_anova,
    parameter_importance,
    spearman_correlation,
)
from src.storage import LocalStorageBackend


def _runs(specs, metric="accuracy"):
    runs = []
    for params, value in specs:
        run = Run(experiment_id="exp", name="r", params=params)
        run.log_metric(metric, value, step=1)
        runs.append(run)
    return runs


def test_one_way_anova_matches_textbook_example():
    # Three groups; hand-computed SSB = 26, SSW = 6, df = (2, 6).
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [3.0, 4.0, 5.0]]
    res = one_way_anova(groups)
    grand = 33.0 / 9
    ssb = sum(3 * (sum(g) / 3 - grand) ** 2 for g in groups)
    ssw = 6.0
    assert res["df_between"] == 2 and res["df_within"] == 6
    assert res["eta_squared"] == pytest.approx(ssb / (ssb + ssw))
    assert res["f_statistic"] == pytest.approx((ssb / 2) / (ssw / 6))
    # F(2, 6) survival: closed form for df1 = 2 is (1 + 2F/6)^(-3).
    assert res["p_value"] == pytest.approx((1 + 2 * res["f_statistic"] / 6) ** -3, rel=1e-9)
    omega = (ssb - 2 * (ssw / 6)) / (ssb + ssw + ssw / 6)
    assert res["omega_squared"] == pytest.approx(omega)


def test_one_way_anova_matches_scipy_when_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = random.Random(0)
    groups = [[rng.gauss(mu, 1.0) for _ in range(n)] for mu, n in ((0, 7), (0.5, 9), (1.2, 5), (0.1, 11))]
    res = one_way_anova(groups)
    ref = scipy_stats.f_oneway(*groups)
    assert res["f_statistic"] == pytest.approx(ref.statistic, rel=1e-10)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-7)


def test_one_way_anova_undefined_cases():
    assert one_way_anova([[1.0, 2.0]]) is None
    assert one_way_anova([[1.0], [2.0]]) is None  # no within df
    assert one_way_anova([[1.0, 1.0], [1.0, 1.0]]) is None  # zero variance
    perfect = one_way_anova([[1.0, 1.0], [2.0, 2.0]])
    assert math.isinf(perfect["f_statistic"]) and perfect["p_value"] == 0.0
    assert perfect["eta_squared"] == pytest.approx(1.0)


def test_spearman_handles_ties_and_monotone_nonlinear():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert spearman_correlation(xs, [x ** 3 for x in xs]) == pytest.approx(1.0)
    scipy_stats = pytest.importorskip("scipy.stats")
    a = [1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0]
    b = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
    assert spearman_correlation(a, b) == pytest.approx(scipy_stats.spearmanr(a, b).statistic)


def test_categorical_param_dominates_and_seed_is_ignored():
    rng = random.Random(1)
    specs = []
    for opt, base in (("adam", 0.9), ("sgd", 0.7), ("rmsprop", 0.8)):
        for seed in range(6):
            for width in (64, 128):
                specs.append(({"optimizer": opt, "seed": seed, "width": width},
                              base + rng.gauss(0, 0.01)))
    result = parameter_importance(_runs(specs), "accuracy")
    names = [r["param_name"] for r in result]
    assert "seed" not in names
    assert names[0] == "optimizer"
    top = result[0]
    assert top["kind"] == "categorical" and top["levels"] == 3 and top["run_count"] == 36
    assert top["eta_squared"] > 0.95 and top["p_value"] < 1e-10
    assert top["spearman"] is None
    means = {row["value"]: row["mean"] for row in top["level_means"]}
    assert means["adam"] == pytest.approx(0.9, abs=0.01)
    width = next(r for r in result if r["param_name"] == "width")
    assert width["omega_squared"] < 0.05


def test_non_monotone_numeric_effect_found_by_anova_not_correlation():
    # Metric peaks at the middle learning rate: Pearson ~ 0 but eta^2 ~ 1.
    specs = []
    for lr, acc in ((0.001, 0.6), (0.01, 0.9), (0.1, 0.6)):
        for jitter in (-0.005, 0.0, 0.005):
            specs.append(({"lr": lr}, acc + jitter))
    row = parameter_importance(_runs(specs), "accuracy")[0]
    assert row["param_name"] == "lr" and row["kind"] == "numeric" and not row["binned"]
    assert row["eta_squared"] > 0.99
    assert abs(row["spearman"]) < 0.2


def test_many_distinct_numeric_values_are_binned():
    specs = [({"dropout": i / 100.0}, i / 100.0) for i in range(40)]
    row = parameter_importance(_runs(specs), "accuracy", max_levels=4)[0]
    assert row["binned"] and row["levels"] == 4
    assert all(level["n"] == 10 for level in row["level_means"])
    assert row["level_means"][0]["value"] == {"min": 0.0, "max": 0.09}
    assert row["spearman"] == pytest.approx(1.0)


def test_nested_params_are_flattened_and_validation():
    specs = [({"opt": {"name": n}}, v) for n, v in (("a", 1.0), ("a", 1.1), ("b", 2.0), ("b", 2.1))]
    result = parameter_importance(_runs(specs), "accuracy")
    assert result[0]["param_name"] == "opt.name"
    assert parameter_importance(_runs(specs), "missing_metric") == []
    with pytest.raises(ValueError, match="max_levels"):
        parameter_importance([], "accuracy", max_levels=1)
    with pytest.raises(ValueError, match="metric_name"):
        parameter_importance([], "")


def test_storage_api_and_client(api, tracker, temp_storage):
    experiment = Experiment(name="importance")
    temp_storage.save_experiment(experiment.to_dict())
    for opt, base in (("adam", 0.9), ("sgd", 0.6)):
        for seed in range(3):
            run = Run(experiment_id=experiment.id, name=f"{opt}-{seed}",
                      params={"optimizer": opt, "seed": seed})
            run.log_metric("accuracy", base + 0.01 * seed, step=1)
            run.finish(RunStatus.COMPLETED)
            temp_storage.save_run(run.to_dict())

    stored = temp_storage.experiment_parameter_importance(experiment.id, "accuracy")
    assert [r["param_name"] for r in stored] == ["optimizer"]

    response = api.get(f"/experiments/{experiment.id}/parameter-importance", params={"metric": "accuracy"})
    assert response.status_code == 200
    assert response.json()[0]["param_name"] == "optimizer"

    with_seed = api.get(
        f"/experiments/{experiment.id}/parameter-importance",
        params={"metric": "accuracy", "ignore": ""},
    ).json()
    assert {r["param_name"] for r in with_seed} == {"optimizer", "seed"}

    assert api.get("/experiments/nope/parameter-importance", params={"metric": "accuracy"}).status_code == 404
    bad = api.get(
        f"/experiments/{experiment.id}/parameter-importance",
        params={"metric": "accuracy", "max_levels": 1},
    )
    assert bad.status_code == 400

    via_client = tracker.parameter_importance(experiment.id, "accuracy")
    assert via_client[0]["eta_squared"] == pytest.approx(stored[0]["eta_squared"])
