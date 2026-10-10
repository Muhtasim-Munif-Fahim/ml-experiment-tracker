"""Tests for seed-aggregated run groups with confidence intervals and Welch tests."""

import math

import pytest

from src.models import (
    Experiment,
    Run,
    RunStatus,
    group_runs_by_params,
    student_t_critical,
    summarize_values,
    welch_t_test,
)
from src.storage import LocalStorageBackend


def _seed(storage):
    experiment = Experiment(name="seeds")
    storage.save_experiment(experiment.to_dict())
    specs = {
        ("adam", 0.01): [0.90, 0.92, 0.91],
        ("adam", 0.10): [0.80, 0.84, 0.82],
        ("sgd", 0.10): [0.70, 0.95, 0.88],
    }
    for (opt, lr), accs in specs.items():
        for seed, acc in enumerate(accs):
            run = Run(
                experiment_id=experiment.id,
                name=f"{opt}-{lr}-{seed}",
                params={"optimizer": opt, "lr": lr, "seed": seed, "model": {"depth": 4}},
            )
            run.log_metric("accuracy", acc - 0.1, step=1)
            run.log_metric("accuracy", acc, step=2)
            run.finish(RunStatus.COMPLETED)
            storage.save_run(run.to_dict())
    # a run without the metric is ignored
    stray = Run(experiment_id=experiment.id, name="stray", params={"optimizer": "adam"})
    storage.save_run(stray.to_dict())
    return experiment


def test_student_t_critical_known_values():
    assert student_t_critical(0.95, 1) == pytest.approx(12.7062, rel=1e-4)
    assert student_t_critical(0.95, 2) == pytest.approx(4.3027, rel=1e-4)
    assert student_t_critical(0.95, 10) == pytest.approx(2.2281, rel=1e-4)
    assert student_t_critical(0.99, 30) == pytest.approx(2.7500, rel=1e-3)
    assert student_t_critical(0.95, 1e6) == pytest.approx(1.95996, rel=1e-4)
    with pytest.raises(ValueError):
        student_t_critical(1.0, 3)
    with pytest.raises(ValueError):
        student_t_critical(0.95, 0)


def test_summarize_values():
    s = summarize_values([1.0, 2.0, 3.0])
    assert s["mean"] == pytest.approx(2.0)
    assert s["std"] == pytest.approx(1.0)
    assert s["sem"] == pytest.approx(1 / math.sqrt(3))
    half = 4.302652729911275 / math.sqrt(3)
    assert s["ci_low"] == pytest.approx(2.0 - half, rel=1e-5)
    assert s["ci_high"] == pytest.approx(2.0 + half, rel=1e-5)
    single = summarize_values([5.0])
    assert single["n"] == 1 and single["std"] is None and single["ci_low"] is None
    with pytest.raises(ValueError):
        summarize_values([])


def test_welch_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    a = [0.90, 0.92, 0.91, 0.95]
    b = [0.80, 0.84, 0.82]
    res = welch_t_test(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert res["t_statistic"] == pytest.approx(ref.statistic, rel=1e-8)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-6)
    assert res["significant"] is True
    ci = ref.confidence_interval(0.95)
    assert res["ci_low"] == pytest.approx(ci.low, rel=1e-5)
    assert res["ci_high"] == pytest.approx(ci.high, rel=1e-5)


def test_welch_edge_cases():
    assert welch_t_test([1.0], [1.0, 2.0]) is None
    same = welch_t_test([1.0, 1.0], [1.0, 1.0])
    assert same["p_value"] == 1.0 and same["significant"] is False
    apart = welch_t_test([2.0, 2.0], [1.0, 1.0])
    assert apart["p_value"] == 0.0 and apart["significant"] is True
    with pytest.raises(ValueError):
        welch_t_test([1, 2], [3, 4], alpha=0)


def test_group_runs_ignores_seed_and_ranks(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed(storage)
    groups = storage.experiment_run_groups(experiment.id, "accuracy")
    assert len(groups) == 3
    assert [g["rank"] for g in groups] == [1, 2, 3]
    best = groups[0]
    assert best["params"] == {"lr": 0.01, "model.depth": 4, "optimizer": "adam"}
    assert best["n"] == 3 and best["mean"] == pytest.approx(0.91)
    assert best["vs_best"] is None
    means = [g["mean"] for g in groups]
    assert means == sorted(means, reverse=True)
    # the noisy sgd group is not distinguishable from the best; adam-0.1 is
    by_opt = {(g["params"]["optimizer"], g["params"]["lr"]): g for g in groups}
    assert by_opt[("sgd", 0.1)]["vs_best"]["significant"] is False
    assert by_opt[("adam", 0.1)]["vs_best"]["significant"] is True
    assert all(len(g["run_ids"]) == 3 for g in groups)


def test_group_by_and_minimize(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed(storage)
    groups = storage.experiment_run_groups(
        experiment.id, "accuracy", group_by=["optimizer"], maximize=False
    )
    assert [g["params"] for g in groups] == [{"optimizer": "sgd"}, {"optimizer": "adam"}]
    assert groups[1]["n"] == 6
    with pytest.raises(KeyError):
        storage.experiment_run_groups("missing", "accuracy")


def test_group_runs_by_params_on_in_memory_runs():
    runs = []
    for seed, value in enumerate([0.5, 0.6]):
        run = Run(experiment_id="e", name=f"r{seed}", params={"lr": 0.1, "seed": seed})
        run.log_metric("loss", value)
        runs.append(run)
    groups = group_runs_by_params(runs, "loss", maximize=False)
    assert len(groups) == 1 and groups[0]["n"] == 2
    keep_seed = group_runs_by_params(runs, "loss", ignore=())
    assert len(keep_seed) == 2
    with pytest.raises(ValueError):
        group_runs_by_params(runs, "")


def test_api_and_client(api, tracker, temp_storage):
    experiment = _seed(temp_storage)
    response = api.get(f"/experiments/{experiment.id}/run-groups", params={"metric": "accuracy"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3 and body[0]["rank"] == 1
    grouped = api.get(
        f"/experiments/{experiment.id}/run-groups",
        params={"metric": "accuracy", "group_by": "optimizer"},
    ).json()
    assert len(grouped) == 2
    assert api.get("/experiments/nope/run-groups", params={"metric": "accuracy"}).status_code == 404
    bad = api.get(
        f"/experiments/{experiment.id}/run-groups",
        params={"metric": "accuracy", "confidence": 1.5},
    )
    assert bad.status_code == 400
    via_client = tracker.run_groups(experiment.id, "accuracy", group_by=["optimizer"], maximize=False)
    assert [g["params"]["optimizer"] for g in via_client] == ["sgd", "adam"]
