"""Tests for the Friedman test with Nemenyi post-hoc comparison."""

import math

import pytest

from src.models import (
    Experiment,
    Run,
    RunStatus,
    chi_square_survival,
    friedman_comparison,
    nemenyi_critical_value,
)


def _run(opt, seed, value, metric="accuracy", **extra):
    run = Run(experiment_id="exp", name=f"{opt}-{seed}", params={"optimizer": opt, "seed": seed, **extra})
    run.log_metric(metric, value - 0.5, step=1)
    run.log_metric(metric, value, step=2)  # latest value is used
    return run


def test_nemenyi_matches_demsar_table():
    table_05 = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
    table_10 = [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920]
    for k, (q05, q10) in enumerate(zip(table_05, table_10), start=2):
        assert nemenyi_critical_value(k, 0.05) == pytest.approx(q05, abs=2e-3)
        assert nemenyi_critical_value(k, 0.10) == pytest.approx(q10, abs=2e-3)
    assert nemenyi_critical_value(15) > nemenyi_critical_value(10)
    with pytest.raises(ValueError):
        nemenyi_critical_value(1)
    with pytest.raises(ValueError):
        nemenyi_critical_value(3, alpha=1.5)


def test_chi_square_survival_known_values():
    assert chi_square_survival(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-9)
    assert chi_square_survival(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-9)
    assert chi_square_survival(2.0, 2) == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert chi_square_survival(0.0, 3) == 1.0
    with pytest.raises(ValueError):
        chi_square_survival(1.0, 0)


def test_textbook_example_matches_scipy_reference():
    # 3 configurations x 6 seeds; reference statistic from
    # scipy.stats.friedmanchisquare (tie-corrected) on the same table.
    table = {
        "a": [0.90, 0.88, 0.91, 0.89, 0.92, 0.90],
        "b": [0.85, 0.88, 0.86, 0.84, 0.87, 0.86],
        "c": [0.80, 0.82, 0.86, 0.81, 0.79, 0.83],
    }
    runs = [_run(opt, seed, v) for opt, vals in table.items() for seed, v in enumerate(vals)]
    res = friedman_comparison(runs, "accuracy")
    # ranks per seed (1 best): a=[1,1.5,1,1,1,1] b=[2,1.5,2.5,2,2,2] c=[3,3,2.5,3,3,3]
    avg = {c["params"]["optimizer"]: c["average_rank"] for c in res["configurations"]}
    assert avg["a"] == pytest.approx(6.5 / 6)
    assert avg["b"] == pytest.approx(12.0 / 6)
    assert avg["c"] == pytest.approx(17.5 / 6)
    assert res["k"] == 3 and res["n_blocks"] == 6 and res["df"] == 2
    assert res["chi_square"] == pytest.approx(11.0, rel=1e-9)
    assert res["p_value"] == pytest.approx(0.004086771438464077, rel=1e-6)
    assert res["iman_davenport_df"] == [2, 10]
    f = 5 * res["chi_square"] / (6 * 2 - res["chi_square"])
    assert res["iman_davenport_f"] == pytest.approx(f)
    cd = nemenyi_critical_value(3) * math.sqrt(3 * 4 / 36.0)
    assert res["critical_difference"] == pytest.approx(cd)
    assert [c["params"]["optimizer"] for c in res["configurations"]] == ["a", "b", "c"]
    flags = {c["params"]["optimizer"]: c["worse_than_best"] for c in res["configurations"]}
    assert flags == {"a": False, "b": False, "c": True}  # gap 1.83 > CD 1.35; b's 0.92 is not
    sig = {(p["a"], p["b"]): p["significant"] for p in res["pairwise"]}
    assert sig == {(0, 1): False, (0, 2): True, (1, 2): False}


def test_minimize_reverses_ranks_and_incomplete_blocks_are_dropped():
    runs = [
        _run("sgd", 0, 0.5, metric="loss"),
        _run("sgd", 1, 0.6, metric="loss"),
        _run("sgd", 2, 0.4, metric="loss"),
        _run("adam", 0, 0.3, metric="loss"),
        _run("adam", 1, 0.2, metric="loss"),
        # adam never ran seed 2 -> seed 2 dropped
    ]
    res = friedman_comparison(runs, "loss", maximize=False)
    assert res["blocks"] == [0, 1]
    assert res["dropped_blocks"] == [2]
    assert res["configurations"][0]["params"] == {"optimizer": "adam"}
    assert res["configurations"][0]["average_rank"] == 1.0
    assert res["configurations"][0]["mean"] == pytest.approx(0.25)


def test_duplicates_averaged_ties_and_group_by():
    runs = [
        _run("x", 0, 0.7, width=1),
        _run("x", 0, 0.9, width=2),  # same optimizer & seed: averaged under group_by
        _run("y", 0, 0.8, width=1),
        _run("x", 1, 0.5, width=1),
        _run("y", 1, 0.5, width=1),
    ]
    res = friedman_comparison(runs, "accuracy", group_by=["optimizer"])
    assert res["k"] == 2
    ranks = {c["params"]["optimizer"]: c["average_rank"] for c in res["configurations"]}
    # seed 0: x=0.8 vs y=0.8 tie; seed 1: tie -> everything tied
    assert ranks == {"x": 1.5, "y": 1.5}
    assert res["chi_square"] == 0.0 and res["p_value"] == 1.0


def test_not_enough_data_and_validation():
    assert friedman_comparison([_run("a", 0, 0.9), _run("a", 1, 0.8)], "accuracy") is None
    assert friedman_comparison([_run("a", 0, 0.9), _run("b", 0, 0.8)], "accuracy") is None
    assert friedman_comparison([], "accuracy") is None
    with pytest.raises(ValueError):
        friedman_comparison([], "")
    with pytest.raises(ValueError):
        friedman_comparison([], "accuracy", alpha=0.0)


def test_storage_api_and_client(api, tracker, temp_storage):
    experiment = Experiment(name="friedman")
    temp_storage.save_experiment(experiment.to_dict())
    table = {"adam": [0.91, 0.92, 0.90, 0.93], "sgd": [0.85, 0.86, 0.84, 0.88], "rms": [0.80, 0.83, 0.81, 0.82]}
    for opt, vals in table.items():
        for ds, v in enumerate(vals):
            run = Run(experiment_id=experiment.id, name=f"{opt}-{ds}",
                      params={"optimizer": opt, "dataset": f"d{ds}", "seed": 0})
            run.log_metric("accuracy", v, step=1)
            run.finish(RunStatus.COMPLETED)
            temp_storage.save_run(run.to_dict())

    stored = temp_storage.experiment_friedman_comparison(
        experiment.id, "accuracy", block_param="dataset", ignore=["seed"]
    )
    assert stored["n_blocks"] == 4
    assert [c["params"]["optimizer"] for c in stored["configurations"]] == ["adam", "sgd", "rms"]
    assert stored["chi_square"] == pytest.approx(8.0)
    assert stored["p_value"] == pytest.approx(0.018315638888734182, rel=1e-6)
    # identical ranking in every block: F is infinite -> None, p = 0
    assert stored["iman_davenport_f"] is None and stored["iman_davenport_p_value"] == 0.0

    response = api.get(
        f"/experiments/{experiment.id}/friedman",
        params={"metric": "accuracy", "block_param": "dataset", "ignore": "seed"},
    )
    assert response.status_code == 200
    assert response.json()["p_value"] == pytest.approx(stored["p_value"])
    assert api.get("/experiments/nope/friedman", params={"metric": "accuracy"}).status_code == 404
    bad = api.get(f"/experiments/{experiment.id}/friedman", params={"metric": "accuracy", "alpha": 2})
    assert bad.status_code == 422
    # no seed variation -> no complete design with >= 2 blocks
    assert api.get(f"/experiments/{experiment.id}/friedman", params={"metric": "accuracy"}).json() is None

    via_client = tracker.friedman_comparison(experiment.id, "accuracy", block_param="dataset")
    assert via_client["critical_difference"] == pytest.approx(stored["critical_difference"])
