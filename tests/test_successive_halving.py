"""Tests for the successive-halving replay of logged learning curves."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run, RunStatus, successive_halving_replay


def _run(name: str, curve: dict, metric: str = "acc") -> Run:
    run = Run(experiment_id="e", name=name)
    run.id = name
    for step, value in curve.items():
        run.log_metric(metric, value, step=step)
    return run


def _bracket() -> list:
    # Nine runs logged at steps 1..9. r0..r7 grow linearly from different
    # starts; "late" starts worst but ends best.
    runs = []
    for i in range(8):
        runs.append(_run(f"r{i}", {s: 0.1 * i + 0.01 * s for s in range(1, 10)}))
    runs.append(_run("late", {s: 0.0 if s <= 3 else 1.5 for s in range(1, 10)}))
    return runs


def test_default_rungs_promotion_and_budget():
    result = successive_halving_replay(_bracket(), "acc", eta=3)
    assert result["run_count"] == 9
    assert [r["step"] for r in result["rungs"]] == [1, 3]
    first, second = result["rungs"]
    assert first["candidates"] == 9
    assert first["promoted"] == ["r7", "r6", "r5"]
    assert "late" in first["stopped"]
    assert first["cutoff"] == pytest.approx(0.51)
    assert second["candidates"] == 3
    assert second["promoted"] == ["r7"]
    assert second["stopped"] == ["r6", "r5"]
    assert result["survivors"] == ["r7"]
    assert result["budget_full"] == 81
    assert result["budget_used"] == 6 * 1 + 2 * 3 + 9
    assert result["budget_saved_fraction"] == pytest.approx(1 - 21 / 81, abs=1e-6)


def test_late_bloomer_is_killed_and_regret_reported():
    result = successive_halving_replay(_bracket(), "acc", eta=3)
    assert result["best_run_id"] == "late"
    assert result["best_final_value"] == pytest.approx(1.5)
    assert result["best_run_survived"] is False
    assert result["best_surviving_run_id"] == "r7"
    assert result["regret"] == pytest.approx(1.5 - 0.79)
    late = next(r for r in result["runs"] if r["run_id"] == "late")
    assert late["stopped_at"] == 1 and late["steps_used"] == 1
    # Rung 1 is a weak proxy because of the late bloomer; it is still positive.
    assert result["rungs"][0]["spearman_with_final"] < 1.0


def test_minimize_and_explicit_rungs():
    runs = [_run(f"r{i}", {2: 1.0 - 0.1 * i, 4: 0.9 - 0.1 * i, 8: 0.8 - 0.1 * i}, "loss") for i in range(4)]
    result = successive_halving_replay(runs, "loss", maximize=False, eta=2, rungs=[2, 4])
    assert [r["step"] for r in result["rungs"]] == [2, 4]
    assert result["rungs"][0]["promoted"] == ["r3", "r2"]
    assert result["rungs"][1]["promoted"] == ["r3"]
    assert result["best_run_survived"] is True
    assert result["regret"] == 0.0
    assert result["rungs"][0]["spearman_with_final"] == pytest.approx(1.0)


def test_uses_latest_value_before_rung_and_short_runs_finish():
    runs = [
        _run("sparse", {1: 0.9, 10: 0.95}),  # value at rung 3 is the step-1 value
        _run("dense", {1: 0.1, 2: 0.2, 3: 0.3, 10: 0.4}),
        _run("short", {1: 0.5, 2: 0.6}),  # ends before rung 3
    ]
    result = successive_halving_replay(runs, "acc", eta=2, rungs=[1, 3])
    r1, r3 = result["rungs"]
    assert r1["promoted"] == ["sparse"]
    assert r3["candidates"] == 1 and r3["promoted"] == ["sparse"]
    assert r3["cutoff"] == pytest.approx(0.9)  # latest value at or before step 3
    rows = {r["run_id"]: r for r in result["runs"]}
    assert rows["sparse"]["stopped_at"] is None
    assert rows["short"]["stopped_at"] == 1 and rows["short"]["steps_used"] == 1


def test_short_run_that_survives_finishes_at_its_end():
    runs = [
        _run("a", {1: 0.9, 2: 0.95}),
        _run("b", {1: 0.1, 2: 0.2, 3: 0.3, 9: 0.4}),
    ]
    result = successive_halving_replay(runs, "acc", eta=2, rungs=[1, 3])
    assert [r["step"] for r in result["rungs"]] == [1]  # nobody alive reaches 3
    rows = {r["run_id"]: r for r in result["runs"]}
    assert rows["a"]["stopped_at"] is None and rows["a"]["steps_used"] == 2
    assert rows["b"]["stopped_at"] == 1


def test_duplicate_steps_last_wins_and_unstepped_ignored():
    run = Run(experiment_id="e", name="x")
    run.id = "x"
    run.log_metric("acc", 0.1, step=1)
    run.log_metric("acc", 0.7, step=1)
    run.log_metric("acc", 0.99)  # no step
    other = _run("y", {1: 0.5})
    result = successive_halving_replay([run, other], "acc", eta=2, rungs=[1])
    assert result["rungs"][0]["promoted"] == ["x"]


def test_none_and_validation():
    assert successive_halving_replay([_run("a", {1: 0.5})], "acc") is None
    assert successive_halving_replay([], "acc") is None
    with pytest.raises(ValueError, match="metric_name"):
        successive_halving_replay([], "")
    with pytest.raises(ValueError, match="eta"):
        successive_halving_replay(_bracket(), "acc", eta=1.0)
    with pytest.raises(ValueError, match="min_step"):
        successive_halving_replay(_bracket(), "acc", min_step=0)
    with pytest.raises(ValueError, match="rungs"):
        successive_halving_replay(_bracket(), "acc", rungs=[0, 3])


def test_min_step_controls_first_rung():
    result = successive_halving_replay(_bracket(), "acc", eta=2, min_step=2)
    assert [r["step"] for r in result["rungs"]] == [2, 4, 8]


def _store(storage):
    experiment = Experiment(name="sweep")
    storage.save_experiment(experiment.to_dict())
    for run in _bracket():
        run.experiment_id = experiment.id
        run.finish(RunStatus.COMPLETED)
        storage.save_run(run.to_dict())
    return experiment.id


def test_storage_api_and_client(api, temp_storage):
    exp_id = _store(temp_storage)
    direct = temp_storage.experiment_successive_halving(exp_id, "acc")
    assert direct["best_run_survived"] is False

    response = api.get(f"/experiments/{exp_id}/successive-halving", params={"metric": "acc", "eta": 3})
    assert response.status_code == 200
    assert response.json()["survivors"] == ["r7"]

    response = api.get(
        f"/experiments/{exp_id}/successive-halving", params={"metric": "acc", "rungs": "2,5"}
    )
    assert [r["step"] for r in response.json()["rungs"]] == [2, 5]

    assert api.get(f"/experiments/{exp_id}/successive-halving", params={"metric": "nope"}).json() is None
    assert api.get(f"/experiments/{exp_id}/successive-halving", params={"metric": "acc", "rungs": "a"}).status_code == 400
    assert api.get(f"/experiments/{exp_id}/successive-halving", params={"metric": "acc", "eta": 1}).status_code == 422
    assert api.get("/experiments/missing/successive-halving", params={"metric": "acc"}).status_code == 404


def test_http_client(tracker, temp_storage):
    exp_id = _store(temp_storage)
    result = tracker.successive_halving(exp_id, "acc", eta=3, rungs=[1, 3])
    assert result["best_surviving_run_id"] == "r7"
    assert result["budget_used"] == 21
