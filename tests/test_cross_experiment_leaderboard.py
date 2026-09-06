"""Tests for the cross-experiment run leaderboard."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run, RunStatus
from src.storage import LocalStorageBackend


def _seed_cross(storage):
    exp_a = Experiment(name="cv-a")
    exp_b = Experiment(name="cv-b")
    storage.save_experiment(exp_a.to_dict())
    storage.save_experiment(exp_b.to_dict())

    names = ("a-high", "a-low", "b-mid")
    values = (0.95, 0.81, 0.90)
    runs = {}
    for run_name, value in zip(names, values):
        run = Run(
            experiment_id=exp_a.id if run_name.startswith("a") else exp_b.id,
            name=run_name,
        )
        run.log_metric("val_accuracy", value, step=1)
        storage.save_run(run.to_dict())
        runs[run_name] = run
    return exp_a, exp_b, runs


def test_cross_experiment_leaderboard_ranks_across_experiments(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    exp_a, exp_b, _ = _seed_cross(storage)
    ranking = storage.cross_experiment_leaderboard("val_accuracy")
    assert [entry["name"] for entry in ranking] == ["a-high", "b-mid", "a-low"]
    assert [entry["experiment_id"] for entry in ranking] == [
        exp_a.id,
        exp_b.id,
        exp_a.id,
    ]
    assert ranking[0]["value"] == 0.95
    assert ranking[0]["step"] == 1


def test_cross_experiment_leaderboard_minimize_orders_ascending(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    exp = Experiment(name="loss-land")
    storage.save_experiment(exp.to_dict())
    for value in (0.4, 0.1, 0.9):
        run = Run(experiment_id=exp.id, name=f"r-{value}")
        run.log_metric("loss", value, step=1)
        storage.save_run(run.to_dict())
    ranking = storage.cross_experiment_leaderboard("loss", maximize=False)
    assert [entry["value"] for entry in ranking] == [0.1, 0.4, 0.9]


def test_cross_experiment_leaderboard_limit_truncates(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    _seed_cross(storage)
    ranking = storage.cross_experiment_leaderboard("val_accuracy", limit=2)
    assert len(ranking) == 2
    assert [entry["name"] for entry in ranking] == ["a-high", "b-mid"]


def test_cross_experiment_leaderboard_omits_runs_without_metric(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    exp_a, _, _ = _seed_cross(storage)
    no_metric = Run(experiment_id=exp_a.id, name="no-metric")
    no_metric.log_metric("loss", 0.1, step=1)
    storage.save_run(no_metric.to_dict())
    ranking = storage.cross_experiment_leaderboard("val_accuracy")
    names = [entry["name"] for entry in ranking]
    assert "no-metric" not in names


def test_cross_experiment_leaderboard_requires_metric_name(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    _seed_cross(storage)
    with pytest.raises(ValueError, match="metric_name"):
        storage.cross_experiment_leaderboard("")
    with pytest.raises(ValueError, match="limit"):
        storage.cross_experiment_leaderboard("val_accuracy", limit=0)


def test_cross_experiment_leaderboard_excludes_archived_by_default(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    _seed_cross(storage)
    archived_exp = Experiment(name="archived-exp")
    storage.save_experiment(archived_exp.to_dict())
    storage.set_experiment_archived(archived_exp.id, archived=True)
    run = Run(experiment_id=archived_exp.id, name="archived-best")
    run.log_metric("val_accuracy", 0.99, step=1)
    storage.save_run(run.to_dict())

    default_ranking = storage.cross_experiment_leaderboard("val_accuracy")
    assert "archived-best" not in [entry["name"] for entry in default_ranking]

    archived_ranking = storage.cross_experiment_leaderboard(
        "val_accuracy", include_archived=True
    )
    assert "archived-best" in [entry["name"] for entry in archived_ranking]
    assert archived_ranking[0]["name"] == "archived-best"


def test_cross_experiment_leaderboard_empty_store(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    assert storage.cross_experiment_leaderboard("val_accuracy") == []


def test_api_cross_experiment_leaderboard(api, temp_storage):
    _seed_cross(temp_storage)
    response = api.get("/leaderboard", params={"metric": "val_accuracy"})
    assert response.status_code == 200
    body = response.json()
    assert [entry["name"] for entry in body] == ["a-high", "b-mid", "a-low"]
    assert "experiment_id" in body[0]


def test_api_cross_experiment_leaderboard_requires_metric(api):
    response = api.get("/leaderboard")
    assert response.status_code == 422


def test_api_cross_experiment_leaderboard_csv(api, temp_storage):
    _seed_cross(temp_storage)
    response = api.get("/leaderboard.csv", params={"metric": "val_accuracy", "limit": 2})
    assert response.status_code == 200
    text = response.text
    rows = [row for row in text.splitlines() if row]
    assert rows[0] == "rank,experiment_id,run_id,name,value,step"
    assert rows[1].startswith("1,")
    assert rows[2].startswith("2,")


def test_client_cross_experiment_leaderboard(tracker):
    exp_a = tracker.create_experiment("client-cv-a")
    exp_b = tracker.create_experiment("client-cv-b")
    rows = {
        "a-high": (exp_a, 0.95),
        "a-low": (exp_a, 0.81),
        "b-mid": (exp_b, 0.90),
    }
    run_ids = {}
    for name, (exp, _value) in rows.items():
        run = tracker.create_run(exp["id"], name, params={})
        run_ids[name] = run["id"]
    for name, (_exp, value) in rows.items():
        tracker.log_metric(run_ids[name], "val_accuracy", value, step=1)

    ranking = tracker.cross_experiment_leaderboard("val_accuracy", limit=10)
    assert [entry["name"] for entry in ranking] == ["a-high", "b-mid", "a-low"]
    assert ranking[0]["experiment_id"] == exp_a["id"]

    csv_text = tracker.cross_experiment_leaderboard_csv("val_accuracy", limit=3)
    assert "rank,experiment_id,run_id,name,value,step" in csv_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
