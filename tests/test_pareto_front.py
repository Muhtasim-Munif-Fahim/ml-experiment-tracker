"""Tests for multi-objective Pareto-front run selection."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend

# accuracy up, latency down. "balanced" and "fast" are on the front;
# "slow" is dominated by "balanced" on both objectives at once.
FLEET = {
    "accurate": (0.95, 100.0),
    "balanced": (0.90, 40.0),
    "fast": (0.80, 10.0),
    "slow": (0.85, 60.0),
}

OBJECTIVES = [
    {"metric": "accuracy", "maximize": True},
    {"metric": "latency", "maximize": False},
]


def _seed(storage, fleet=FLEET):
    experiment = Experiment(name="multi-objective")
    storage.save_experiment(experiment.to_dict())
    for name, (accuracy, latency) in fleet.items():
        run = Run(experiment_id=experiment.id, name=name)
        run.log_metric("accuracy", accuracy, step=1)
        run.log_metric("latency", latency, step=1)
        storage.save_run(run.to_dict())
    return experiment


@pytest.fixture()
def seeded(temp_storage):
    return temp_storage, _seed(temp_storage)


def _names(entries):
    return {entry["name"] for entry in entries}


class TestParetoFront:
    def test_dominated_runs_are_excluded(self, seeded):
        storage, experiment = seeded
        front = storage.experiment_pareto_front(experiment.id, OBJECTIVES)
        assert _names(front) == {"accurate", "balanced", "fast"}

    def test_a_dominated_run_is_named_when_requested(self, seeded):
        storage, experiment = seeded
        entries = storage.experiment_pareto_front(
            experiment.id, OBJECTIVES, include_dominated=True
        )
        assert _names(entries) == set(FLEET)
        slow = next(entry for entry in entries if entry["name"] == "slow")
        assert slow["rank"] == 1

    def test_front_members_have_rank_zero(self, seeded):
        storage, experiment = seeded
        front = storage.experiment_pareto_front(experiment.id, OBJECTIVES)
        assert all(entry["rank"] == 0 for entry in front)

    def test_entries_carry_every_objective_value(self, seeded):
        storage, experiment = seeded
        front = storage.experiment_pareto_front(experiment.id, OBJECTIVES)
        balanced = next(entry for entry in front if entry["name"] == "balanced")
        assert balanced["values"] == {"accuracy": 0.90, "latency": 40.0}

    def test_front_is_ordered_by_how_many_runs_each_beats(self, seeded):
        storage, experiment = seeded
        front = storage.experiment_pareto_front(experiment.id, OBJECTIVES)
        counts = [entry["dominated_count"] for entry in front]
        assert counts == sorted(counts, reverse=True)
        assert front[0]["name"] == "balanced"

    def test_a_single_objective_reduces_to_the_best_run(self, seeded):
        storage, experiment = seeded
        front = storage.experiment_pareto_front(
            experiment.id, [{"metric": "accuracy", "maximize": True}]
        )
        assert _names(front) == {"accurate"}

    def test_minimizing_a_single_objective_flips_the_winner(self, seeded):
        storage, experiment = seeded
        front = storage.experiment_pareto_front(
            experiment.id, [{"metric": "accuracy", "maximize": False}]
        )
        assert _names(front) == {"fast"}

    def test_maximize_defaults_to_true(self, seeded):
        storage, experiment = seeded
        front = storage.experiment_pareto_front(experiment.id, [{"metric": "accuracy"}])
        assert _names(front) == {"accurate"}

    def test_runs_equal_on_every_objective_both_survive(self, temp_storage):
        experiment = _seed(temp_storage, {"twin-a": (0.9, 10.0), "twin-b": (0.9, 10.0)})
        front = temp_storage.experiment_pareto_front(experiment.id, OBJECTIVES)
        assert _names(front) == {"twin-a", "twin-b"}
        assert all(entry["dominated_count"] == 0 for entry in front)

    def test_a_run_missing_an_objective_is_excluded(self, temp_storage):
        experiment = Experiment(name="partial")
        temp_storage.save_experiment(experiment.to_dict())
        complete = Run(experiment_id=experiment.id, name="complete")
        complete.log_metric("accuracy", 0.9, step=1)
        complete.log_metric("latency", 10.0, step=1)
        temp_storage.save_run(complete.to_dict())
        partial = Run(experiment_id=experiment.id, name="partial")
        partial.log_metric("accuracy", 0.99, step=1)
        temp_storage.save_run(partial.to_dict())

        front = temp_storage.experiment_pareto_front(experiment.id, OBJECTIVES)
        assert _names(front) == {"complete"}

    def test_the_latest_metric_value_is_used(self, temp_storage):
        experiment = Experiment(name="stepped")
        temp_storage.save_experiment(experiment.to_dict())
        run = Run(experiment_id=experiment.id, name="improving")
        run.log_metric("accuracy", 0.10, step=1)
        run.log_metric("accuracy", 0.99, step=2)
        run.log_metric("latency", 5.0, step=1)
        temp_storage.save_run(run.to_dict())
        front = temp_storage.experiment_pareto_front(experiment.id, OBJECTIVES)
        assert front[0]["values"]["accuracy"] == 0.99

    def test_an_experiment_with_no_runs_gives_an_empty_front(self, temp_storage):
        experiment = Experiment(name="empty")
        temp_storage.save_experiment(experiment.to_dict())
        assert temp_storage.experiment_pareto_front(experiment.id, OBJECTIVES) == []

    def test_a_missing_experiment_raises(self, temp_storage):
        with pytest.raises(KeyError, match="experiment not found"):
            temp_storage.experiment_pareto_front("nope", OBJECTIVES)

    def test_no_objectives_is_rejected(self, seeded):
        storage, experiment = seeded
        with pytest.raises(ValueError, match="at least one objective"):
            storage.experiment_pareto_front(experiment.id, [])

    def test_an_objective_without_a_metric_is_rejected(self, seeded):
        storage, experiment = seeded
        with pytest.raises(ValueError, match="requires a metric name"):
            storage.experiment_pareto_front(experiment.id, [{"maximize": True}])


class TestParetoApi:
    def test_endpoint_returns_the_front(self, api, temp_storage):
        experiment = _seed(temp_storage)
        response = api.post(f"/experiments/{experiment.id}/pareto", json=OBJECTIVES)
        assert response.status_code == 200
        assert _names(response.json()) == {"accurate", "balanced", "fast"}

    def test_endpoint_honours_include_dominated(self, api, temp_storage):
        experiment = _seed(temp_storage)
        response = api.post(
            f"/experiments/{experiment.id}/pareto",
            json=OBJECTIVES,
            params={"include_dominated": True},
        )
        assert response.status_code == 200
        assert _names(response.json()) == set(FLEET)

    def test_unknown_experiment_is_404(self, api):
        response = api.post("/experiments/missing/pareto", json=OBJECTIVES)
        assert response.status_code == 404

    def test_empty_objectives_is_400(self, api, temp_storage):
        experiment = _seed(temp_storage)
        response = api.post(f"/experiments/{experiment.id}/pareto", json=[])
        assert response.status_code == 400


class TestParetoClient:
    def test_client_fetches_the_front(self, tracker, temp_storage):
        experiment = _seed(temp_storage)
        front = tracker.experiment_pareto_front(experiment.id, OBJECTIVES)
        assert _names(front) == {"accurate", "balanced", "fast"}
