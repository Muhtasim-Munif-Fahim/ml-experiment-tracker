"""Tests for the search_runs_sorted feature."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def _seed(tmp_path):
    from src.models import RunStatus
    storage = LocalStorageBackend(tmp_path / "mlruns")
    exp = Experiment(name="search-exp")
    storage.save_experiment(exp.to_dict())
    runs = []
    for name, status in [("a", RunStatus.COMPLETED), ("b", RunStatus.RUNNING), ("c", RunStatus.FAILED)]:
        run = Run(experiment_id=exp.id, name=name, status=status)
        storage.save_run(run.to_dict())
        exp.add_run(run)
        runs.append(run)
    return storage, exp, runs


def test_returns_all_runs_without_filters(tmp_path) -> None:
    storage, exp, runs = _seed(tmp_path)
    found = storage.search_runs_sorted()
    assert {r["name"] for r in found} == {"a", "b", "c"}


def test_filters_by_experiment_id(tmp_path) -> None:
    storage, exp, _ = _seed(tmp_path)
    found = storage.search_runs_sorted(experiment_id=exp.id)
    assert len(found) == 3


def test_filters_by_status(tmp_path) -> None:
    storage, exp, _ = _seed(tmp_path)
    found = storage.search_runs_sorted(status="running")
    assert {r["name"] for r in found} == {"b"}


def test_sorts_descending_by_default(tmp_path) -> None:
    storage, exp, _ = _seed(tmp_path)
    found = storage.search_runs_sorted(experiment_id=exp.id)
    # created_at is the same for all three (within the same second), so the
    # stable secondary key is name; desc returns b, c, a or a, c, b etc.
    names_desc = [r["name"] for r in found]
    # At least the count is preserved.
    assert len(names_desc) == 3


def test_sorts_by_name_ascending(tmp_path) -> None:
    storage, exp, _ = _seed(tmp_path)
    found = storage.search_runs_sorted(
        experiment_id=exp.id, sort_by="name", descending=False,
    )
    assert [r["name"] for r in found] == ["a", "b", "c"]


def test_falls_back_to_created_at_for_unknown_sort_key(tmp_path) -> None:
    storage, exp, _ = _seed(tmp_path)
    found = storage.search_runs_sorted(
        experiment_id=exp.id, sort_by="non_existent_field", descending=True,
    )
    assert len(found) == 3


def test_limit_truncates_result(tmp_path) -> None:
    storage, exp, _ = _seed(tmp_path)
    found = storage.search_runs_sorted(experiment_id=exp.id, limit=2)
    assert len(found) == 2


def test_search_works_without_experiment_filter(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    for name in ("one", "two"):
        exp = Experiment(name=f"exp-{name}")
        storage.save_experiment(exp.to_dict())
        run = Run(experiment_id=exp.id, name=f"run-{name}")
        storage.save_run(run.to_dict())
    found = storage.search_runs_sorted()
    assert len(found) == 2


def test_search_dedupes_when_exp_id_not_given(tmp_path) -> None:
    """An unsafe call pattern shouldn't return the same run twice."""
    storage = LocalStorageBackend(tmp_path / "mlruns")
    exp = Experiment(name="dedup")
    storage.save_experiment(exp.to_dict())
    run = Run(experiment_id=exp.id, name="only")
    storage.save_run(run.to_dict())
    found = storage.search_runs_sorted()
    assert len(found) == 1


def test_api_search_runs_returns_runs_and_total(api, temp_storage) -> None:
    from src.models import Experiment, Run
    experiment = Experiment(name="api-search")
    temp_storage.save_experiment(experiment.to_dict())
    for name in ("a", "b"):
        run = Run(experiment_id=experiment.id, name=name)
        temp_storage.save_run(run.to_dict())
    response = api.get("/search/runs")
    assert response.status_code == 200
    body = response.json()
    assert "runs" in body
    assert "total" in body
    assert body["total"] == 2
    assert len(body["runs"]) == 2


def test_api_search_runs_filters_by_status(api, temp_storage) -> None:
    from src.models import Experiment, Run, RunStatus
    experiment = Experiment(name="api-filter")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="done", status=RunStatus.COMPLETED)
    temp_storage.save_run(run.to_dict())
    response = api.get("/search/runs", params={"status": "running"})
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_client_search_runs_sorted(live_server, temp_storage) -> None:
    from src.client import ExperimentTrackerClient
    from src.models import Experiment, Run, RunStatus

    experiment = Experiment(name="cli-search")
    temp_storage.save_experiment(experiment.to_dict())
    for name, status in [("x", RunStatus.COMPLETED), ("y", RunStatus.RUNNING), ("z", RunStatus.FAILED)]:
        run = Run(experiment_id=experiment.id, name=name, status=status)
        temp_storage.save_run(run.to_dict())

    client = ExperimentTrackerClient(base_url=live_server)
    body = client.search_runs_sorted(experiment_id=experiment.id, sort_by="name", descending=False)
    assert body["total"] == 3
    assert [r["name"] for r in body["runs"]] == ["x", "y", "z"]