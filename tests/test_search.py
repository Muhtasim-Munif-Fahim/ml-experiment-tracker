"""Tests for cross-experiment run search."""

import tempfile
from datetime import datetime, timedelta

import pytest

from src.models import Experiment, Run, RunStatus
from src.storage import LocalStorageBackend


def seed_search_corpus(storage):
    first = Experiment(name="vision")
    second = Experiment(name="text")
    storage.save_experiment(first.to_dict())
    storage.save_experiment(second.to_dict())

    base = datetime(2026, 8, 1, 12, 0, 0)
    runs = []
    for created_at, name, metric, status in (
        (base, "resnet baseline", ("accuracy", 0.7), "completed"),
        (base + timedelta(hours=1), "resnet tuned", ("accuracy", 0.9), "running"),
        (base + timedelta(hours=2), "bert draft", ("f1", 0.5), "failed"),
        (
            base + timedelta(hours=3),
            "bert tuned",
            ("f1", 0.8),
            "completed",
        ),
    ):
        run = Run(
            experiment_id=first.id if name.startswith("resnet") else second.id,
            name=name,
            created_at=created_at,
            updated_at=created_at,
        )
        if status == "completed":
            run.finish()
        elif status == "failed":
            run.finish(RunStatus.FAILED)
        if metric:
            run.log_metric(metric[0], metric[1], step=1)
        storage.save_run(run.to_dict())
        runs.append(run)
    return first, second, runs


def test_search_runs_spans_experiments_newest_first():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        _, _, runs = seed_search_corpus(storage)

        result = storage.search_runs()
        assert result["total"] == 4
        assert [run["name"] for run in result["runs"]] == [
            "bert tuned",
            "bert draft",
            "resnet tuned",
            "resnet baseline",
        ]
        assert result["runs"][0]["experiment_id"] != result["runs"][-1]["experiment_id"]


def test_search_runs_paginates_with_total_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        seed_search_corpus(storage)

        page_one = storage.search_runs(limit=2)
        assert [run["name"] for run in page_one["runs"]] == ["bert tuned", "bert draft"]
        assert page_one["total"] == 4

        page_two = storage.search_runs(limit=2, offset=2)
        assert [run["name"] for run in page_two["runs"]] == [
            "resnet tuned",
            "resnet baseline",
        ]

        past_end = storage.search_runs(offset=99)
        assert past_end["runs"] == [] and past_end["total"] == 4


def test_search_runs_combines_name_and_latest_metric_filters():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        seed_search_corpus(storage)

        result = storage.search_runs(
            name_contains="tuned", metric_name="accuracy", min_metric=0.8
        )
        assert [run["name"] for run in result["runs"]] == ["resnet tuned"]

        completed = storage.search_runs(statuses=["completed"])
        assert {run["name"] for run in completed["runs"]} == {
            "bert tuned",
            "resnet baseline",
        }


def test_search_runs_includes_archived_experiments():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        first, _, _ = seed_search_corpus(storage)
        storage.set_experiment_archived(first.id, archived=True)

        names = {run["name"] for run in storage.search_runs()["runs"]}
        assert {"resnet baseline", "resnet tuned"} <= names


def test_search_runs_validates_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        with pytest.raises(ValueError, match="metric_name"):
            storage.search_runs(min_metric=0.5)
        with pytest.raises(ValueError, match="min_metric"):
            storage.search_runs(metric_name="loss", min_metric=2, max_metric=1)
        with pytest.raises(ValueError, match="limit"):
            storage.search_runs(limit=0)
        with pytest.raises(ValueError, match="offset"):
            storage.search_runs(offset=-1)


def test_api_cross_experiment_search(api):
    vision = api.post("/experiments/", json={"name": "vision"}).json()
    text = api.post("/experiments/", json={"name": "text"}).json()
    api.post(f"/experiments/{vision['id']}/runs/", json={"name": "resnet-a"})
    api.post(f"/experiments/{text['id']}/runs/", json={"name": "bert-b"})
    api.post(f"/experiments/{text['id']}/runs/", json={"name": "bert-c"})

    filtered = api.get("/runs/search", params={"name_contains": "bert"})
    assert filtered.status_code == 200
    body = filtered.json()
    assert body["total"] == 2
    assert {run["name"] for run in body["runs"]} == {"bert-b", "bert-c"}

    paged = api.get("/runs/search", params={"limit": 1, "offset": 2})
    assert paged.json()["total"] >= 3 and len(paged.json()["runs"]) == 1

    bad = api.get("/runs/search", params={"limit": 0})
    assert bad.status_code == 400

    # the literal search route must not shadow run detail lookups
    some_run = api.get("/runs/search").json()["runs"][0]
    assert api.get(f"/runs/{some_run['id']}").json()["id"] == some_run["id"]


def test_client_search_runs_across_experiments(tracker):
    vision = tracker.create_experiment("vision")
    text = tracker.create_experiment("text")
    tracker.create_run(vision["id"], "resnet-a", params={"seed": 1})
    tracker.create_run(text["id"], "bert-b")

    result = tracker.search_runs(name_contains="net")
    assert result["total"] == 1
    assert result["runs"][0]["name"] == "resnet-a"

    everything = tracker.search_runs()
    assert everything["total"] == 2
    assert len(everything["runs"]) == 2
