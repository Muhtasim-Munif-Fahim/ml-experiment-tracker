"""Tests for ML Experiment Tracker"""

import uuid
import tempfile
from pathlib import Path

import pytest

from src.models import Experiment, Run, RunStatus, Param, Metric, Artifact, ArtifactType
from src.storage import LocalStorageBackend, StorageFactory


def test_experiment_creation():
    exp = Experiment(name="test_exp", description="test", tags=["test"])
    assert exp.name == "test_exp"
    assert exp.description == "test"
    assert "test" in exp.tags
    assert exp.id is not None


def test_run_creation():
    exp = Experiment(name="test_exp")
    run = Run(experiment_id="test-exp", name="test_run", params={"lr": 0.01})
    assert run.experiment_id == "test-exp"
    assert run.name == "test_run"
    assert run.status == RunStatus.RUNNING


def test_run_log_param():
    run = Run(experiment_id="test", name="test")
    run.log_param("lr", 0.01)
    assert run.params["lr"] == 0.01


def test_run_clone_copies_configuration_and_preserves_lineage():
    source = Run(
        experiment_id="test", name="baseline", params={"lr": 0.01}, tags={"stage": "base"}
    )
    clone = source.clone("candidate")

    assert clone.name == "candidate"
    assert clone.parent_run_id == source.id
    assert clone.params == source.params and clone.params is not source.params
    assert clone.tags == source.tags and clone.tags is not source.tags
    assert clone.metrics == []


def test_run_log_metric():
    run = Run(experiment_id="test", name="test")
    run.log_metric("loss", 0.5, step=1)
    run.log_metric("loss", 0.3, step=2)
    assert len(run.metrics) == 2
    assert run.metrics[0].name == "loss"
    assert run.metrics[0].value == 0.5
    assert run.metrics[0].step == 1


def test_run_log_metrics_records_a_shared_step():
    run = Run(experiment_id="test", name="test")
    run.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=4)

    assert [(metric.name, metric.value, metric.step) for metric in run.metrics] == [
        ("loss", 0.3, 4),
        ("accuracy", 0.9, 4),
    ]


def test_run_log_metrics_rejects_empty_mapping():
    with pytest.raises(ValueError, match="non-empty"):
        Run(experiment_id="test", name="test").log_metrics({})


def test_metric_summary_preserves_history_and_steps():
    run = Run(experiment_id="test", name="test")
    run.log_metric("accuracy", 0.7, step=1)
    run.log_metric("loss", 0.5, step=1)
    run.log_metric("accuracy", 0.9, step=2)

    summary = run.metric_summary("accuracy")
    assert summary == {
        "name": "accuracy",
        "count": 2,
        "min": 0.7,
        "max": 0.9,
        "mean": 0.8,
        "last": 0.9,
        "last_step": 2,
        "best_step": 2,
    }
    assert run.metric_summary("missing") is None


def test_metric_history_exports_only_requested_series():
    run = Run(experiment_id="test", name="test")
    run.log_metric("loss", 0.5, step=1)
    run.log_metric("accuracy", 0.8, step=1)
    run.log_metric("loss", 0.3, step=2)

    history = run.metric_history("loss")

    assert [point["value"] for point in history] == [0.5, 0.3]
    assert [point["step"] for point in history] == [1, 2]
    assert all(point["name"] == "loss" for point in history)
    assert run.metric_history("missing") == []


def test_metric_history_can_be_sliced_by_inclusive_steps():
    run = Run(experiment_id="test", name="test")
    for step in range(1, 5):
        run.log_metric("loss", 1.0 / step, step=step)

    history = run.metric_history("loss", start_step=2, end_step=3)
    assert [point["step"] for point in history] == [2, 3]

    with pytest.raises(ValueError, match="start_step"):
        run.metric_history("loss", start_step=3, end_step=2)


def test_storage_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        exp = Experiment(name="test", description="test")
        storage.save_experiment(exp.to_dict())
        loaded = storage.load_experiment(exp.id)
        assert loaded["name"] == exp.name
        assert loaded["id"] == exp.id

        run = Run(experiment_id=exp.id, name="test_run")
        storage.save_run(run.to_dict())
        loaded = storage.load_run(run.id)
        assert loaded["id"] == run.id

        runs = storage.list_runs(exp.id)
        assert len(runs) == 1


def test_artifact_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        model_path = Path(tmpdir) / "model.pkl"
        model_path.write_bytes(b"test data")

        saved_path = storage.save_artifact("model_1", model_path)
        loaded = storage.load_artifact(saved_path)
        assert loaded == b"test data"
        checksum = storage.artifact_checksum(saved_path)
        assert checksum == "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"
        assert storage.verify_artifact(saved_path, checksum)

        Path(saved_path).write_bytes(b"tampered")
        assert not storage.verify_artifact(saved_path, checksum)


def test_experiment_bundle_round_trip():
    with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
        source = LocalStorageBackend(source_dir)
        experiment = Experiment(name="portable")
        run = Run(experiment_id=experiment.id, name="baseline", params={"seed": 7})
        run.log_metric("accuracy", 0.91, step=1)
        run.finish()
        source.save_experiment(experiment.to_dict())
        source.save_run(run.to_dict())

        bundle = Path(source_dir) / "exports" / "experiment.json"
        source.export_experiment(experiment.id, bundle)

        target = LocalStorageBackend(target_dir)
        imported_id = target.import_experiment(bundle)
        assert imported_id == experiment.id
        assert target.load_experiment(imported_id)["name"] == "portable"
        imported_runs = target.list_runs(imported_id)
        assert imported_runs[0]["params"] == {"seed": 7}
        assert imported_runs[0]["metrics"][0]["value"] == 0.91


def test_experiment_bundle_refuses_to_overwrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = Experiment(name="existing")
        storage.save_experiment(experiment.to_dict())
        bundle = Path(tmpdir) / "experiment.json"
        storage.export_experiment(experiment.id, bundle)
        with pytest.raises(FileExistsError, match="already exists"):
            storage.import_experiment(bundle)


def test_query_runs_combines_status_tags_and_name_filters():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = Experiment(name="searchable")
        completed = Run(
            experiment_id=experiment.id,
            name="ResNet tuned",
            tags={"dataset": "cifar10", "stage": "candidate"},
        )
        completed.finish()
        running = Run(
            experiment_id=experiment.id,
            name="ResNet exploratory",
            tags={"dataset": "cifar10", "stage": "draft"},
        )
        other = Run(
            experiment_id=experiment.id,
            name="Transformer tuned",
            tags={"dataset": "text", "stage": "candidate"},
        )
        for run in (completed, running, other):
            storage.save_run(run.to_dict())

        matches = storage.query_runs(
            experiment.id,
            statuses=["completed"],
            tags={"dataset": "cifar10"},
            name_contains="resnet",
        )
        assert [run["id"] for run in matches] == [completed.id]


def test_query_runs_requires_all_requested_tags():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = Run(experiment_id="exp", name="candidate", tags={"dataset": "tabular"})
        storage.save_run(run.to_dict())
        assert storage.query_runs("exp", tags={"dataset": "tabular", "stage": "prod"}) == []


def test_query_runs_filters_by_latest_metric_value():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        inside = Run(experiment_id="exp", name="inside")
        inside.log_metric("accuracy", 0.7, step=1)
        inside.log_metric("accuracy", 0.9, step=2)
        outside = Run(experiment_id="exp", name="outside")
        outside.log_metric("accuracy", 0.6, step=1)
        for run in (inside, outside):
            storage.save_run(run.to_dict())

        matches = storage.query_runs(
            "exp", metric_name="accuracy", min_metric=0.8
        )

        assert [run["name"] for run in matches] == ["inside"]


def test_query_runs_requires_metric_name_for_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        with pytest.raises(ValueError, match="metric_name"):
            storage.query_runs("exp", min_metric=0.8)


def test_get_best_run():
    exp = Experiment(name="test")
    run_a = Run(experiment_id=exp.id, name="a")
    run_a.log_metric("accuracy", 0.8)
    run_a.finish(RunStatus.COMPLETED)
    run_b = Run(experiment_id=exp.id, name="b")
    run_b.log_metric("accuracy", 0.9)
    run_b.finish(RunStatus.COMPLETED)
    exp.add_run(run_a)
    exp.add_run(run_b)

    assert exp.get_best_run("accuracy").id == run_b.id
    assert exp.get_best_run("loss", maximize=False) is None


def test_metric_table_ranks_completed_runs_and_includes_params():
    exp = Experiment(name="test")
    for name, score, status in (
        ("baseline", 0.8, RunStatus.COMPLETED),
        ("candidate", 0.9, RunStatus.COMPLETED),
        ("unfinished", 0.99, RunStatus.RUNNING),
    ):
        run = Run(experiment_id=exp.id, name=name, params={"seed": 42})
        run.log_metric("accuracy", score)
        if status == RunStatus.COMPLETED:
            run.finish(status)
        exp.add_run(run)

    table = exp.metric_table("accuracy")
    assert [row["run_name"] for row in table] == ["candidate", "baseline"]
    assert table[0]["params"] == {"seed": 42}


def test_experiment_summary_aggregates_lifecycle_and_metrics():
    exp = Experiment(name="summary")
    completed = Run(experiment_id=exp.id, name="completed")
    completed.log_metrics({"accuracy": 0.8, "loss": 0.4})
    completed.finish(RunStatus.COMPLETED)
    failed = Run(experiment_id=exp.id, name="failed")
    failed.log_metric("accuracy", 0.6)
    failed.finish(RunStatus.FAILED)
    running = Run(experiment_id=exp.id, name="running")
    exp.runs.extend([completed, failed, running])

    summary = exp.summary()

    assert summary["run_count"] == 3
    assert summary["status_counts"] == {
        "running": 1,
        "completed": 1,
        "failed": 1,
        "aborted": 0,
    }
    assert summary["metrics"]["accuracy"] == {
        "count": 2,
        "min": 0.6,
        "max": 0.8,
        "mean": 0.7,
        "last": 0.6,
    }


def test_run_lineage_round_trips_and_orders_ancestors():
    exp = Experiment(name="lineage")
    root = Run(experiment_id=exp.id, name="baseline")
    child = Run(experiment_id=exp.id, name="tuned", parent_run_id=root.id)
    exp.add_run(root)
    exp.add_run(child)

    lineage = exp.run_lineage(child.id)
    assert [run.name for run in lineage] == ["baseline", "tuned"]
    restored = Experiment.from_dict(exp.to_dict())
    assert restored.runs[1].parent_run_id == root.id


def test_run_lineage_rejects_missing_parent():
    exp = Experiment(name="lineage")
    orphan = Run(experiment_id=exp.id, name="orphan", parent_run_id="missing")
    exp.add_run(orphan)
    with pytest.raises(ValueError, match="missing parent"):
        exp.run_lineage(orphan.id)


def test_compare_runs_reports_metric_deltas_and_changed_params():
    exp = Experiment(name="comparison")
    baseline = Run(experiment_id=exp.id, name="baseline", params={"lr": 0.1, "seed": 7})
    baseline.log_metric("accuracy", 0.80, step=1)
    baseline.log_metric("accuracy", 0.82, step=2)
    candidate = Run(experiment_id=exp.id, name="candidate", params={"lr": 0.05, "seed": 7})
    candidate.log_metric("accuracy", 0.87, step=2)
    candidate.log_metric("f1", 0.84, step=2)
    exp.add_run(baseline)
    exp.add_run(candidate)

    comparison = exp.compare_runs(baseline.id, candidate.id)
    assert comparison["metric_changes"]["accuracy"]["delta"] == pytest.approx(0.05)
    assert comparison["metric_changes"]["f1"]["baseline"] is None
    assert comparison["parameter_changes"] == {
        "lr": {"baseline": 0.1, "candidate": 0.05}
    }


def test_compare_runs_rejects_unknown_ids():
    exp = Experiment(name="comparison")
    run = Run(experiment_id=exp.id, name="baseline")
    exp.add_run(run)
    with pytest.raises(KeyError, match="missing"):
        exp.compare_runs(run.id, "missing")


def test_artifact_serialization():
    artifact = Artifact(
        name="model",
        artifact_type=ArtifactType.MODEL,
        path="/tmp/model.pkl",
        size_bytes=10,
        checksum_sha256="a" * 64,
    )
    data = artifact.to_dict()
    assert data["type"] == "model"
    assert data["size_bytes"] == 10
    assert data["checksum_sha256"] == "a" * 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_run_leaderboard_ranks_by_latest_metric():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        best = Run(experiment_id="exp", name="best")
        best.log_metric("accuracy", 0.7, step=1)
        best.log_metric("accuracy", 0.95, step=2)
        worst = Run(experiment_id="exp", name="worst")
        worst.log_metric("accuracy", 0.5, step=1)
        for run in (best, worst):
            storage.save_run(run.to_dict())

        ranking = storage.run_leaderboard("exp", "accuracy", maximize=True)

        assert [entry["name"] for entry in ranking] == ["best", "worst"]
        assert ranking[0]["value"] == 0.95


def test_run_leaderboard_minimize_orders_ascending():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        low = Run(experiment_id="exp", name="low")
        low.log_metric("loss", 0.1)
        high = Run(experiment_id="exp", name="high")
        high.log_metric("loss", 0.9)
        for run in (low, high):
            storage.save_run(run.to_dict())

        ranking = storage.run_leaderboard("exp", "loss", maximize=False)

        assert [entry["name"] for entry in ranking] == ["low", "high"]


def test_run_leaderboard_omits_runs_without_metric():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        with_metric = Run(experiment_id="exp", name="has")
        with_metric.log_metric("accuracy", 0.8)
        without = Run(experiment_id="exp", name="none")
        for run in (with_metric, without):
            storage.save_run(run.to_dict())

        ranking = storage.run_leaderboard("exp", "accuracy")

        assert [entry["name"] for entry in ranking] == ["has"]


def test_run_leaderboard_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        for i in range(5):
            run = Run(experiment_id="exp", name=f"run-{i}")
            run.log_metric("accuracy", i / 10)
            storage.save_run(run.to_dict())

        ranking = storage.run_leaderboard("exp", "accuracy", limit=2)

        assert len(ranking) == 2
        assert ranking[0]["value"] == 0.4


def test_run_leaderboard_requires_metric_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        with pytest.raises(ValueError, match="metric_name"):
            storage.run_leaderboard("exp", "")

def test_delete_run_removes_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = Run(experiment_id="exp", name="to-remove")
        storage.save_run(run.to_dict())
        assert storage.load_run(run.id) is not None

        assert storage.delete_run(run.id) is True
        assert storage.load_run(run.id) is None


def test_delete_run_missing_returns_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        assert storage.delete_run("does-not-exist") is False
