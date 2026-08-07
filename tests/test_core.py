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


def test_run_log_metric():
    run = Run(experiment_id="test", name="test")
    run.log_metric("loss", 0.5, step=1)
    run.log_metric("loss", 0.3, step=2)
    assert len(run.metrics) == 2
    assert run.metrics[0].name == "loss"
    assert run.metrics[0].value == 0.5
    assert run.metrics[0].step == 1


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


def test_artifact_serialization():
    artifact = Artifact(name="model", artifact_type=ArtifactType.MODEL, path="/tmp/model.pkl", size_bytes=10)
    data = artifact.to_dict()
    assert data["type"] == "model"
    assert data["size_bytes"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])