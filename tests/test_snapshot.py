"""Tests for single-run snapshot export."""

import io
import json
import tempfile
from pathlib import Path

import pytest

from src.models import Artifact, ArtifactType, Experiment, Run
from src.storage import LocalStorageBackend


def seed_run_with_artifact(storage):
    experiment = Experiment(name="portable run")
    run = Run(experiment_id=experiment.id, name="baseline", params={"lr": 0.05})
    run.log_metric("accuracy", 0.9, step=2)
    run.log_artifact(
        Artifact("weights", ArtifactType.MODEL, "stored/weights", 128, checksum_sha256="a" * 64)
    )
    storage.save_experiment(experiment.to_dict())
    storage.save_run(run.to_dict())
    return experiment, run


def test_build_run_snapshot_contains_context_and_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment, run = seed_run_with_artifact(storage)

        snapshot = storage.build_run_snapshot(run.id)
        assert snapshot["schema_version"] == 1
        assert snapshot["snapshot_type"] == "run"
        assert snapshot["experiment"] == {"id": experiment.id, "name": "portable run"}
        assert snapshot["run"]["params"] == {"lr": 0.05}
        assert snapshot["run"]["metrics"][0]["value"] == 0.9
        # model-serialized records store the compact artifact shape without
        # checksums; API uploads persist them and surface in manifests below
        assert snapshot["artifact_manifest"] == [
            {
                "name": "weights",
                "type": "model",
                "size_bytes": 128,
                "checksum_sha256": None,
            }
        ]


def test_export_run_snapshot_writes_standalone_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        _, run = seed_run_with_artifact(storage)

        destination = Path(tmpdir) / "exports" / "run.json"
        written = storage.export_run_snapshot(run.id, destination)
        assert written == destination

        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["run"]["id"] == run.id

        with pytest.raises(KeyError, match="run not found"):
            storage.export_run_snapshot("missing", destination)


def test_api_run_snapshot_endpoint(api):
    experiment = api.post("/experiments/", json={"name": "portable"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/",
        json={"name": "baseline", "params": {"seed": 3}},
    ).json()
    api.post(f"/runs/{run['id']}/metrics", json={"name": "accuracy", "value": 0.9, "step": 1})
    api.post(
        f"/runs/{run['id']}/artifacts",
        data={"name": "notes.txt", "artifact_type": "config", "metadata": "{}"},
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )

    response = api.get(f"/runs/{run['id']}/snapshot")
    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["snapshot_type"] == "run"
    assert snapshot["experiment"]["id"] == experiment["id"]
    assert snapshot["artifact_manifest"][0]["name"] == "notes.txt"
    assert len(snapshot["artifact_manifest"][0]["checksum_sha256"]) == 64

    assert api.get("/runs/missing/snapshot").status_code == 404


def test_client_fetches_and_exports_run_snapshot(tracker, tmp_path):
    experiment = tracker.create_experiment("portable")
    tracker.update_experiment(experiment["id"], {"description": "kept"})
    run = tracker.create_run(experiment["id"], "baseline", params={"seed": 3})
    tracker.log_metric(run["id"], "accuracy", 0.9, step=1)

    snapshot = tracker.run_snapshot(run["id"])
    assert snapshot["run"]["metrics"][0]["value"] == 0.9

    destination = tmp_path / "snapshots" / "run.json"
    exported = tracker.export_run_snapshot(run["id"], str(destination))
    assert Path(exported).exists()
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["run"]["id"] == run["id"]
