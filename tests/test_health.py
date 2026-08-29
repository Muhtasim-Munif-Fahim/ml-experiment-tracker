"""Tests for the health endpoint and its storage statistics."""

import io

from src.models import Artifact, ArtifactType, Experiment, Run


def upload_artifact(api, run_id: str, name: str, payload: bytes):
    return api.post(
        f"/runs/{run_id}/artifacts",
        data={"name": name, "artifact_type": "model", "metadata": "{}"},
        files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
    )


def test_storage_stats_empty_store(temp_storage):
    assert temp_storage.storage_stats() == {
        "experiment_count": 0,
        "run_count": 0,
        "artifact_count": 0,
        "artifact_bytes": 0,
        "artifact_store_bytes": 0,
    }


def test_storage_stats_counts_records_and_artifact_bytes(temp_storage, tmp_path):
    experiment = Experiment(name="stats")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="trainer")
    temp_storage.save_run(run.to_dict())

    source = tmp_path / "blob.bin"
    source.write_bytes(b"payload")
    artifact_path = temp_storage.save_artifact("artifact_stats", source)
    run.artifacts.append(
        Artifact(
            name="blob.bin",
            artifact_type=ArtifactType.MODEL,
            path=artifact_path,
            size_bytes=7,
        )
    )
    temp_storage.save_run(run.to_dict())

    stats = temp_storage.storage_stats()
    assert stats["experiment_count"] == 1
    assert stats["run_count"] == 1
    assert stats["artifact_count"] == 1
    assert stats["artifact_bytes"] == 7
    assert stats["artifact_store_bytes"] == 7


def test_health_reports_empty_storage_counts(api):
    body = api.get("/health").json()
    assert body["status"] == "healthy"
    assert body["timestamp"]
    assert body["storage"]["experiment_count"] == 0
    assert body["storage"]["run_count"] == 0
    assert body["storage"]["artifact_count"] == 0
    assert body["storage"]["artifact_bytes"] == 0


def test_health_tracks_counts_across_records(api):
    experiment = api.post("/experiments/", json={"name": "vision"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()
    upload_artifact(api, run["id"], "weights.bin", b"bytes-here")

    body = api.get("/health").json()
    storage = body["storage"]
    assert storage["experiment_count"] == 1
    assert storage["run_count"] == 1
    assert storage["artifact_count"] == 1
    assert storage["artifact_bytes"] == len(b"bytes-here")
    assert storage["artifact_store_bytes"] > 0


def test_health_stays_open_without_auth(monkeypatch, api):
    import src.api as api_module

    monkeypatch.setenv(api_module.API_TOKEN_ENV_VAR, "secret")
    assert api.get("/health").status_code == 200


def test_client_health_reports_storage_stats(tracker):
    experiment = tracker.create_experiment("client-health")
    tracker.create_run(experiment["id"], "trainer")

    body = tracker.health()
    assert body["status"] == "healthy"
    assert body["storage"]["experiment_count"] == 1
    assert body["storage"]["run_count"] == 1
