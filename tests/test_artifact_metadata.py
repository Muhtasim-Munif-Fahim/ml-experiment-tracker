"""Tests for updating artifact metadata without re-uploading bytes."""

import io
import tempfile

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def seed_run_with_artifact(storage) -> Run:
    experiment = Experiment(name="artifacted")
    run = Run(experiment_id=experiment.id, name="trainer")
    storage.save_experiment(experiment.to_dict())
    storage.save_run(run.to_dict())
    run_data = storage.load_run(run.id)
    run_data["artifacts"] = [
        {
            "artifact_id": "artifact_abc123",
            "name": "weights.bin",
            "type": "model",
            "path": "stored/weights.bin",
            "size_bytes": 5,
            "checksum_sha256": "a" * 64,
            "metadata": {"framework": "pytorch"},
            "created_at": "2026-08-29T00:00:00",
        }
    ]
    storage.save_run(run_data)
    return run


def test_update_artifact_renames_and_replaces_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run_with_artifact(storage)

        updated = storage.update_artifact(
            run.id,
            "artifact_abc123",
            {"name": "weights-final.bin", "metadata": {"framework": "jax"}},
        )
        assert updated["name"] == "weights-final.bin"
        assert updated["metadata"] == {"framework": "jax"}
        assert updated["type"] == "model"
        # bytes are untouched: path and checksum still reference the same file
        assert updated["path"] == "stored/weights.bin"
        assert updated["checksum_sha256"] == "a" * 64

        stored = storage.load_run(run.id)["artifacts"][0]
        assert stored["name"] == "weights-final.bin"
        assert stored["metadata"] == {"framework": "jax"}


def test_update_artifact_matches_by_name_and_changes_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run_with_artifact(storage)

        updated = storage.update_artifact(
            run.id, "weights.bin", {"artifact_type": "dataset"}
        )
        assert updated["type"] == "dataset"
        assert storage.load_run(run.id)["artifacts"][0]["type"] == "dataset"


def test_update_artifact_validates_name_type_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run_with_artifact(storage)

        with pytest.raises(ValueError, match="non-empty"):
            storage.update_artifact(run.id, "artifact_abc123", {"name": "  "})
        with pytest.raises(ValueError, match="unknown artifact type"):
            storage.update_artifact(run.id, "artifact_abc123", {"artifact_type": "bin"})
        with pytest.raises(ValueError, match="mapping"):
            storage.update_artifact(run.id, "artifact_abc123", {"metadata": "nope"})


def test_update_artifact_missing_run_and_artifact():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run_with_artifact(storage)

        with pytest.raises(KeyError, match="run not found"):
            storage.update_artifact("missing", "x", {"name": "y"})
        assert (
            storage.update_artifact(run.id, "artifact_nope", {"name": "y"}) is None
        )


def upload_artifact(api, run_id: str, name: str, payload: bytes):
    response = api.post(
        f"/runs/{run_id}/artifacts",
        data={"name": name, "artifact_type": "model", "metadata": "{}"},
        files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
    )
    assert response.status_code == 200
    return response.json()


def test_api_patch_updates_artifact_and_download_by_new_name(api):
    experiment = api.post("/experiments/", json={"name": "artifacted"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()
    uploaded = upload_artifact(api, run["id"], "weights.bin", b"bytes")

    patched = api.patch(
        f"/runs/{run['id']}/artifacts/{uploaded['artifact_id']}",
        json={"name": "final.bin", "artifact_type": "dataset", "metadata": {"k": "v"}},
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["name"] == "final.bin"
    assert body["type"] == "dataset"
    assert body["metadata"] == {"k": "v"}

    detail = api.get(f"/runs/{run['id']}").json()
    assert detail["artifacts"][0]["name"] == "final.bin"

    downloaded = api.get(f"/runs/{run['id']}/artifacts/final.bin")
    assert downloaded.status_code == 200
    assert downloaded.content == b"bytes"


def test_api_patch_rejects_unknown_inputs(api):
    experiment = api.post("/experiments/", json={"name": "artifacted"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()
    uploaded = upload_artifact(api, run["id"], "weights.bin", b"bytes")

    bad_type = api.patch(
        f"/runs/{run['id']}/artifacts/{uploaded['artifact_id']}",
        json={"artifact_type": "binary"},
    )
    assert bad_type.status_code == 400

    assert (
        api.patch(
            f"/runs/{run['id']}/artifacts/artifact_nope",
            json={"name": "x"},
        ).status_code
        == 404
    )
    assert (
        api.patch("/runs/missing/artifacts/artifact_abc", json={"name": "x"}).status_code
        == 404
    )


def test_client_updates_artifact_metadata(tracker, tmp_path):
    experiment = tracker.create_experiment("artifacted")
    run = tracker.create_run(experiment["id"], "trainer")
    source = tmp_path / "weights.bin"
    source.write_bytes(b"binary-payload")
    uploaded = tracker.upload_artifact(run["id"], "weights.bin", "model", str(source))

    updated = tracker.update_artifact(
        run["id"],
        uploaded["artifact_id"],
        {"name": "final.bin", "metadata": {"framework": "pytorch"}},
    )
    assert updated["name"] == "final.bin"

    detail = tracker.get_run(run["id"])
    assert detail["artifacts"][0]["name"] == "final.bin"
    assert tracker.download_artifact(run["id"], "final.bin") == b"binary-payload"
