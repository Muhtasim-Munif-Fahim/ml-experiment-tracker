"""Tests for downloading a run's artifacts as one zip archive."""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def seed_run_with_artifacts(storage) -> Run:
    experiment = Experiment(name="zipped")
    run = Run(experiment_id=experiment.id, name="trainer")
    storage.save_experiment(experiment.to_dict())
    storage.save_run(run.to_dict())

    source_dir = Path(storage.base_path) / "seed"
    source_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (name, payload) in enumerate(
        [("weights.bin", b"binary-payload"), ("config.json", b'{"lr": 0.1}')]
    ):
        source = source_dir / name
        source.write_bytes(payload)
        artifact_id = f"artifact_{index}"
        artifact_path = storage.save_artifact(artifact_id, source)
        entries.append(
            {
                "artifact_id": artifact_id,
                "name": name,
                "type": "model" if name.endswith(".bin") else "config",
                "path": artifact_path,
                "size_bytes": len(payload),
                "checksum_sha256": storage.artifact_checksum(artifact_path),
                "metadata": {},
                "created_at": "2026-08-29T00:00:00",
            }
        )

    run_data = storage.load_run(run.id)
    run_data["artifacts"] = entries
    storage.save_run(run_data)
    return run


def test_export_run_artifacts_zip_bundles_bytes_and_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run_with_artifacts(storage)

        destination = Path(tmpdir) / "exports" / "artifacts.zip"
        written = storage.export_run_artifacts_zip(run.id, destination)
        assert written == destination

        with zipfile.ZipFile(destination) as archive:
            assert set(archive.namelist()) == {
                "manifest.json",
                "weights.bin",
                "config.json",
            }
            assert archive.read("weights.bin") == b"binary-payload"
            assert archive.read("config.json") == b'{"lr": 0.1}'
            manifest = json.loads(archive.read("manifest.json"))
            assert {entry["name"] for entry in manifest} == {
                "weights.bin",
                "config.json",
            }
            assert all(len(entry["checksum_sha256"]) == 64 for entry in manifest)


def test_export_run_artifacts_zip_handles_missing_runs_and_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        run = seed_run_with_artifacts(storage)

        run_data = storage.load_run(run.id)
        run_data["artifacts"][0]["path"] = str(Path(tmpdir) / "gone.bin")
        storage.save_run(run_data)

        destination = Path(tmpdir) / "artifacts.zip"
        storage.export_run_artifacts_zip(run.id, destination)
        with zipfile.ZipFile(destination) as archive:
            # the stale record is skipped, the surviving one is kept
            assert "config.json" in archive.namelist()
            assert "weights.bin" not in archive.namelist()

        with pytest.raises(KeyError, match="run not found"):
            storage.export_run_artifacts_zip("missing", destination)


def seed_run_uploaded(api) -> dict:
    experiment = api.post("/experiments/", json={"name": "zipped"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()
    for name, payload in [("weights.bin", b"binary-payload"), ("config.json", b'{"lr": 0.1}')]:
        response = api.post(
            f"/runs/{run['id']}/artifacts",
            data={"name": name, "artifact_type": "model", "metadata": "{}"},
            files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
        )
        assert response.status_code == 200
    return run


def test_api_serves_run_artifacts_as_zip(api):
    run = seed_run_uploaded(api)

    response = api.get(f"/runs/{run['id']}/artifacts.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "weights.bin",
            "config.json",
        }
        assert archive.read("weights.bin") == b"binary-payload"

    assert api.get("/runs/missing/artifacts.zip").status_code == 404


def test_client_downloads_run_artifacts_zip(tracker, tmp_path):
    experiment = tracker.create_experiment("zipped")
    run = tracker.create_run(experiment["id"], "trainer")
    source = tmp_path / "weights.bin"
    source.write_bytes(b"binary-payload")
    tracker.upload_artifact(run["id"], "weights.bin", "model", str(source))

    destination = tmp_path / "downloads" / "artifacts.zip"
    content = tracker.download_run_artifacts_zip(run["id"], str(destination))

    assert destination.exists()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert archive.read("weights.bin") == b"binary-payload"
        assert json.loads(archive.read("manifest.json"))[0]["name"] == "weights.bin"
