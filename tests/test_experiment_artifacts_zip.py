"""Tests for downloading all artifacts of an experiment as one zip archive."""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from src.models import Experiment, Run
from src.storage import LocalStorageBackend


def seed_experiment_with_artifacts(storage) -> Experiment:
    experiment = Experiment(name="zipped-exp")
    storage.save_experiment(experiment.to_dict())

    run_a = Run(experiment_id=experiment.id, name="train-a")
    run_b = Run(experiment_id=experiment.id, name="train-b")
    storage.save_run(run_a.to_dict())
    storage.save_run(run_b.to_dict())
    experiment.add_run(run_a)
    experiment.add_run(run_b)

    source_dir = Path(storage.base_path) / "seed"
    source_dir.mkdir(parents=True, exist_ok=True)

    entries_a = []
    for index, (name, payload) in enumerate(
        [("weights.bin", b"model-a"), ("config.json", b'{"lr": 0.1}')]
    ):
        source = source_dir / name
        source.write_bytes(payload)
        artifact_id = f"artifact_a_{index}"
        artifact_path = storage.save_artifact(artifact_id, source)
        entries_a.append(
            {
                "artifact_id": artifact_id,
                "name": name,
                "type": "model" if name.endswith(".bin") else "config",
                "path": artifact_path,
                "size_bytes": len(payload),
                "checksum_sha256": storage.artifact_checksum(artifact_path),
                "metadata": {},
                "created_at": "2026-08-31T00:00:00",
            }
        )

    entries_b = []
    for index, (name, payload) in enumerate(
        [("weights.bin", b"model-b"), ("plot.png", b"chart-data")]
    ):
        source = source_dir / name
        source.write_bytes(payload)
        artifact_id = f"artifact_b_{index}"
        artifact_path = storage.save_artifact(artifact_id, source)
        entries_b.append(
            {
                "artifact_id": artifact_id,
                "name": name,
                "type": "model" if name.endswith(".bin") else "plot",
                "path": artifact_path,
                "size_bytes": len(payload),
                "checksum_sha256": storage.artifact_checksum(artifact_path),
                "metadata": {},
                "created_at": "2026-08-31T01:00:00",
            }
        )

    run_data_a = storage.load_run(run_a.id)
    run_data_a["artifacts"] = entries_a
    storage.save_run(run_data_a)

    run_data_b = storage.load_run(run_b.id)
    run_data_b["artifacts"] = entries_b
    storage.save_run(run_data_b)

    return experiment


def test_export_experiment_artifacts_zip_bundles_all_runs():
    with pytest.raises(KeyError, match="experiment not found"):
        LocalStorageBackend("/tmp").export_experiment_artifacts_zip(
            "missing", Path("/tmp/out.zip")
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment_with_artifacts(storage)

        destination = Path(tmpdir) / "exports" / "artifacts.zip"
        written = storage.export_experiment_artifacts_zip(experiment.id, destination)
        assert written == destination

        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            assert "manifest.json" in names
            assert any("weights.bin" in name for name in names)
            assert any("config.json" in name for name in names)
            assert any("plot.png" in name for name in names)
            assert len([n for n in names if n != "manifest.json"]) == 4

            manifest = json.loads(archive.read("manifest.json"))
            assert len(manifest) == 4
            run_ids = {entry["run_id"] for entry in manifest}
            assert run_ids == {experiment.runs[0].id, experiment.runs[1].id}
            assert all(len(entry["checksum_sha256"]) == 64 for entry in manifest)


def test_export_experiment_artifacts_zip_handles_missing_files_and_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment = seed_experiment_with_artifacts(storage)

        run_data = storage.load_run(experiment.runs[0].id)
        run_data["artifacts"][0]["path"] = str(Path(tmpdir) / "gone.bin")
        storage.save_run(run_data)

        destination = Path(tmpdir) / "partial.zip"
        storage.export_experiment_artifacts_zip(experiment.id, destination)
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            assert "manifest.json" in names
            assert not any("gone.bin" in name for name in names)
            assert len([n for n in names if n != "manifest.json"]) == 3


def test_api_serves_experiment_artifacts_as_zip(api):
    experiment = api.post("/experiments/", json={"name": "zipped-exp"}).json()
    run_a = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "train-a"}
    ).json()
    run_b = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "train-b"}
    ).json()

    for name, payload in [
        ("weights.bin", b"model-a"),
        ("config.json", b'{"lr": 0.1}'),
    ]:
        api.post(
            f"/runs/{run_a['id']}/artifacts",
            data={"name": name, "artifact_type": "model", "metadata": "{}"},
            files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
        )
    for name, payload in [
        ("weights.bin", b"model-b"),
        ("plot.png", b"chart-data"),
    ]:
        api.post(
            f"/runs/{run_b['id']}/artifacts",
            data={"name": name, "artifact_type": "model", "metadata": "{}"},
            files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
        )

    response = api.get(f"/experiments/{experiment['id']}/artifacts.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "manifest.json" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest) == 4
        assert {entry["run_id"] for entry in manifest} == {run_a["id"], run_b["id"]}


def test_api_experiments_artifacts_zip_404_for_missing(api):
    assert api.get("/experiments/missing/artifacts.zip").status_code == 404


def test_client_downloads_experiment_artifacts_zip(tracker, tmp_path):
    experiment = tracker.create_experiment("client-zipped")
    run_a = tracker.create_run(experiment["id"], "run-a")
    run_b = tracker.create_run(experiment["id"], "run-b")

    source = tmp_path / "weights.bin"
    source.write_bytes(b"model-a")
    tracker.upload_artifact(run_a["id"], "weights.bin", "model", str(source))

    source2 = tmp_path / "plot.png"
    source2.write_bytes(b"chart-b")
    tracker.upload_artifact(run_b["id"], "plot.png", "plot", str(source2))

    destination = tmp_path / "downloads" / "experiment.zip"
    content = tracker.download_experiment_artifacts_zip(experiment["id"], str(destination))

    assert destination.exists()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert "manifest.json" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest) == 2
        names = {entry["name"] for entry in manifest}
        assert names == {"weights.bin", "plot.png"}
