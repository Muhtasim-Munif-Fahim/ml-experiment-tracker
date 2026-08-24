"""Tests for artifact download over the API and client."""

import io

from fastapi.testclient import TestClient


def upload_artifact(api: TestClient, run_id: str, name: str, payload: bytes):
    response = api.post(
        f"/runs/{run_id}/artifacts",
        data={"name": name, "artifact_type": "model", "metadata": "{}"},
        files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
    )
    assert response.status_code == 200
    return response.json()


def seed_run(api: TestClient) -> dict:
    experiment = api.post("/experiments/", json={"name": "artifacted"}).json()
    return api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()


def test_api_downloads_artifact_by_id_with_content_type(api, temp_storage):
    run = seed_run(api)
    uploaded = upload_artifact(api, run["id"], "weights.bin", b"binary-payload")

    response = api.get(f"/runs/{run['id']}/artifacts/{uploaded['artifact_id']}")
    assert response.status_code == 200
    assert response.content == b"binary-payload"
    assert response.headers["content-type"] == "application/octet-stream"
    assert (
        response.headers["content-disposition"].startswith("attachment;")
    )
    assert response.headers["x-artifact-sha256"] == uploaded["checksum_sha256"]
    assert len(response.headers["x-artifact-sha256"]) == 64


def test_api_downloads_artifact_by_name_for_legacy_records(api):
    run = seed_run(api)
    upload_artifact(api, run["id"], "notes.txt", b"hello note")

    response = api.get(f"/runs/{run['id']}/artifacts/notes.txt")
    assert response.status_code == 200
    assert response.content == b"hello note"
    assert response.headers["content-type"].startswith("text/plain")


def test_api_download_reports_missing_artifacts(api):
    run = seed_run(api)
    assert api.get(f"/runs/{run['id']}/artifacts/nope").status_code == 404
    assert api.get("/runs/missing/artifacts/nope").status_code == 404

    uploaded = upload_artifact(api, run["id"], "gone.bin", b"payload")
    entry = api.get(f"/runs/{run['id']}/artifacts/").json()[0]
    from pathlib import Path

    Path(entry["path"]).unlink()
    missing_file = api.get(f"/runs/{run['id']}/artifacts/{uploaded['artifact_id']}")
    assert missing_file.status_code == 404
    assert missing_file.json()["detail"] == "Artifact file missing"


def test_client_downloads_artifact_bytes_and_saves_copy(tracker, tmp_path):
    experiment = tracker.create_experiment("artifacted")
    run = tracker.create_run(experiment["id"], "trainer")
    source = tmp_path / "weights.bin"
    source.write_bytes(b"binary-payload")

    uploaded = tracker.upload_artifact(run["id"], "weights.bin", "model", str(source))
    destination = tmp_path / "copy.bin"

    content = tracker.download_artifact(
        run["id"], uploaded["artifact_id"], str(destination)
    )
    assert content == b"binary-payload"
    assert destination.read_bytes() == b"binary-payload"
    assert tracker.download_artifact(run["id"], "weights.bin") == b"binary-payload"
