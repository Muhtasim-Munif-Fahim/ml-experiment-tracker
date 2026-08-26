"""Tests for the per-experiment artifact inventory."""

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


def test_experiment_artifacts_lists_runs_newest_first(api, temp_storage):
    from src.models import Run
    from datetime import datetime, timedelta

    experiment = api.post("/experiments/", json={"name": "sweeps"}).json()
    base = datetime(2026, 8, 20, 9, 0, 0)
    older = api.post(
        f"/experiments/{experiment['id']}/runs/",
        json={"name": "early-run"},
    ).json()
    seeded_at = datetime.utcnow() + timedelta(hours=2)
    newer_run = Run(
        experiment_id=experiment["id"],
        name="late-run",
        created_at=seeded_at,
        updated_at=seeded_at,
    )
    temp_storage.save_run(newer_run.to_dict())

    first = upload_artifact(api, older["id"], "weights.bin", b"abc")
    upload_artifact(api, newer_run.id, "chart.png", b"defgh")

    response = api.get(f"/experiments/{experiment['id']}/artifacts")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [entry["run_name"] for entry in body["artifacts"]] == [
        "late-run",
        "early-run",
    ]
    early = body["artifacts"][1]
    assert early["run_id"] == older["id"]
    assert early["artifact_id"] == first["artifact_id"]
    assert early["name"] == "weights.bin"
    assert early["type"] == "model"
    assert early["size_bytes"] == 3
    assert early["sha256_prefix"]
    assert first["checksum_sha256"].startswith(early["sha256_prefix"])
    late = body["artifacts"][0]
    assert late["size_bytes"] == 5


def test_experiment_artifacts_supports_legacy_size_records(api, temp_storage):
    from src.models import Experiment

    experiment = Experiment(name="legacy")
    temp_storage.save_experiment(experiment.to_dict())
    run = {
        "id": "legacy-run",
        "experiment_id": experiment.id,
        "name": "old-trainer",
        "artifacts": [
            {"name": "checkpoint.pkl", "type": "model", "path": "gone", "size": 4096}
        ],
    }
    temp_storage.save_run(run)

    stored = temp_storage.experiment_artifacts(experiment.id)
    assert stored[0]["run_id"] == "legacy-run"
    assert stored[0]["size"] == 4096

    body = api.get(f"/experiments/{experiment.id}/artifacts").json()
    assert body["total"] == 1
    entry = body["artifacts"][0]
    assert entry["size_bytes"] == 4096
    assert entry["sha256_prefix"] == ""


def test_experiment_artifacts_paginates_and_validates(api):
    experiment = api.post("/experiments/", json={"name": "paged"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()
    for index in range(3):
        upload_artifact(api, run["id"], f"file-{index}.bin", bytes([index]))

    page_one = api.get(f"/experiments/{experiment['id']}/artifacts", params={"limit": 2}).json()
    page_two = api.get(
        f"/experiments/{experiment['id']}/artifacts",
        params={"limit": 2, "offset": 2},
    ).json()
    assert page_one["total"] == 3 and len(page_one["artifacts"]) == 2
    assert len(page_two["artifacts"]) == 1

    assert (
        api.get(
            f"/experiments/{experiment['id']}/artifacts", params={"limit": 0}
        ).status_code
        == 400
    )
    assert (
        api.get(
            f"/experiments/{experiment['id']}/artifacts", params={"offset": -1}
        ).status_code
        == 400
    )
    assert api.get("/experiments/missing/artifacts").status_code == 404


def test_client_pages_experiment_artifacts(tracker, tmp_path):
    experiment = tracker.create_experiment("client-inventory")
    run_a = tracker.create_run(experiment["id"], "run-a")
    run_b = tracker.create_run(experiment["id"], "run-b")
    source = tmp_path / "blob.bin"
    source.write_bytes(b"payload")

    uploaded_a = tracker.upload_artifact(run_a["id"], "blob.bin", "dataset", str(source))
    tracker.upload_artifact(run_b["id"], "other.bin", "model", str(source))

    inventory = tracker.experiment_artifacts(experiment["id"], limit=1)
    assert inventory["total"] == 2
    assert len(inventory["artifacts"]) == 1
    names = {entry["name"] for entry in inventory["artifacts"]}
    assert names <= {"blob.bin", "other.bin"}
    assert uploaded_a["artifact_id"] in {
        entry["artifact_id"] for entry in tracker.experiment_artifacts(experiment["id"])["artifacts"]
    }