"""Tests for X-Total-Count pagination headers and client page objects."""

import io

from fastapi.testclient import TestClient

from src.client import PaginatedList


def upload_artifact(api: TestClient, run_id: str, name: str, payload: bytes):
    response = api.post(
        f"/runs/{run_id}/artifacts",
        data={"name": name, "artifact_type": "model", "metadata": "{}"},
        files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
    )
    assert response.status_code == 200
    return response.json()


def test_experiments_list_reports_total_count_header(api):
    for index in range(3):
        api.post("/experiments/", json={"name": f"exp-{index}"})

    page = api.get("/experiments/", params={"limit": 2})
    assert page.status_code == 200
    assert page.headers["x-total-count"] == "3"
    assert len(page.json()) == 2

    rest = api.get("/experiments/", params={"limit": 2, "offset": 2})
    assert rest.headers["x-total-count"] == "3"
    assert len(rest.json()) == 1


def test_experiments_list_validates_pagination_inputs(api):
    assert api.get("/experiments/", params={"limit": 0}).status_code == 400
    assert api.get("/experiments/", params={"offset": -1}).status_code == 400


def test_runs_list_reports_total_count_header(api):
    experiment = api.post("/experiments/", json={"name": "paginated"}).json()
    for index in range(3):
        api.post(
            f"/experiments/{experiment['id']}/runs/",
            json={"name": f"run-{index}"},
        )

    page = api.get(f"/experiments/{experiment['id']}/runs/", params={"limit": 2})
    assert page.headers["x-total-count"] == "3"
    assert len(page.json()) == 2

    rest = api.get(f"/experiments/{experiment['id']}/runs/", params={"limit": 2, "offset": 2})
    assert rest.headers["x-total-count"] == "3"
    assert len(rest.json()) == 1

    assert (
        api.get(f"/experiments/{experiment['id']}/runs/", params={"limit": 0}).status_code
        == 400
    )


def test_search_reports_total_count_header(api):
    first = api.post("/experiments/", json={"name": "vision"}).json()
    second = api.post("/experiments/", json={"name": "text"}).json()
    api.post(f"/experiments/{first['id']}/runs/", json={"name": "resnet-a"})
    api.post(f"/experiments/{second['id']}/runs/", json={"name": "bert-b"})
    api.post(f"/experiments/{second['id']}/runs/", json={"name": "bert-c"})

    page = api.get("/runs/search", params={"limit": 1})
    assert page.status_code == 200
    assert page.headers["x-total-count"] == "3"
    assert len(page.json()["runs"]) == 1

    filtered = api.get("/runs/search", params={"name_contains": "bert"})
    assert filtered.headers["x-total-count"] == "2"
    assert filtered.json()["total"] == 2


def test_experiment_artifacts_reports_total_count_header(api):
    experiment = api.post("/experiments/", json={"name": "inventory"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()
    for index in range(3):
        upload_artifact(api, run["id"], f"file-{index}.bin", bytes([index]))

    page = api.get(f"/experiments/{experiment['id']}/artifacts", params={"limit": 2})
    assert page.headers["x-total-count"] == "3"
    assert len(page.json()["artifacts"]) == 2


def test_run_artifacts_list_reports_total_count_and_paginates(api):
    experiment = api.post("/experiments/", json={"name": "run-artifacts"}).json()
    run = api.post(f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}).json()
    for index in range(3):
        upload_artifact(api, run["id"], f"file-{index}.bin", bytes([index]))

    page = api.get(f"/runs/{run['id']}/artifacts/", params={"limit": 2})
    assert page.headers["x-total-count"] == "3"
    assert len(page.json()) == 2

    rest = api.get(f"/runs/{run['id']}/artifacts/", params={"limit": 2, "offset": 2})
    assert rest.headers["x-total-count"] == "3"
    assert len(rest.json()) == 1

    assert api.get(f"/runs/{run['id']}/artifacts/", params={"limit": 0}).status_code == 400


def test_client_exposes_total_on_experiment_pages(tracker):
    for index in range(3):
        tracker.create_experiment(f"client-exp-{index}")

    page = tracker.list_experiments(limit=2)
    assert isinstance(page, PaginatedList)
    assert isinstance(page, list)
    assert page.total == 3
    assert len(page) == 2
    assert page[0]["name"].startswith("client-exp-")
    assert {exp["name"] for exp in page} <= {"client-exp-0", "client-exp-1", "client-exp-2"}


def test_client_exposes_total_on_run_pages(tracker):
    experiment = tracker.create_experiment("client-runs")
    for index in range(3):
        tracker.create_run(experiment["id"], f"run-{index}")

    page = tracker.list_runs(experiment["id"], limit=2)
    assert isinstance(page, PaginatedList)
    assert page.total == 3
    assert len(page) == 2
    assert len(tracker.list_runs(experiment["id"], limit=2, offset=2)) == 1


def test_client_exposes_total_on_artifact_pages(tracker, tmp_path):
    experiment = tracker.create_experiment("client-artifacts")
    run = tracker.create_run(experiment["id"], "trainer")
    source = tmp_path / "blob.bin"
    source.write_bytes(b"payload")
    for index in range(3):
        tracker.upload_artifact(run["id"], f"file-{index}.bin", "model", str(source))

    page = tracker.list_artifacts(run["id"], limit=2)
    assert isinstance(page, PaginatedList)
    assert page.total == 3
    assert len(page) == 2
    assert page[0]["artifact_id"]


def test_client_search_still_returns_total_in_body(tracker):
    experiment = tracker.create_experiment("client-search")
    tracker.create_run(experiment["id"], "alpha")
    tracker.create_run(experiment["id"], "beta")

    result = tracker.search_runs(limit=1)
    assert isinstance(result, dict)
    assert result["total"] == 2
    assert len(result["runs"]) == 1
