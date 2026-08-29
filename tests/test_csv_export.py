"""Tests for CSV export of search results and leaderboards."""

import csv
import io


def read_csv_rows(body: str):
    return list(csv.reader(io.StringIO(body)))


def seed_api_corpus(api):
    vision = api.post("/experiments/", json={"name": "vision"}).json()
    text = api.post("/experiments/", json={"name": "text, nlp"}).json()
    resnet_a = api.post(
        f"/experiments/{vision['id']}/runs/", json={"name": "resnet-a"}
    ).json()
    resnet_b = api.post(
        f"/experiments/{vision['id']}/runs/", json={"name": "resnet-b"}
    ).json()
    api.post(
        f"/runs/{resnet_a['id']}/metrics",
        json={"name": "accuracy", "value": 0.5, "step": 1},
    )
    api.post(
        f"/runs/{resnet_b['id']}/metrics",
        json={"name": "accuracy", "value": 0.9, "step": 3},
    )
    bert = api.post(
        f"/experiments/{text['id']}/runs/",
        json={"name": "bert, tuned", "tags": {"size": "base"}},
    ).json()
    api.post(
        f"/runs/{bert['id']}/metrics",
        json={"name": "accuracy", "value": 0.8, "step": 2},
    )
    return vision, text


def test_search_csv_has_stable_columns_and_quoting(api):
    seed_api_corpus(api)

    response = api.get("/runs/search.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        'attachment; filename="run-search.csv"'
        in response.headers["content-disposition"]
    )

    rows = read_csv_rows(response.text)
    assert rows[0] == [
        "experiment_id",
        "run_id",
        "name",
        "status",
        "created_at",
        "updated_at",
    ]
    assert len(rows) == 4
    by_name = {row[2]: row for row in rows[1:]}
    assert by_name["bert, tuned"][3] == "running"
    assert by_name["bert, tuned"][0] != by_name["resnet-a"][0]


def test_search_csv_applies_filters_and_pagination(api):
    seed_api_corpus(api)

    filtered = api.get("/runs/search.csv", params={"name_contains": "resnet"})
    rows = read_csv_rows(filtered.text)
    assert [row[2] for row in rows[1:]] == ["resnet-b", "resnet-a"]

    paged = api.get(
        "/runs/search.csv", params={"name_contains": "resnet", "limit": 1}
    )
    assert len(read_csv_rows(paged.text)) == 2

    invalid = api.get("/runs/search.csv", params={"min_metric": 0.5})
    assert invalid.status_code == 400


def test_leaderboard_csv_orders_and_ranks_rows(api):
    vision, _ = seed_api_corpus(api)

    response = api.get(
        f"/experiments/{vision['id']}/leaderboard.csv", params={"metric": "accuracy"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")

    rows = read_csv_rows(response.text)
    assert rows[0] == ["rank", "run_id", "name", "value", "step"]
    assert len(rows) == 3
    assert rows[1][0] == "1" and float(rows[1][3]) == 0.9 and rows[1][4] == "3"
    assert rows[2][0] == "2" and float(rows[2][3]) == 0.5 and rows[2][4] == "1"


def test_leaderboard_csv_requires_metric_query(api):
    vision, _ = seed_api_corpus(api)
    response = api.get(f"/experiments/{vision['id']}/leaderboard.csv")
    assert response.status_code == 422


def test_client_downloads_and_saves_csv(tracker, tmp_path):
    experiment = tracker.create_experiment("vision")
    run_a = tracker.create_run(experiment["id"], "resnet-a")
    run_b = tracker.create_run(experiment["id"], "resnet-b")
    tracker.log_metric(run_a["id"], "accuracy", 0.5, step=1)
    tracker.log_metric(run_b["id"], "accuracy", 0.9, step=2)

    search_csv = tracker.search_runs_csv(name_contains="resnet")
    rows = read_csv_rows(search_csv)
    assert rows[0][:3] == ["experiment_id", "run_id", "name"]
    assert sorted(row[2] for row in rows[1:]) == ["resnet-a", "resnet-b"]

    destination = tmp_path / "exports" / "leaderboard.csv"
    leaderboard_csv = tracker.run_leaderboard_csv(
        experiment["id"], "accuracy", destination=str(destination)
    )
    ranked = read_csv_rows(leaderboard_csv)
    assert ranked[1][1] == run_b["id"]
    assert ranked[2][1] == run_a["id"]
    assert destination.read_bytes().decode("utf-8") == leaderboard_csv


def seed_metric_history_run(api):
    experiment = api.post("/experiments/", json={"name": "history"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "trainer"}
    ).json()
    for step, (name, value) in enumerate(
        [("loss", 0.5), ("accuracy", 0.6), ("loss", 0.3), ("accuracy", 0.8)],
        start=1,
    ):
        api.post(
            f"/runs/{run['id']}/metrics",
            json={"name": name, "value": value, "step": step},
        )
    return run


def test_run_metrics_csv_flattens_full_history(api):
    run = seed_metric_history_run(api)

    response = api.get(f"/runs/{run['id']}/metrics.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        'attachment; filename="run-metrics.csv"'
        in response.headers["content-disposition"]
    )

    rows = read_csv_rows(response.text)
    assert rows[0] == ["name", "step", "value", "timestamp"]
    assert len(rows) == 5
    assert [row[1] for row in rows[1:]] == ["1", "2", "3", "4"]
    assert [row[2] for row in rows[1:]] == ["0.5", "0.6", "0.3", "0.8"]
    assert [row[0] for row in rows[1:]] == ["loss", "accuracy", "loss", "accuracy"]


def test_run_metrics_csv_is_header_only_without_metrics(api):
    experiment = api.post("/experiments/", json={"name": "empty-history"}).json()
    run = api.post(
        f"/experiments/{experiment['id']}/runs/", json={"name": "bare"}
    ).json()

    rows = read_csv_rows(api.get(f"/runs/{run['id']}/metrics.csv").text)
    assert rows == [["name", "step", "value", "timestamp"]]


def test_run_metrics_csv_missing_run_404(api):
    assert api.get("/runs/missing/metrics.csv").status_code == 404


def test_client_exports_full_metric_history_csv(tracker, tmp_path):
    experiment = tracker.create_experiment("history-client")
    run = tracker.create_run(experiment["id"], "trainer")
    tracker.log_metric(run["id"], "loss", 0.5, step=1)
    tracker.log_metric(run["id"], "loss", 0.3, step=2)

    destination = tmp_path / "exports" / "run-metrics.csv"
    text = tracker.metric_history_csv(run["id"], str(destination))
    rows = read_csv_rows(text)
    assert rows[0] == ["name", "step", "value", "timestamp"]
    assert [row[1] for row in rows[1:]] == ["1", "2"]
    assert [row[2] for row in rows[1:]] == ["0.5", "0.3"]
    assert destination.read_bytes().decode("utf-8") == text
