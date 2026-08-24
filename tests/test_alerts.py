"""Tests for metric threshold alert rules."""

import tempfile
from pathlib import Path

import pytest

from src.models import AlertRule, Experiment, Run
from src.storage import LocalStorageBackend


def seed_experiment(storage):
    experiment = Experiment(name="watched")
    storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="trainer")
    storage.save_run(run.to_dict())
    return experiment, run


def test_alert_rule_validates_comparator():
    rule = AlertRule(metric_name="loss", comparator="lt", threshold=0.1)
    assert rule.matches(0.05) and not rule.matches(0.2)
    assert AlertRule(metric_name="loss", comparator="gt", threshold=9).matches(9.5)
    with pytest.raises(ValueError, match="comparator"):
        AlertRule(metric_name="loss", comparator="gte", threshold=1)


def test_alert_rules_round_trip_on_experiment():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment, _ = seed_experiment(storage)
        rule = AlertRule(metric_name="loss", comparator="lt", threshold=0.1)

        saved = storage.save_alert_rule(experiment.id, rule.to_dict())
        assert storage.list_alert_rules(experiment.id) == [saved]
        assert storage.delete_alert_rule(experiment.id, saved["id"]) is True
        assert storage.list_alert_rules(experiment.id) == []
        assert storage.delete_alert_rule(experiment.id, saved["id"]) is False


def test_alert_rules_require_existing_experiment():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        with pytest.raises(KeyError, match="experiment not found"):
            storage.save_alert_rule("missing", {"id": "r"})
        with pytest.raises(KeyError, match="experiment not found"):
            storage.list_alert_rules("missing")


def test_apply_alert_rules_flags_breaching_points_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        experiment, run = seed_experiment(storage)
        storage.save_alert_rule(
            experiment.id,
            AlertRule(metric_name="loss", comparator="lt", threshold=0.1).to_dict(),
        )
        run_data = storage.load_run(run.id)

        triggered = storage.apply_alert_rules(
            run_data, [{"name": "loss", "value": 0.05, "step": 3}]
        )
        assert len(triggered) == 1
        assert triggered[0]["metric_name"] == "loss"
        assert triggered[0]["value"] == 0.05
        assert triggered[0]["step"] == 3
        assert triggered[0]["triggered_at"]

        stored = storage.load_run(run.id)["alerts"]
        assert stored == triggered

        assert storage.apply_alert_rules(
            run_data, [{"name": "loss", "value": 0.5, "step": 4}]
        ) == []
        assert storage.apply_alert_rules(
            run_data, [{"name": "accuracy", "value": 0.01, "step": 4}]
        ) == []
        assert len(storage.load_run(run.id)["alerts"]) == 1


def test_apply_alert_rules_without_rules_is_noop():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(tmpdir)
        _, run = seed_experiment(storage)
        run_data = storage.load_run(run.id)
        assert storage.apply_alert_rules(run_data, [{"name": "loss", "value": 0.0}]) == []
        assert storage.load_run(run.id).get("alerts", []) == []


def test_api_alert_flow_records_alerts_when_metrics_logged(api):
    exp = api.post("/experiments/", json={"name": "watched"}).json()
    run = api.post(f"/experiments/{exp['id']}/runs/", json={"name": "trainer"}).json()

    created = api.post(
        f"/experiments/{exp['id']}/alert-rules",
        json={"metric_name": "loss", "comparator": "lt", "threshold": 0.1},
    )
    assert created.status_code == 200
    rule = created.json()
    assert api.get(f"/experiments/{exp['id']}/alert-rules").json() == [rule]

    single = api.post(
        f"/runs/{run['id']}/metrics", json={"name": "loss", "value": 0.02, "step": 1}
    ).json()
    assert [alert["rule_id"] for alert in single["alerts"]] == [rule["id"]]

    batch = api.post(
        f"/runs/{run['id']}/metrics/batch",
        json={"metrics": {"loss": 0.01}, "step": 2},
    ).json()
    assert [alert["rule_id"] for alert in batch["alerts"]] == [rule["id"]]
    assert api.get(f"/runs/{run['id']}/alerts").json() == (
        single["alerts"] + batch["alerts"]
    )

    assert (
        api.delete(f"/experiments/{exp['id']}/alert-rules/{rule['id']}").json()["message"]
        == "Alert rule deleted"
    )
    assert api.get(f"/experiments/{exp['id']}/alert-rules").json() == []


def test_api_alert_endpoints_reject_unknown_and_invalid_input(api):
    exp = api.post("/experiments/", json={"name": "watched"}).json()
    assert (
        api.post(
            "/experiments/missing/alert-rules",
            json={"metric_name": "loss", "comparator": "lt", "threshold": 0.1},
        ).status_code
        == 404
    )
    invalid = api.post(
        f"/experiments/{exp['id']}/alert-rules",
        json={"metric_name": "loss", "comparator": "gte", "threshold": 0.1},
    )
    assert invalid.status_code == 400
    assert "comparator" in invalid.json()["detail"]

    assert api.get("/experiments/missing/alert-rules").status_code == 404
    assert (
        api.delete(f"/experiments/{exp['id']}/alert-rules/missing").status_code == 404
    )
    assert api.get("/runs/missing/alerts").status_code == 404


def test_client_manages_alert_rules_and_reads_run_alerts(tracker):
    exp = tracker.create_experiment("watched")
    run = tracker.create_run(exp["id"], "trainer")

    rule = tracker.create_alert_rule(exp["id"], "loss", "lt", 0.1)
    assert tracker.list_alert_rules(exp["id"]) == [rule]

    logged = tracker.log_metric(run["id"], "loss", 0.02, step=1)
    assert [alert["rule_id"] for alert in logged["alerts"]] == [rule["id"]]
    assert [alert["value"] for alert in tracker.run_alerts(run["id"])] == [0.02]

    tracker.delete_alert_rule(exp["id"], rule["id"])
    assert tracker.list_alert_rules(exp["id"]) == []
