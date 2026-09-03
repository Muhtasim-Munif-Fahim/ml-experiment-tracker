"""Tests for the experiment_timeline feature."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run, RunStatus
from src.storage import LocalStorageBackend


def _seed(tmp_path):
    from src.models import Metric
    from datetime import datetime
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="timeline-exp")
    storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1", status=RunStatus.COMPLETED)
    run.metrics.append(Metric(name="accuracy", value=0.8, step=1, timestamp=datetime(2026, 1, 1, 12, 0, 0)))
    storage.save_run(run.to_dict())
    storage.add_note(run.id, "looks good")
    storage.set_run_tag(run.id, "phase", "training")
    return storage, experiment, run


def test_timeline_returns_creation_event_first(tmp_path) -> None:
    storage, experiment, _ = _seed(tmp_path)
    events = storage.experiment_timeline(experiment.id)
    assert events
    event_names = [e["event"] for e in events]
    assert "created" in event_names
    assert "metric" in event_names
    assert "note" in event_names
    assert "tag" in event_names
    # Events are sorted by timestamp.
    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps)


def test_timeline_rejects_unknown_experiment(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    with pytest.raises(KeyError, match="experiment not found"):
        storage.experiment_timeline("missing")


def test_timeline_handles_run_with_minimal_data(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="minimal")
    storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    storage.save_run(run.to_dict())
    events = storage.experiment_timeline(experiment.id)
    # Only "created" and "updated" events are present (no metrics/notes/tags).
    assert len(events) == 2
    event_names = {e["event"] for e in events}
    assert event_names == {"created", "updated"}


def test_timeline_contains_metric_details(tmp_path) -> None:
    storage, experiment, _ = _seed(tmp_path)
    events = storage.experiment_timeline(experiment.id)
    metric_events = [e for e in events if e["event"] == "metric"]
    assert any("accuracy" in e["detail"] for e in metric_events)


def test_timeline_contains_note_and_tag_events(tmp_path) -> None:
    storage, experiment, _ = _seed(tmp_path)
    events = storage.experiment_timeline(experiment.id)
    note_events = [e for e in events if e["event"] == "note"]
    tag_events = [e for e in events if e["event"] == "tag"]
    assert note_events
    assert any("looks good" in e["detail"] for e in note_events)
    assert tag_events
    assert any("phase=training" in e["detail"] for e in tag_events)


def test_timeline_each_event_carries_run_id_and_name(tmp_path) -> None:
    storage, experiment, run = _seed(tmp_path)
    events = storage.experiment_timeline(experiment.id)
    for event in events:
        assert event["run_id"] == run.id
        assert event["run_name"] == "r1"
        assert "timestamp" in event
        assert "detail" in event


def test_api_timeline_returns_events_and_count(api, temp_storage) -> None:
    experiment = Experiment(name="api-timeline")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    temp_storage.save_run(run.to_dict())
    response = api.get(f"/experiments/{experiment.id}/timeline.json")
    assert response.status_code == 200
    body = response.json()
    assert "events" in body
    assert "count" in body
    assert body["count"] == 2


def test_api_timeline_404_for_missing_experiment(api) -> None:
    response = api.get("/experiments/missing/timeline.json")
    assert response.status_code == 404


def test_client_experiment_timeline(live_server, temp_storage) -> None:
    from src.client import ExperimentTrackerClient

    experiment = Experiment(name="cli-timeline")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r1")
    temp_storage.save_run(run.to_dict())

    client = ExperimentTrackerClient(base_url=live_server)
    body = client.experiment_timeline(experiment.id)
    assert "events" in body
    assert body["count"] == 2