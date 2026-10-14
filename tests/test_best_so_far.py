"""Tests for best-so-far / simple-regret curves."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models import Experiment, Run, RunStatus, best_so_far_curve
from src.storage import LocalStorageBackend


def test_best_so_far_improves_and_regret_falls():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runs = []
    for i, value in enumerate([0.5, 0.7, 0.6, 0.9]):
        run = Run(experiment_id="e", name=f"r{i}", params={"lr": 0.1})
        run.created_at = base + timedelta(days=i)
        run.log_metric("accuracy", value, step=1)
        runs.append(run)
    curve = best_so_far_curve(runs, "accuracy", maximize=True)
    assert curve["run_count"] == 4
    assert curve["final_best"] == pytest.approx(0.9)
    assert curve["final_regret"] == 0.0
    assert [p["best_so_far"] for p in curve["points"]] == pytest.approx([0.5, 0.7, 0.7, 0.9])
    assert [p["simple_regret"] for p in curve["points"]] == pytest.approx([0.4, 0.2, 0.2, 0.0])
    assert [p["improved"] for p in curve["points"]] == [True, True, False, True]


def test_minimize_mode():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runs = []
    for i, value in enumerate([2.0, 1.5, 1.8]):
        run = Run(experiment_id="e", name=f"r{i}", params={})
        run.created_at = base + timedelta(hours=i)
        run.log_metric("loss", value, step=1)
        runs.append(run)
    curve = best_so_far_curve(runs, "loss", maximize=False)
    assert curve["final_best"] == pytest.approx(1.5)
    assert [p["best_so_far"] for p in curve["points"]] == pytest.approx([2.0, 1.5, 1.5])
    assert [p["simple_regret"] for p in curve["points"]] == pytest.approx([0.5, 0.0, 0.0])


def test_empty_and_validation():
    assert best_so_far_curve([], "accuracy")["run_count"] == 0
    with pytest.raises(ValueError, match="metric_name"):
        best_so_far_curve([], "")


def test_storage_wrapper(tmp_path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="sweep")
    storage.save_experiment(experiment.to_dict())
    base = datetime(2026, 2, 1, tzinfo=timezone.utc)
    for i, value in enumerate([0.4, 0.8, 0.7]):
        run = Run(experiment_id=experiment.id, name=f"r{i}", params={"i": i})
        run.created_at = base + timedelta(minutes=i)
        run.log_metric("accuracy", value, step=1)
        run.finish(RunStatus.COMPLETED)
        storage.save_run(run.to_dict())
    curve = storage.experiment_best_so_far(experiment.id, "accuracy", maximize=True)
    assert curve["final_best"] == pytest.approx(0.8)
    assert curve["points"][-1]["best_so_far"] == pytest.approx(0.8)
    assert [p["improved"] for p in curve["points"]] == [True, True, False]
