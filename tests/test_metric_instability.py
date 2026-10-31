"""Tests for training-instability detection (spikes, NaN/inf, divergence)."""

from __future__ import annotations


import pytest

from src.models import Experiment, Run, detect_metric_instability

SMOOTH = [2.0, 1.6, 1.3, 1.1, 1.0, 0.95, 0.9, 0.87, 0.85, 0.84]
SPIKY = [2.0, 1.6, 1.3, 1.1, 1.0, 9.0, 0.95, 0.9, 0.87, 0.85]


class TestDetector:
    def test_smooth_curve_is_stable(self):
        result = detect_metric_instability(SMOOTH)
        assert result["status"] == "stable"
        assert result["n_spikes"] == 0
        assert result["diverged"] is False
        assert result["best_value"] == 0.84

    def test_loss_spike_is_flagged_but_recovered(self):
        result = detect_metric_instability(SPIKY, steps=[s * 10 for s in range(10)])
        assert result["status"] == "unstable"
        assert result["n_spikes"] == 1
        spike = result["spikes"][0]
        assert spike["index"] == 5 and spike["step"] == 50 and spike["value"] == 9.0
        assert spike["robust_z"] > 6
        assert result["diverged"] is False

    def test_direction_matters(self):
        # a sudden *drop* is good for a loss but a spike for an accuracy
        acc = [0.5, 0.6, 0.65, 0.7, 0.72, 0.1, 0.74, 0.75]
        assert detect_metric_instability(acc, maximize=True)["n_spikes"] == 1
        assert detect_metric_instability(acc, maximize=False)["n_spikes"] == 0

    def test_nan_and_divergence(self):
        result = detect_metric_instability([2.0, 1.5, 1.2, float("nan"), float("inf")])
        assert result["non_finite_indices"] == [3, 4]
        assert result["first_non_finite_step"] == 3
        assert result["diverged"] is True and result["status"] == "diverged"
        assert result["final_value"] is None

    def test_ending_worse_than_start_is_divergence(self):
        result = detect_metric_instability([1.0, 0.8, 0.6, 0.9, 1.2, 1.5], spike_threshold=100)
        assert result["diverged"] is True
        never_improved = detect_metric_instability([1.0, 1.1, 1.2])
        assert never_improved["diverged"] is False

    def test_plateau_wiggle_is_not_a_spike(self):
        flat = [1.0] * 6 + [1.0005]
        assert detect_metric_instability(flat)["n_spikes"] == 0

    def test_validation(self):
        with pytest.raises(ValueError):
            detect_metric_instability([])
        with pytest.raises(ValueError):
            detect_metric_instability([1.0], window=1)
        with pytest.raises(ValueError):
            detect_metric_instability([1.0], spike_threshold=0)
        with pytest.raises(ValueError):
            detect_metric_instability([1.0, 2.0], steps=[0])


def _experiment(temp_storage, curves):
    experiment = Experiment(name="instability")
    temp_storage.save_experiment(experiment.to_dict())
    runs = {}
    for name, values in curves.items():
        run = Run(experiment_id=experiment.id, name=name)
        for step, value in enumerate(values):
            run.log_metric("loss", value, step=step)
        temp_storage.save_run(run.to_dict())
        runs[name] = run
    return experiment, runs


class TestStorageAndApi:
    def test_run_and_experiment_reports(self, temp_storage):
        exp, runs = _experiment(
            temp_storage,
            {"ok": SMOOTH, "spiky": SPIKY, "boom": [2.0, 1.0, 0.8, 5.0, 50.0, 500.0]},
        )
        result = temp_storage.run_metric_instability(runs["spiky"].id, "loss")
        assert result["n_spikes"] == 1 and result["run_id"] == runs["spiky"].id
        rows = temp_storage.experiment_instability_report(exp.id, "loss")
        assert [r["run_name"] for r in rows] == ["boom", "spiky", "ok"]
        assert [r["status"] for r in rows] == ["diverged", "unstable", "stable"]
        with pytest.raises(KeyError):
            temp_storage.run_metric_instability("nope", "loss")
        with pytest.raises(ValueError):
            temp_storage.run_metric_instability(runs["ok"].id, "accuracy")
        with pytest.raises(KeyError):
            temp_storage.experiment_instability_report("nope", "loss")

    def test_api_endpoints(self, api, temp_storage):
        exp, runs = _experiment(temp_storage, {"spiky": SPIKY, "ok": SMOOTH})
        resp = api.get(f"/runs/{runs['spiky'].id}/instability", params={"metric": "loss"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "unstable"
        resp = api.get(f"/experiments/{exp.id}/instability", params={"metric": "loss"})
        assert resp.status_code == 200
        assert [r["run_name"] for r in resp.json()] == ["spiky", "ok"]
        assert api.get("/runs/missing/instability", params={"metric": "loss"}).status_code == 404
        resp = api.get(f"/runs/{runs['ok'].id}/instability", params={"metric": "acc"})
        assert resp.status_code == 400
