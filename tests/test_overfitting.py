"""Tests for overfitting-onset / generalization-gap detection."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run, detect_overfitting

TRAIN = [2.0, 1.5, 1.1, 0.8, 0.6, 0.45, 0.35, 0.28]
VAL = [2.1, 1.6, 1.3, 1.15, 1.2, 1.3, 1.45, 1.6]


class TestDetector:
    def test_readme_example(self):
        r = detect_overfitting(TRAIN, VAL, patience=3)
        assert (r["best_val_step"], r["overfit_onset_step"], r["status"]) == (3, 4, "overfitting")
        assert r["gap_at_best"] == pytest.approx(1.15 - 0.8)
        assert r["final_gap"] == pytest.approx(1.6 - 0.28)
        assert r["gap_growth"] == pytest.approx(r["final_gap"] - r["gap_at_best"])
        assert r["points_after_best"] == 4
        assert r["n_points"] == 8

    def test_healthy_curve_is_ok(self):
        r = detect_overfitting([1.0, 0.8, 0.6, 0.5], [1.1, 0.9, 0.7, 0.6])
        assert r["status"] == "ok" and r["overfit_onset_step"] is None
        assert r["best_val_step"] == 3

    def test_patience_and_min_delta(self):
        assert detect_overfitting(TRAIN, VAL, patience=5)["status"] == "ok"
        # with a large min_delta the late rise isn't "worse" enough
        assert detect_overfitting(TRAIN, VAL, patience=3, min_delta=0.5)["overfit_onset_step"] is None

    def test_requires_training_to_keep_improving(self):
        # validation degrades but training is also degrading -> not overfitting
        train = [2.0, 1.5, 1.2, 1.3, 1.4, 1.5]
        val = [2.1, 1.6, 1.3, 1.4, 1.5, 1.6]
        assert detect_overfitting(train, val, patience=3)["status"] == "ok"

    def test_maximize_scores(self):
        train_acc = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
        val_acc = [0.55, 0.65, 0.72, 0.7, 0.69, 0.67, 0.66]
        r = detect_overfitting(train_acc, val_acc, maximize=True, patience=2)
        assert r["status"] == "overfitting"
        assert r["best_val_step"] == 2 and r["overfit_onset_step"] == 3
        assert r["gap_at_best"] == pytest.approx(0.8 - 0.72)

    def test_stalled_and_alignment_on_sparse_validation(self):
        r = detect_overfitting([1.0, 0.9, 0.8], [1.0, 1.1, 1.2], patience=5)
        assert r["status"] == "stalled"
        # train every step, val every 10 steps; val at step 0 has a train value at step 0
        train = [1.0 - 0.01 * i for i in range(31)]
        r = detect_overfitting(train, [0.9, 0.8, 0.85, 0.9], val_steps=[0, 10, 20, 30], patience=2)
        assert [p["train"] for p in r["points"]] == pytest.approx([1.0, 0.9, 0.8, 0.7])
        assert r["overfit_onset_step"] == 20

    def test_drops_unpaired_and_non_finite(self):
        r = detect_overfitting([1.0, 0.5], [9.0, float("nan"), 0.7], train_steps=[5, 6], val_steps=[0, 5, 6])
        assert [p["step"] for p in r["points"]] == [6]

    def test_validation(self):
        with pytest.raises(ValueError, match="patience"):
            detect_overfitting(TRAIN, VAL, patience=0)
        with pytest.raises(ValueError, match="min_delta"):
            detect_overfitting(TRAIN, VAL, min_delta=-1)
        with pytest.raises(ValueError, match="non-empty"):
            detect_overfitting(TRAIN, [])
        with pytest.raises(ValueError, match="same length"):
            detect_overfitting(TRAIN, VAL, val_steps=[0, 1])
        with pytest.raises(ValueError, match="preceding"):
            detect_overfitting([1.0], [1.0], train_steps=[10], val_steps=[0])


def _run(temp_storage):
    experiment = Experiment(name="overfit")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name="r")
    for step, (t, v) in enumerate(zip(TRAIN, VAL)):
        run.log_metric("train_loss", t, step=step)
        run.log_metric("val_loss", v, step=step)
    temp_storage.save_run(run.to_dict())
    return run


class TestStorageAndApi:
    def test_storage(self, temp_storage):
        run = _run(temp_storage)
        r = temp_storage.run_overfitting_report(run.id, "train_loss", "val_loss")
        assert r["status"] == "overfitting" and r["overfit_onset_step"] == 4
        assert r["run_id"] == run.id and r["val_metric"] == "val_loss"
        with pytest.raises(KeyError):
            temp_storage.run_overfitting_report("nope", "train_loss", "val_loss")
        with pytest.raises(ValueError):
            temp_storage.run_overfitting_report(run.id, "train_loss", "val_acc")

    def test_api(self, api, temp_storage):
        run = _run(temp_storage)
        params = {"train_metric": "train_loss", "val_metric": "val_loss", "patience": 3}
        resp = api.get(f"/runs/{run.id}/overfitting", params=params)
        assert resp.status_code == 200
        assert resp.json()["best_val_step"] == 3
        assert api.get("/runs/missing/overfitting", params=params).status_code == 404
        bad = dict(params, val_metric="nope")
        assert api.get(f"/runs/{run.id}/overfitting", params=bad).status_code == 400
        assert api.get(f"/runs/{run.id}/overfitting", params=dict(params, patience=0)).status_code == 422
