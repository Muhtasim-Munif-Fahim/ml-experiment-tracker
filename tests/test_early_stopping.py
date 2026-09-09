"""Tests for early-stopping / peak detection on a run's metric curve."""

from __future__ import annotations

import pytest

from src.models import Experiment, Run

# Rises to 0.95 at step 4, then decays: the classic overfitting curve.
OVERFIT = [0.50, 0.70, 0.88, 0.93, 0.95, 0.94, 0.92, 0.90]


def _run_with(temp_storage, values, metric="val_accuracy", name="curve"):
    experiment = Experiment(name="early-stopping")
    temp_storage.save_experiment(experiment.to_dict())
    run = Run(experiment_id=experiment.id, name=name)
    for step, value in enumerate(values):
        run.log_metric(metric, value, step=step)
    temp_storage.save_run(run.to_dict())
    return run


class TestPeakDetection:
    def test_finds_the_peak_of_an_overfitting_curve(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        result = temp_storage.run_early_stopping_point(run.id, "val_accuracy")
        assert result["best_value"] == 0.95
        assert result["best_step"] == 4
        assert result["best_index"] == 4

    def test_counts_the_observations_wasted_after_the_peak(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        result = temp_storage.run_early_stopping_point(run.id, "val_accuracy")
        assert result["steps_after_best"] == 3
        assert result["total_points"] == 8

    def test_regression_measures_the_shortfall_at_the_end(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        result = temp_storage.run_early_stopping_point(run.id, "val_accuracy")
        assert result["final_value"] == 0.90
        assert result["regression"] == pytest.approx(0.05)

    def test_a_still_improving_curve_peaks_at_the_end(self, temp_storage):
        run = _run_with(temp_storage, [0.1, 0.3, 0.6, 0.9])
        result = temp_storage.run_early_stopping_point(run.id, "val_accuracy")
        assert result["best_index"] == 3
        assert result["steps_after_best"] == 0
        assert result["regression"] == 0.0
        assert result["would_have_stopped"] is False

    def test_minimizing_tracks_the_trough(self, temp_storage):
        run = _run_with(temp_storage, [1.0, 0.4, 0.2, 0.5, 0.8], metric="val_loss")
        result = temp_storage.run_early_stopping_point(
            run.id, "val_loss", maximize=False
        )
        assert result["best_value"] == 0.2
        assert result["best_step"] == 2
        assert result["regression"] == pytest.approx(0.6)

    def test_regression_is_never_negative_when_minimizing(self, temp_storage):
        run = _run_with(temp_storage, [1.0, 0.5, 0.2], metric="val_loss")
        result = temp_storage.run_early_stopping_point(
            run.id, "val_loss", maximize=False
        )
        assert result["regression"] == 0.0

    def test_a_single_observation_is_its_own_peak(self, temp_storage):
        run = _run_with(temp_storage, [0.42])
        result = temp_storage.run_early_stopping_point(run.id, "val_accuracy")
        assert result["best_value"] == 0.42
        assert result["steps_after_best"] == 0
        assert result["would_have_stopped"] is False

    def test_out_of_order_steps_are_read_as_a_curve(self, temp_storage):
        experiment = Experiment(name="shuffled")
        temp_storage.save_experiment(experiment.to_dict())
        run = Run(experiment_id=experiment.id, name="shuffled")
        for step, value in ((2, 0.9), (0, 0.1), (1, 0.5)):
            run.log_metric("val_accuracy", value, step=step)
        temp_storage.save_run(run.to_dict())
        result = temp_storage.run_early_stopping_point(run.id, "val_accuracy")
        assert result["best_step"] == 2
        assert result["final_step"] == 2

    def test_other_metrics_are_ignored(self, temp_storage):
        experiment = Experiment(name="mixed")
        temp_storage.save_experiment(experiment.to_dict())
        run = Run(experiment_id=experiment.id, name="mixed")
        run.log_metric("val_accuracy", 0.5, step=0)
        run.log_metric("train_loss", 99.0, step=0)
        run.log_metric("val_accuracy", 0.7, step=1)
        temp_storage.save_run(run.to_dict())
        result = temp_storage.run_early_stopping_point(run.id, "val_accuracy")
        assert result["total_points"] == 2
        assert result["best_value"] == 0.7


class TestPatienceAndDelta:
    def test_patience_fires_after_enough_flat_observations(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        result = temp_storage.run_early_stopping_point(
            run.id, "val_accuracy", patience=2
        )
        assert result["would_have_stopped"] is True
        assert result["stopped_at_step"] == 6

    def test_generous_patience_never_fires(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        result = temp_storage.run_early_stopping_point(
            run.id, "val_accuracy", patience=50
        )
        assert result["would_have_stopped"] is False
        assert result["stopped_at_step"] is None

    def test_the_stop_point_is_the_first_time_patience_runs_out(self, temp_storage):
        run = _run_with(temp_storage, [0.5, 0.4, 0.3, 0.2, 0.1])
        result = temp_storage.run_early_stopping_point(
            run.id, "val_accuracy", patience=1
        )
        assert result["stopped_at_step"] == 1

    def test_a_renewed_improvement_resets_patience(self, temp_storage):
        run = _run_with(temp_storage, [0.5, 0.4, 0.4, 0.9, 0.8])
        result = temp_storage.run_early_stopping_point(
            run.id, "val_accuracy", patience=3
        )
        assert result["best_step"] == 3
        assert result["would_have_stopped"] is False

    def test_min_delta_ignores_noise_around_a_plateau(self, temp_storage):
        run = _run_with(temp_storage, [0.90, 0.9001, 0.9002, 0.9003])
        strict = temp_storage.run_early_stopping_point(
            run.id, "val_accuracy", patience=2
        )
        tolerant = temp_storage.run_early_stopping_point(
            run.id, "val_accuracy", patience=2, min_delta=0.01
        )
        # Without min_delta each tick counts as progress and nothing fires.
        assert strict["would_have_stopped"] is False
        assert tolerant["would_have_stopped"] is True
        assert tolerant["best_index"] == 0


class TestValidation:
    def test_a_missing_run_raises(self, temp_storage):
        with pytest.raises(KeyError, match="run not found"):
            temp_storage.run_early_stopping_point("nope", "val_accuracy")

    def test_a_metric_with_no_values_raises(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        with pytest.raises(ValueError, match="no recorded values for metric"):
            temp_storage.run_early_stopping_point(run.id, "absent")

    def test_patience_must_be_positive(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        with pytest.raises(ValueError, match="patience must be a positive integer"):
            temp_storage.run_early_stopping_point(run.id, "val_accuracy", patience=0)

    def test_min_delta_must_be_non_negative(self, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        with pytest.raises(ValueError, match="min_delta must be non-negative"):
            temp_storage.run_early_stopping_point(
                run.id, "val_accuracy", min_delta=-0.1
            )


class TestEarlyStoppingApi:
    def test_endpoint_reports_the_peak(self, api, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        response = api.get(
            f"/runs/{run.id}/early-stopping", params={"metric": "val_accuracy"}
        )
        assert response.status_code == 200
        assert response.json()["best_step"] == 4

    def test_endpoint_honours_maximize_false(self, api, temp_storage):
        run = _run_with(temp_storage, [1.0, 0.2, 0.9], metric="val_loss")
        response = api.get(
            f"/runs/{run.id}/early-stopping",
            params={"metric": "val_loss", "maximize": False},
        )
        assert response.json()["best_value"] == 0.2

    def test_unknown_run_is_404(self, api):
        response = api.get(
            "/runs/missing/early-stopping", params={"metric": "val_accuracy"}
        )
        assert response.status_code == 404

    def test_unknown_metric_is_400(self, api, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        response = api.get(
            f"/runs/{run.id}/early-stopping", params={"metric": "absent"}
        )
        assert response.status_code == 400

    def test_invalid_patience_is_rejected_by_the_query_model(self, api, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        response = api.get(
            f"/runs/{run.id}/early-stopping",
            params={"metric": "val_accuracy", "patience": 0},
        )
        assert response.status_code == 422


class TestEarlyStoppingClient:
    def test_client_reports_the_peak(self, tracker, temp_storage):
        run = _run_with(temp_storage, OVERFIT)
        result = tracker.run_early_stopping_point(run.id, "val_accuracy")
        assert result["best_step"] == 4
        assert result["steps_after_best"] == 3
