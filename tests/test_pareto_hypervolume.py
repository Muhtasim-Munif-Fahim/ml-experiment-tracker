"""Tests for the hypervolume indicator of an experiment's Pareto front."""

from __future__ import annotations

import itertools

import pytest

from src.client import ExperimentTrackerClient
from src.models import Experiment, Run, pareto_hypervolume

FLEET = {
    "accurate": (0.95, 100.0),
    "balanced": (0.90, 40.0),
    "fast": (0.80, 10.0),
    "slow": (0.85, 60.0),
}
OBJECTIVES = [
    {"metric": "accuracy", "maximize": True, "reference": 0.7},
    {"metric": "latency", "maximize": False, "reference": 120.0},
]


def _seed(storage, fleet=FLEET):
    experiment = Experiment(name="hv")
    storage.save_experiment(experiment.to_dict())
    for name, (accuracy, latency) in fleet.items():
        run = Run(experiment_id=experiment.id, name=name)
        run.log_metric("accuracy", accuracy, step=1)
        run.log_metric("latency", latency, step=1)
        storage.save_run(run.to_dict())
    return experiment


def _brute_force_2d(points, ref):
    """Union area of boxes [p, ref] (minimization) on a fine coordinate grid."""
    xs = sorted({p[0] for p in points} | {ref[0]})
    ys = sorted({p[1] for p in points} | {ref[1]})
    area = 0.0
    for (x0, x1), (y0, y1) in itertools.product(zip(xs, xs[1:]), zip(ys, ys[1:])):
        if any(p[0] <= x0 and p[1] <= y0 for p in points):
            area += (x1 - x0) * (y1 - y0)
    return area


class TestHypervolumeFunction:
    def test_two_objectives_closed_form(self):
        # maximize both, reference (0, 0): staircase of three points
        res = pareto_hypervolume([[3, 1], [2, 2], [1, 3]], [0, 0])
        assert res["hypervolume"] == pytest.approx(6.0)
        assert res["contributions"] == pytest.approx([1.0, 1.0, 1.0])

    def test_matches_brute_force_with_dominated_points(self):
        pts = [[1.0, 5.0], [2.0, 3.0], [4.0, 1.0], [3.0, 4.0], [2.5, 3.5]]
        res = pareto_hypervolume(pts, [6.0, 6.0], maximize=[False, False])
        assert res["hypervolume"] == pytest.approx(_brute_force_2d(pts, [6.0, 6.0]))
        # [3, 4] and [2.5, 3.5] are dominated by [2, 3]: no exclusive volume
        assert res["contributions"][3] == 0.0 and res["contributions"][4] == 0.0

    def test_three_objectives_inclusion_exclusion(self):
        # two boxes from the origin-reference in 3-D (maximize): volumes 2*2*1 and 1*1*3
        res = pareto_hypervolume([[2, 2, 1], [1, 1, 3]], [0, 0, 0])
        # union = 4 + 3 - overlap (1*1*1)
        assert res["hypervolume"] == pytest.approx(6.0)
        assert res["contributions"] == pytest.approx([3.0, 2.0])

    def test_points_not_beating_reference_add_nothing(self):
        res = pareto_hypervolume([[1.0, 1.0], [5.0, -1.0]], [0.0, 0.0])
        assert res["hypervolume"] == pytest.approx(1.0)
        assert res["contributions"] == [pytest.approx(1.0), 0.0]
        assert pareto_hypervolume([], [0.0])["hypervolume"] == 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match="one value per objective"):
            pareto_hypervolume([[1.0]], [0.0, 0.0])
        with pytest.raises(ValueError, match="one flag"):
            pareto_hypervolume([[1.0, 1.0]], [0.0, 0.0], maximize=[True])
        with pytest.raises(ValueError, match="finite"):
            pareto_hypervolume([[float("nan"), 1.0]], [0.0, 0.0])


class TestExperimentHypervolume:
    def test_explicit_reference(self, temp_storage):
        experiment = _seed(temp_storage)
        res = temp_storage.experiment_pareto_hypervolume(experiment.id, OBJECTIVES)
        # front: accurate (0.95,100), balanced (0.90,40), fast (0.80,10); ref (0.7, 120)
        expected = 0.10 * 110 + 0.10 * 80 + 0.05 * 20  # slabs along accuracy = 20
        assert res["hypervolume"] == pytest.approx(expected)
        assert {row["name"] for row in res["front"]} == {"accurate", "balanced", "fast"}
        assert res["n_runs"] == 4
        assert res["reference"] == {"accuracy": 0.7, "latency": 120.0}
        shares = [row["contribution_share"] for row in res["front"]]
        assert shares == sorted(shares, reverse=True)
        fast = next(row for row in res["front"] if row["name"] == "fast")
        assert fast["contribution"] == pytest.approx(0.10 * 30)

    def test_default_reference_pads_nadir(self, temp_storage):
        experiment = _seed(temp_storage)
        objectives = [{"metric": "accuracy"}, {"metric": "latency", "maximize": False}]
        res = temp_storage.experiment_pareto_hypervolume(experiment.id, objectives, reference_margin=0.1)
        assert res["reference"]["accuracy"] == pytest.approx(0.80 - 0.1 * 0.15)
        assert res["reference"]["latency"] == pytest.approx(100.0 + 0.1 * 90.0)
        assert all(row["contribution"] > 0 for row in res["front"])

    def test_more_runs_never_shrink_hypervolume(self, temp_storage):
        experiment = _seed(temp_storage)
        before = temp_storage.experiment_pareto_hypervolume(experiment.id, OBJECTIVES)["hypervolume"]
        run = Run(experiment_id=experiment.id, name="new-tradeoff")
        run.log_metric("accuracy", 0.93, step=1)
        run.log_metric("latency", 25.0, step=1)
        temp_storage.save_run(run.to_dict())
        after = temp_storage.experiment_pareto_hypervolume(experiment.id, OBJECTIVES)
        assert after["hypervolume"] > before
        assert after["front"][0]["name"] == "new-tradeoff"

    def test_errors(self, temp_storage):
        with pytest.raises(KeyError):
            temp_storage.experiment_pareto_hypervolume("missing", OBJECTIVES)
        experiment = _seed(temp_storage)
        with pytest.raises(ValueError, match="no run"):
            temp_storage.experiment_pareto_hypervolume(experiment.id, [{"metric": "nope"}])
        with pytest.raises(ValueError, match="reference_margin"):
            temp_storage.experiment_pareto_hypervolume(experiment.id, OBJECTIVES, reference_margin=-1)


def test_api_and_client(api, temp_storage, live_server):
    experiment = _seed(temp_storage)
    response = api.post(f"/experiments/{experiment.id}/pareto/hypervolume", json=OBJECTIVES)
    assert response.status_code == 200
    assert response.json()["hypervolume"] == pytest.approx(20.0)
    assert api.post("/experiments/missing/pareto/hypervolume", json=OBJECTIVES).status_code == 404
    assert api.post(f"/experiments/{experiment.id}/pareto/hypervolume", json=[]).status_code == 400
    client = ExperimentTrackerClient(base_url=live_server)
    res = client.experiment_pareto_hypervolume(experiment.id, OBJECTIVES)
    assert len(res["front"]) == 3
