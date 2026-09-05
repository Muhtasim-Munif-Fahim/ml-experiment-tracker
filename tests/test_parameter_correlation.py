"""Tests for parameter importance via Pearson correlation with a target metric."""

import pytest

from src.models import Experiment, Run, RunStatus, pearson_correlation
from src.storage import LocalStorageBackend


def _seed(storage, exp_name="sensitivity"):
    experiment = Experiment(name=exp_name)
    storage.save_experiment(experiment.to_dict())
    specs = [
        ("a", {"lr": 0.1, "seed": 1, "optimizer": "adam", "beta": 0.9}, 0.9),
        ("b", {"lr": 0.2, "seed": 3, "optimizer": "adam"}, 0.7),
        ("c", {"lr": 0.3, "seed": 2, "optimizer": "adam"}, 0.5),
    ]
    for name, params, loss in specs:
        run = Run(experiment_id=experiment.id, name=name, params=params)
        run.log_metric("loss", loss, step=1)
        run.finish(RunStatus.COMPLETED)
        storage.save_run(run.to_dict())
    return experiment


def test_pearson_perfect_positive_and_negative():
    assert pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson_correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)


def test_pearson_returns_none_when_undefined():
    assert pearson_correlation([1.0], [1.0]) is None
    assert pearson_correlation([1.0, 2.0], [1.0, 2.0, 3.0]) is None
    assert pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson_correlation([1.0, 2.0], [5.0, 5.0]) is None


def test_parameter_correlation_ranks_by_absolute_value(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed(storage)
    result = storage.experiment_parameter_correlation(experiment.id, "loss")
    names = [entry["param_name"] for entry in result]
    assert names == ["lr", "seed"]
    assert result[0]["correlation"] == pytest.approx(-1.0)
    assert result[0]["run_count"] == 3


def test_parameter_correlation_skips_non_numeric_and_single_observations(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed(storage)
    result = storage.experiment_parameter_correlation(experiment.id, "loss")
    names = [entry["param_name"] for entry in result]
    assert "optimizer" not in names
    assert "beta" not in names


def test_parameter_correlation_raises_for_unknown_experiment(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    with pytest.raises(KeyError, match="experiment not found"):
        storage.experiment_parameter_correlation("missing", "loss")


def test_parameter_correlation_omits_metric_without_latest_values(tmp_path) -> None:
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = _seed(storage)
    assert storage.experiment_parameter_correlation(experiment.id, "missing_metric") == []


def test_api_parameter_correlation(api, temp_storage) -> None:
    experiment = _seed(temp_storage)
    response = api.get(
        f"/experiments/{experiment.id}/parameter-correlation", params={"metric": "loss"}
    )
    assert response.status_code == 200
    body = response.json()
    assert [entry["param_name"] for entry in body] == ["lr", "seed"]
    assert body[0]["correlation"] == pytest.approx(-1.0)


def test_api_parameter_correlation_404_for_missing_experiment(api) -> None:
    response = api.get(
        "/experiments/missing/parameter-correlation", params={"metric": "loss"}
    )
    assert response.status_code == 404


def test_api_parameter_correlation_requires_metric_query(api, temp_storage) -> None:
    experiment = _seed(temp_storage)
    response = api.get(f"/experiments/{experiment.id}/parameter-correlation")
    assert response.status_code == 422


def test_client_parameter_correlation(tracker) -> None:
    experiment = tracker.create_experiment("client-sensitivity")
    specs = [("a", {"lr": 0.1}, 0.9), ("b", {"lr": 0.2}, 0.7), ("c", {"lr": 0.3}, 0.5)]
    for name, params, loss in specs:
        run = tracker.create_run(experiment["id"], name, params=params)
        tracker.log_metric(run["id"], "loss", loss, step=1)

    result = tracker.parameter_correlation(experiment["id"], "loss")
    assert result == [
        {"param_name": "lr", "correlation": pytest.approx(-1.0), "run_count": 3}
    ]
