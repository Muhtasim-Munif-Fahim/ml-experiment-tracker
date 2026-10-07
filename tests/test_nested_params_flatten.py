"""Tests for nested hyperparameter flattening."""

from __future__ import annotations

import pytest

from src.models import Run, flatten_params


def test_flatten_nested_dict_and_list():
    flat = flatten_params(
        {
            "optimizer": {"name": "adam", "lr": 0.01},
            "layers": [64, 32],
            "seed": 7,
        }
    )
    assert flat == {
        "optimizer.name": "adam",
        "optimizer.lr": 0.01,
        "layers.0": 64,
        "layers.1": 32,
        "seed": 7,
    }


def test_flatten_custom_separator():
    flat = flatten_params({"a": {"b": 1}}, separator="/")
    assert flat == {"a/b": 1}


def test_flatten_rejects_non_mapping_and_cycles():
    with pytest.raises(ValueError, match="mapping"):
        flatten_params([1, 2])
    cyclic = {"a": {}}
    cyclic["a"]["self"] = cyclic
    with pytest.raises(ValueError, match="cyclic"):
        flatten_params(cyclic)


def test_flatten_rejects_bad_separator_and_depth():
    with pytest.raises(ValueError):
        flatten_params({"a": 1}, separator="")
    with pytest.raises(ValueError, match="max_depth"):
        flatten_params({"a": {"b": {"c": 1}}}, max_depth=2)


def test_run_log_param_flattens_nested_value():
    run = Run(experiment_id="e", name="r")
    run.log_param("model", {"lr": 0.1, "dropout": 0.2})
    assert run.params["model.lr"] == 0.1
    assert run.params["model.dropout"] == 0.2
    assert "model" not in run.params


def test_run_log_param_can_disable_flatten():
    run = Run(experiment_id="e", name="r")
    nested = {"lr": 0.1}
    run.log_param("model", nested, flatten=False)
    assert run.params["model"] == nested


def test_run_log_params_bulk():
    run = Run(experiment_id="e", name="r")
    run.log_params({"train": {"epochs": 10}, "seed": 1})
    assert run.params == {"train.epochs": 10, "seed": 1}


def test_run_log_params_rejects_empty():
    run = Run(experiment_id="e", name="r")
    with pytest.raises(ValueError):
        run.log_params({})
