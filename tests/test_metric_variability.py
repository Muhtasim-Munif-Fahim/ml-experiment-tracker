import pytest

from src.models import metric_variability


def test_reports_metric_volatility():
    result = metric_variability([{"value": 1.0}, {"value": 3.0}, {"value": 5.0}])
    assert result["mean"] == 3.0
    assert result["stddev"] == pytest.approx(2.0)
    assert result["mean_absolute_change"] == 2.0


def test_handles_empty_and_single_series():
    assert metric_variability([]) is None
    assert metric_variability([{"value": 2.0}])["stddev"] == 0.0


def test_zero_mean_has_no_coefficient_of_variation():
    assert metric_variability([{"value": -1.0}, {"value": 1.0}])["coefficient_of_variation"] is None
