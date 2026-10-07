"""ML Experiment Tracker - Core Models"""

from __future__ import annotations

import json
import math
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from dataclasses import dataclass, field



def flatten_params(
    params: Any,
    *,
    separator: str = ".",
    prefix: str = "",
    max_depth: int = 32,
) -> Dict[str, Any]:
    """Flatten nested dict / list hyperparameters into dotted keys.

    Nested mappings become ``parent.child`` keys. Sequences (except strings /
    bytes) become ``parent.0``, ``parent.1``, .... Scalars are kept as-is.
    Empty mappings collapse to an empty dict under the current prefix only
    when a prefix is set (so ``{"a": {}}`` yields ``{"a": {}}``). Cycles or
    depths beyond ``max_depth`` raise ``ValueError``.
    """
    if not isinstance(separator, str) or not separator:
        raise ValueError("separator must be a non-empty string")
    if max_depth < 1:
        raise ValueError("max_depth must be a positive integer")

    def _walk(value: Any, path: str, depth: int, seen: set) -> Dict[str, Any]:
        if depth > max_depth:
            raise ValueError("params nesting exceeds max_depth")
        if isinstance(value, Mapping):
            obj_id = id(value)
            if obj_id in seen:
                raise ValueError("params contain a cyclic reference")
            if not value:
                return {path: {}} if path else {}
            seen = set(seen)
            seen.add(obj_id)
            out: Dict[str, Any] = {}
            for key, child in value.items():
                key_s = str(key)
                child_path = f"{path}{separator}{key_s}" if path else key_s
                out.update(_walk(child, child_path, depth + 1, seen))
            return out
        if isinstance(value, (list, tuple)):
            obj_id = id(value)
            if obj_id in seen:
                raise ValueError("params contain a cyclic reference")
            if not value:
                return {path: []} if path else {}
            seen = set(seen)
            seen.add(obj_id)
            out = {}
            for index, child in enumerate(value):
                child_path = f"{path}{separator}{index}" if path else str(index)
                out.update(_walk(child, child_path, depth + 1, seen))
            return out
        if not path:
            raise ValueError("top-level params must be a mapping")
        return {path: value}

    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    return _walk(params, prefix, 1, set())


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


RUN_STATUS_TRANSITIONS = {
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.ABORTED}),
    RunStatus.FAILED: frozenset({RunStatus.RUNNING}),
    RunStatus.ABORTED: frozenset({RunStatus.RUNNING}),
    RunStatus.COMPLETED: frozenset(),
}


def parse_run_status(value: Any) -> RunStatus:
    """Coerce a raw stored or request value into a RunStatus."""
    try:
        return RunStatus(value)
    except ValueError as exc:
        raise ValueError(f"unknown run status: {value!r}") from exc


def validate_status_transition(current: RunStatus, target: RunStatus) -> None:
    """Reject moves outside the declared map; restating a status is a no-op."""
    if target is current:
        return
    if target not in RUN_STATUS_TRANSITIONS[current]:
        raise ValueError(
            f"run status cannot move from '{current.value}' to '{target.value}'"
        )


class ArtifactType(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    METRIC = "metric"
    PLOT = "plot"
    CONFIG = "config"


@dataclass
class Param:
    name: str
    value: Any
    param_type: str = "any"


@dataclass
class Metric:
    name: str
    value: float
    step: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
        }


def lttb_downsample(series: List[dict], points: int) -> List[dict]:
    """Reduce a metric series to ``points`` samples with largest-triangle buckets.

    Keeps the first and last points, then fills the remaining buckets with
    the samples forming the largest triangle against the previously chosen
    point and the average of the next bucket, so spikes and trend shape
    survive aggressive reduction. Series already at or below ``points`` are
    returned as copies.
    """
    if points < 2:
        raise ValueError("points must be at least 2")
    total = len(series)
    if total <= points:
        return [dict(sample) for sample in series]

    sampled = [dict(series[0])]
    every = (total - 2) / (points - 2)
    anchor_x = 0.0
    anchor_y = float(series[0]["value"])
    for index in range(1, points - 1):
        bucket_start = int((index - 1) * every) + 1
        bucket_end = int(index * every) + 1
        next_start = int(index * every) + 1
        next_end = min(int((index + 1) * every) + 1, total)
        next_bucket = series[next_start:next_end]
        avg_x = (next_start + (next_end - 1)) / 2.0
        avg_y = sum(float(sample["value"]) for sample in next_bucket) / len(next_bucket)

        best_index = bucket_start
        best_area = -1.0
        for position in range(bucket_start, bucket_end):
            x = float(position)
            y = float(series[position]["value"])
            area = abs(
                (anchor_x - avg_x) * (y - anchor_y)
                - (anchor_x - x) * (avg_y - anchor_y)
            )
            if area > best_area:
                best_area = area
                best_index = position
        sampled.append(dict(series[best_index]))
        anchor_x = float(best_index)
        anchor_y = float(series[best_index]["value"])

    sampled.append(dict(series[-1]))
    return sampled


def smooth_metric_series(
    series: List[dict], window: int = 5, method: str = "ema"
) -> List[dict]:
    """Smooth a metric series while keeping every original point.

    Unlike :func:`lttb_downsample`, which reduces the number of samples, this
    helper preserves the full resolution of the series and only blends the
    ``value`` of each point to suppress step noise in training curves.

    ``method`` selects the blending rule:

    * ``"ema"`` -- exponential moving average with span ``window``
      (``alpha = 2 / (window + 1)``). The first point seeds the filter and is
      returned unchanged; each subsequent point is ``alpha * value +
      (1 - alpha) * previous``.
    * ``"sma"`` -- simple moving average over a trailing window of up to
      ``window`` points. Positions near the start, where a full window is not
      yet available, use the expanding mean of the points seen so far.

    The function never mutates its inputs; every returned sample is a shallow
    copy carrying the original ``step`` and ``timestamp`` alongside the
    smoothed ``value``.
    """
    if method not in ("ema", "sma"):
        raise ValueError(f"method must be 'ema' or 'sma', got {method!r}")
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer")
    if not series:
        return []

    values = [float(sample.get("value", 0)) for sample in series]
    smoothed: List[float] = []
    if method == "ema":
        alpha = 2.0 / (window + 1.0)
        ema = values[0]
        smoothed.append(ema)
        for value in values[1:]:
            ema = alpha * value + (1.0 - alpha) * ema
            smoothed.append(ema)
    else:  # sma
        for index, value in enumerate(values):
            start = max(0, index - window + 1)
            segment = values[start : index + 1]
            smoothed.append(sum(segment) / len(segment))

    result: List[dict] = []
    for index, sample in enumerate(series):
        entry = dict(sample)
        entry["value"] = smoothed[index]
        result.append(entry)
    return result


def metric_variability(series: List[dict]) -> Optional[dict]:
    """Summarize metric volatility without altering the recorded series.

    Returns sample standard deviation, coefficient of variation, and the mean
    absolute change between consecutive observations.  It is intended to flag
    unstable training curves before selecting a best checkpoint.
    """
    if not series:
        return None
    values = [float(sample["value"]) for sample in series]
    if len(values) == 1:
        return {"count": 1, "mean": values[0], "stddev": 0.0, "coefficient_of_variation": 0.0, "mean_absolute_change": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stddev = variance ** 0.5
    changes = [abs(right - left) for left, right in zip(values, values[1:])]
    return {"count": len(values), "mean": mean, "stddev": stddev, "coefficient_of_variation": stddev / abs(mean) if mean else None, "mean_absolute_change": sum(changes) / len(changes)}


def interpolate_metric_series(
    series: List[dict], max_gap: Optional[int] = None
) -> List[dict]:
    """Fill integer step gaps in a metric series with linear interpolation.

    Unlike :func:`smooth_metric_series` (which blends each point's value) and
    :func:`lttb_downsample` (which reduces the number of samples), this helper
    preserves every recorded point and inserts new points at the missing
    integer steps between two observed samples, estimating their ``value`` by
    linearly interpolating between the surrounding known points. Each inserted
    point inherits the ``name`` and timestamp of the preceding known sample.

    ``max_gap`` caps how many consecutive missing steps may be filled: a gap
    larger than ``max_gap`` is left untouched (the surrounding known points are
    still emitted). When ``None`` every gap is filled. Returns copies of the
    samples, never the originals.
    """
    if max_gap is not None:
        if not isinstance(max_gap, int) or isinstance(max_gap, bool) or max_gap < 1:
            raise ValueError("max_gap must be a positive integer")
    if not series:
        return []

    indexed: List[tuple] = []
    for point in series:
        step = point.get("step")
        if step is None:
            raise ValueError("metric points must carry a step to interpolate")
        indexed.append((int(step), float(point.get("value", 0.0)), dict(point)))
    indexed.sort(key=lambda item: item[0])

    seen: set[int] = set()
    unique: List[tuple] = []
    for entry in indexed:
        if entry[0] in seen:
            continue
        seen.add(entry[0])
        unique.append(entry)

    if not unique:
        return []

    result: List[dict] = []
    for index, (step, value, point) in enumerate(unique):
        result.append(point)
        if index == len(unique) - 1:
            break
        next_step, next_value, _ = unique[index + 1]
        span = next_step - step
        gap = span - 1
        if gap < 1:
            continue
        if max_gap is not None and gap > max_gap:
            continue
        for offset in range(1, gap + 1):
            interpolated = dict(point)
            interpolated["step"] = step + offset
            fraction = offset / span
            interpolated["value"] = value + (next_value - value) * fraction
            result.append(interpolated)

    result.sort(key=lambda entry: entry.get("step"))
    return result


def pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson correlation coefficient between two equal-length numeric series.

    Returns ``None`` when the coefficient is undefined -- fewer than two
    samples, mismatched lengths, or zero variance in either input -- so callers
    can omit undefined entries rather than emitting a misleading value.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    if denom == 0.0:
        return None
    return cov / denom


def standardize_series(
    values: List[float], mean: float, std: float
) -> List[float]:
    """Return the z-scores of ``values`` given a ``mean`` and ``std``.

    A zero standard deviation collapses every point to ``0.0`` (the series is
    already constant), so callers can standardize without guarding against
    division-by-zero noise themselves.
    """
    if std == 0.0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""

    max_iterations = 200
    epsilon = 3.0e-16
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_p_value(t_statistic: float, degrees_of_freedom: int) -> float:
    """Two-sided p-value for a Student-t statistic (pure-Python, no scipy)."""

    if degrees_of_freedom <= 0:
        return 1.0
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic * t_statistic)
    return _betai(degrees_of_freedom / 2.0, 0.5, x)


def metric_trend(series: List[dict], *, alpha: float = 0.05) -> Optional[dict]:
    """Ordinary least-squares trend of a metric over its step.

    Regresses ``value`` on ``step`` and returns the slope, intercept, R-squared,
    standard error, t-statistic, two-sided p-value and a significance flag for
    the slope. The p-value is derived from the Student-t distribution via an
    in-house regularized incomplete beta function, so no external statistics
    stack is required. Points without a numeric step are skipped; fewer than two
    usable points (or a degenerate, step-less abscissa) yields ``None``.

    ``alpha`` is the significance level applied to the slope's p-value; the
    result is flagged significant when ``p_value < alpha`` and must lie in the
    interval ``(0, 1]``. ``direction`` is ``"increasing"`` when the slope is
    positive, ``"decreasing"`` when it is negative and ``"flat"`` at exactly zero.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in the interval (0, 1]")
    sampled: List[tuple] = []
    seen: set = set()
    for point in series:
        step = point.get("step")
        if step is None:
            continue
        try:
            x = float(step)
            y = float(point.get("value", 0.0))
        except (TypeError, ValueError):
            continue
        if x in seen:
            continue
        seen.add(x)
        sampled.append((x, y))
    n = len(sampled)
    if n < 2:
        return None

    mean_x = sum(sample[0] for sample in sampled) / n
    mean_y = sum(sample[1] for sample in sampled) / n
    sxx = sum((sample[0] - mean_x) ** 2 for sample in sampled)
    sxy = sum((sample[0] - mean_x) * (sample[1] - mean_y) for sample in sampled)
    if sxx == 0.0:
        return None
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in sampled)
    ss_tot = sum((y - mean_y) ** 2 for _, y in sampled)
    if ss_tot == 0.0:
        r_squared = 1.0 if ss_res == 0.0 else 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    if n > 2 and ss_res > 0.0:
        residual_variance = ss_res / (n - 2)
        std_err = math.sqrt(residual_variance / sxx)
        t_statistic = slope / std_err
        p_value = _student_t_two_sided_p_value(t_statistic, n - 2)
        significant = p_value < alpha
    else:
        # Perfect fit (zero residual): the slope is exactly determined rather
        # than estimated from noise. A non-zero slope is a real trend
        # (p -> 0); a zero slope is a flat series with no trend.
        std_err = 0.0
        t_statistic = None
        if slope != 0.0:
            p_value = 0.0
            significant = True
        else:
            p_value = 1.0
            significant = False

    if slope > 0.0:
        direction = "increasing"
    elif slope < 0.0:
        direction = "decreasing"
    else:
        direction = "flat"

    return {
        "n_points": n,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "std_err": std_err,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "alpha": alpha,
        "direction": direction,
        "significant": significant,
    }


@dataclass
class Artifact:
    name: str
    artifact_type: ArtifactType
    path: str
    size_bytes: int
    checksum_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.artifact_type.value,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AlertRule:
    """Threshold condition checked against metrics as they are logged."""

    metric_name: str
    comparator: str
    threshold: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.comparator not in ("gt", "lt"):
            raise ValueError("comparator must be 'gt' or 'lt'")

    def matches(self, value: float) -> bool:
        if self.comparator == "gt":
            return value > self.threshold
        return value < self.threshold

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "comparator": self.comparator,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AlertRule":
        return cls(
            id=data["id"],
            metric_name=data["metric_name"],
            comparator=data["comparator"],
            threshold=float(data["threshold"]),
        )


@dataclass
class Run:
    experiment_id: str
    name: str
    parent_run_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: RunStatus = RunStatus.RUNNING
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: List[Metric] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    status_history: List[Dict[str, Any]] = field(default_factory=list)

    def log_param(self, name: str, value: Any, *, flatten: bool = True) -> None:
        """Record a hyperparameter, optionally flattening nested mappings.

        When ``flatten`` is true (default) and ``value`` is a mapping or a
        non-string sequence, the value is expanded with
        :func:`flatten_params` under the ``name`` prefix (for example
        ``model`` + ``{"lr": 0.1}`` stores ``model.lr``). Scalars are stored
        under ``name`` unchanged.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("param name must be a non-empty string")
        if flatten and (isinstance(value, Mapping) or isinstance(value, (list, tuple))):
            flat = flatten_params({name: value})
            self.params.update(flat)
        else:
            self.params[name] = value

    def log_params(self, params: Dict[str, Any], *, flatten: bool = True) -> None:
        """Record many hyperparameters at once.

        When ``flatten`` is true, nested dicts and lists are expanded into
        dotted keys via :func:`flatten_params` before merging into
        ``self.params``.
        """
        if not isinstance(params, Mapping) or not params:
            raise ValueError("params must be a non-empty mapping")
        if flatten:
            self.params.update(flatten_params(params))
        else:
            self.params.update(dict(params))

    def clone(self, name: Optional[str] = None) -> "Run":
        """Create a new running child run with copied configuration metadata."""

        return Run(
            experiment_id=self.experiment_id,
            name=name or f"{self.name} (clone)",
            parent_run_id=self.id,
            params=deepcopy(self.params),
            tags=deepcopy(self.tags),
        )

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.metrics.append(Metric(name=name, value=value, step=step))

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Record a group of metrics at one training step."""

        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("metrics must be a non-empty mapping")
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Summarize the recorded history for one metric."""

        matching = [metric for metric in self.metrics if metric.name == name]
        if not matching:
            return None
        values = [metric.value for metric in matching]
        best = max(matching, key=lambda metric: metric.value)
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last": matching[-1].value,
            "last_step": matching[-1].step,
            "best_step": best.step,
        }

    def best_metric(self, name: str, *, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """Return the best recorded point for a metric in its original context."""

        matching = [metric for metric in self.metrics if metric.name == name]
        if not matching:
            return None
        return (max if maximize else min)(matching, key=lambda metric: metric.value).to_dict()

    def metric_window_summary(self, name: str, window: int) -> Optional[Dict[str, Any]]:
        """Summarize the most recent ``window`` points for one metric."""

        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("window must be a positive integer")
        matching = [metric for metric in self.metrics if metric.name == name][-window:]
        if not matching:
            return None
        values = [metric.value for metric in matching]
        return {
            "name": name,
            "requested_window": window,
            "count": len(matching),
            "first": values[0],
            "last": values[-1],
            "delta": values[-1] - values[0],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "first_step": matching[0].step,
            "last_step": matching[-1].step,
        }

    def metric_history(
        self,
        name: str,
        *,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Export one metric series, optionally restricted to an inclusive step range."""

        if start_step is not None and end_step is not None and start_step > end_step:
            raise ValueError("start_step must not exceed end_step")
        return [
            metric.to_dict()
            for metric in self.metrics
            if metric.name == name
            and (start_step is None or metric.step is not None and metric.step >= start_step)
            and (end_step is None or metric.step is not None and metric.step <= end_step)
        ]

    def log_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

    def artifact_inventory(self) -> List[Dict[str, Any]]:
        """Summarize retained artifacts by type for storage and review planning."""

        inventory: Dict[str, Dict[str, Any]] = {}
        for artifact in self.artifacts:
            kind = artifact.artifact_type.value
            entry = inventory.setdefault(
                kind, {"type": kind, "count": 0, "size_bytes": 0, "names": []}
            )
            entry["count"] += 1
            entry["size_bytes"] += artifact.size_bytes
            entry["names"].append(artifact.name)
        return [inventory[kind] for kind in sorted(inventory)]

    def finish(self, status: RunStatus = RunStatus.COMPLETED, error: Optional[str] = None) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc)
        if error:
            self.error = error

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "parent_run_id": self.parent_run_id,
            "status": self.status.value,
            "params": self.params,
            "metrics": [m.to_dict() for m in self.metrics],
            "artifacts": [{"name": a.name, "type": a.artifact_type.value, "path": a.path, "size": a.size_bytes} for a in self.artifacts],
            "tags": self.tags,
            "alerts": [dict(alert) for alert in self.alerts],
            "notes": [dict(note) for note in self.notes],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "status_history": [dict(entry) for entry in self.status_history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Run":
        run = cls(
            id=data["id"],
            experiment_id=data["experiment_id"],
            name=data["name"],
            parent_run_id=data.get("parent_run_id"),
            status=RunStatus(data["status"]),
            params=data.get("params", {}),
            tags=data.get("tags", {}),
        )
        run.metrics = [Metric(name=m["name"], value=m["value"], step=m.get("step"), timestamp=datetime.fromisoformat(m["timestamp"])) for m in data.get("metrics", [])]
        run.artifacts = [Artifact(name=a["name"], artifact_type=ArtifactType(a["type"]), path=a["path"], size_bytes=a["size"], metadata=a.get("metadata", {}), created_at=datetime.fromisoformat(a["created_at"])) for a in data.get("artifacts", [])]
        run.created_at = datetime.fromisoformat(data["created_at"])
        run.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("finished_at"):
            run.finished_at = datetime.fromisoformat(data["finished_at"])
        run.error = data.get("error")
        run.alerts = [dict(alert) for alert in data.get("alerts", [])]
        run.notes = [dict(note) for note in data.get("notes", [])]
        run.status_history = [dict(entry) for entry in data.get("status_history", [])]
        return run


@dataclass
class Experiment:
    name: str
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    archived: bool = False
    tags: List[str] = field(default_factory=list)
    runs: List[Run] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_run(self, run: Run) -> None:
        self.runs.append(run)
        self.updated_at = datetime.now(timezone.utc)

    def get_best_run(self, metric: str, maximize: bool = True) -> Optional[Run]:
        best = None
        best_value = None
        for r in self.runs:
            if r.status != RunStatus.COMPLETED:
                continue
            values = [m.value for m in r.metrics if m.name == metric]
            if not values:
                continue
            value = max(values) if maximize else min(values)
            if best is None or (maximize and value > best_value) or (not maximize and value < best_value):
                best = r
                best_value = value
        return best

    def metric_table(self, metric: str, maximize: bool = True) -> List[Dict[str, Any]]:
        """Return completed runs ranked by their best value for ``metric``."""

        rows = []
        for run in self.runs:
            if run.status != RunStatus.COMPLETED:
                continue
            summary = run.metric_summary(metric)
            if summary is None:
                continue
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "value": summary["max"] if maximize else summary["min"],
                    "last": summary["last"],
                    "count": summary["count"],
                    "params": dict(run.params),
                }
            )
        return sorted(rows, key=lambda row: row["value"], reverse=maximize)

    def metric_catalog(self) -> List[Dict[str, Any]]:
        """List metrics recorded in the experiment with run and point coverage."""

        catalog: Dict[str, Dict[str, Any]] = {}
        for run in self.runs:
            seen_in_run = set()
            for metric in run.metrics:
                entry = catalog.setdefault(
                    metric.name,
                    {"name": metric.name, "run_count": 0, "point_count": 0},
                )
                entry["point_count"] += 1
                seen_in_run.add(metric.name)
            for name in seen_in_run:
                catalog[name]["run_count"] += 1
        return [catalog[name] for name in sorted(catalog)]

    def parameter_catalog(self) -> List[Dict[str, Any]]:
        """List configured parameter values and the run coverage for each name."""

        catalog: Dict[str, Dict[str, Any]] = {}
        for run in self.runs:
            for name, value in run.params.items():
                entry = catalog.setdefault(name, {"name": name, "run_count": 0, "values": []})
                entry["run_count"] += 1
                if value not in entry["values"]:
                    entry["values"].append(deepcopy(value))
        return [catalog[name] for name in sorted(catalog)]

    def duration_summary(self) -> Dict[str, Any]:
        """Summarize elapsed durations for runs that have finished."""

        durations = [
            (run.finished_at - run.created_at).total_seconds()
            for run in self.runs
            if run.finished_at is not None and run.finished_at >= run.created_at
        ]
        invalid_count = sum(
            1
            for run in self.runs
            if run.finished_at is not None and run.finished_at < run.created_at
        )
        if not durations:
            return {
                "finished_run_count": 0,
                "invalid_run_count": invalid_count,
                "min_seconds": None,
                "max_seconds": None,
                "mean_seconds": None,
                "total_seconds": 0.0,
            }
        return {
            "finished_run_count": len(durations),
            "invalid_run_count": invalid_count,
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "mean_seconds": sum(durations) / len(durations),
            "total_seconds": sum(durations),
        }

    def summary(self) -> Dict[str, Any]:
        """Return lifecycle counts and aggregate values for the experiment."""

        status_counts = {status.value: 0 for status in RunStatus}
        metric_values: Dict[str, List[float]] = {}
        for run in self.runs:
            status_counts[run.status.value] += 1
            for metric in run.metrics:
                metric_values.setdefault(metric.name, []).append(metric.value)

        metrics = {
            name: {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "last": values[-1],
            }
            for name, values in sorted(metric_values.items())
        }
        return {
            "experiment_id": self.id,
            "name": self.name,
            "run_count": len(self.runs),
            "status_counts": status_counts,
            "metrics": metrics,
        }

    def run_lineage(self, run_id: str) -> List[Run]:
        """Return a run's ancestry from the root run through ``run_id``."""

        by_id = {run.id: run for run in self.runs}
        if run_id not in by_id:
            raise KeyError(f"run not found in experiment: {run_id}")
        lineage: List[Run] = []
        seen = set()
        current: Optional[Run] = by_id[run_id]
        while current is not None:
            if current.id in seen:
                raise ValueError("run lineage contains a cycle")
            seen.add(current.id)
            lineage.append(current)
            if current.parent_run_id is None:
                break
            if current.parent_run_id not in by_id:
                raise ValueError(
                    f"run lineage references missing parent: {current.parent_run_id}"
                )
            current = by_id[current.parent_run_id]
        return list(reversed(lineage))

    def compare_runs(self, baseline_id: str, candidate_id: str) -> Dict[str, Any]:
        """Compare parameters and latest metric values between two runs."""

        by_id = {run.id: run for run in self.runs}
        missing = [run_id for run_id in (baseline_id, candidate_id) if run_id not in by_id]
        if missing:
            raise KeyError(f"run not found in experiment: {', '.join(missing)}")
        baseline = by_id[baseline_id]
        candidate = by_id[candidate_id]

        def latest_metrics(run: Run) -> Dict[str, float]:
            values: Dict[str, float] = {}
            for metric in run.metrics:
                values[metric.name] = metric.value
            return values

        baseline_metrics = latest_metrics(baseline)
        candidate_metrics = latest_metrics(candidate)
        metric_names = sorted(set(baseline_metrics) | set(candidate_metrics))
        metric_changes = {
            name: {
                "baseline": baseline_metrics.get(name),
                "candidate": candidate_metrics.get(name),
                "delta": (
                    candidate_metrics[name] - baseline_metrics[name]
                    if name in baseline_metrics and name in candidate_metrics
                    else None
                ),
            }
            for name in metric_names
        }
        param_names = sorted(set(baseline.params) | set(candidate.params))
        param_changes = {
            name: {
                "baseline": baseline.params.get(name),
                "candidate": candidate.params.get(name),
            }
            for name in param_names
            if baseline.params.get(name) != candidate.params.get(name)
        }
        return {
            "baseline_run_id": baseline.id,
            "candidate_run_id": candidate.id,
            "metric_changes": metric_changes,
            "parameter_changes": param_changes,
        }

    def compare_many(self, run_ids: Sequence[str]) -> Dict[str, Any]:
        """Compare two or more runs: side-by-side metrics and parameter diffs."""

        by_id = {run.id: run for run in self.runs}
        missing = [run_id for run_id in run_ids if run_id not in by_id]
        if missing:
            raise KeyError(f"run not found in experiment: {', '.join(missing)}")
        return compare_run_records([by_id[run_id] for run_id in run_ids])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "archived": self.archived,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        exp = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            archived=bool(data.get("archived", False)),
            tags=data.get("tags", []),
        )
        exp.created_at = datetime.fromisoformat(data["created_at"])
        exp.updated_at = datetime.fromisoformat(data["updated_at"])
        exp.runs = [Run.from_dict(r) for r in data.get("runs", [])]
        return exp


def _run_view(run: Any) -> Dict[str, Any]:
    """Normalize a ``Run`` or stored run mapping for comparison."""
    if isinstance(run, Run):
        created_at = run.created_at.isoformat() if run.created_at else None
        finished_at = run.finished_at.isoformat() if run.finished_at else None
        status = (
            run.status.value if isinstance(run.status, RunStatus) else str(run.status)
        )
        return {
            "id": run.id,
            "name": run.name,
            "status": status,
            "experiment_id": run.experiment_id,
            "created_at": created_at,
            "finished_at": finished_at,
            "params": dict(run.params),
            "metrics": run.metrics,
        }
    if not isinstance(run, dict):
        raise TypeError("runs must be Run objects or mappings")
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "status": run.get("status"),
        "experiment_id": run.get("experiment_id"),
        "created_at": run.get("created_at"),
        "finished_at": run.get("finished_at"),
        "params": dict(run.get("params") or {}),
        "metrics": run.get("metrics") or [],
    }


def _latest_metric_values(metrics: Sequence[Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for metric in metrics:
        if isinstance(metric, Metric):
            values[metric.name] = metric.value
        elif isinstance(metric, dict) and metric.get("name") is not None:
            raw = metric.get("value")
            if raw is None:
                continue
            values[str(metric["name"])] = float(raw)
    return values


def compare_run_records(runs: Sequence[Any]) -> Dict[str, Any]:
    """Compare two or more runs: side-by-side latest metrics and param diffs.

    Accepts ``Run`` objects or stored run mappings (fake records included).
    The first run is the baseline for per-metric deltas. Parameter values that
    are identical on every run are listed under ``shared_params``; everything
    else appears under ``param_diffs``. Missing metric or parameter values are
    recorded as ``None``.
    """
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
        raise ValueError("at least two runs are required to compare")
    if len(runs) < 2:
        raise ValueError("at least two runs are required to compare")

    views = [_run_view(run) for run in runs]
    run_ids = [view.get("id") for view in views]
    if any(not run_id for run_id in run_ids):
        raise ValueError("every run must have an id")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run ids must be unique")

    latest = [_latest_metric_values(view.get("metrics") or []) for view in views]
    metric_names = sorted({name for values in latest for name in values})
    metrics: List[Dict[str, Any]] = []
    baseline_id = run_ids[0]
    for name in metric_names:
        values_by_run = {
            run_ids[index]: latest[index].get(name) for index in range(len(views))
        }
        baseline_value = values_by_run[baseline_id]
        deltas = {}
        for run_id in run_ids:
            value = values_by_run[run_id]
            if run_id == baseline_id:
                deltas[run_id] = None
            elif value is None or baseline_value is None:
                deltas[run_id] = None
            else:
                deltas[run_id] = value - baseline_value
        metrics.append({"name": name, "values": values_by_run, "deltas": deltas})

    all_param_names = sorted(
        {name for view in views for name in (view.get("params") or {})}
    )
    shared_params: Dict[str, Any] = {}
    param_diffs: List[Dict[str, Any]] = []
    first_params = views[0].get("params") or {}
    for name in all_param_names:
        values_by_run = {
            run_ids[index]: (views[index].get("params") or {}).get(name)
            for index in range(len(views))
        }
        first_has = name in first_params
        first_value = first_params.get(name)
        all_equal = first_has and all(
            name in (view.get("params") or {})
            and (view.get("params") or {}).get(name) == first_value
            for view in views
        )
        if all_equal:
            shared_params[name] = first_value
        else:
            param_diffs.append({"name": name, "values": values_by_run})

    return {
        "run_ids": list(run_ids),
        "baseline_run_id": baseline_id,
        "runs": [
            {
                "id": view.get("id"),
                "name": view.get("name"),
                "status": view.get("status"),
                "experiment_id": view.get("experiment_id"),
                "created_at": view.get("created_at"),
                "finished_at": view.get("finished_at"),
            }
            for view in views
        ],
        "metrics": metrics,
        "shared_params": shared_params,
        "param_diffs": param_diffs,
    }


def _report_cell(value: Any) -> str:
    """Plain-text cell for a comparison table."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


def _run_column_labels(runs: Sequence[Dict[str, Any]]) -> List[str]:
    names = [str(run.get("name") or run.get("id") or "") for run in runs]
    if len(set(names)) == len(names):
        return names
    return [
        f"{name} ({str(run.get('id', ''))[:8]})" for name, run in zip(names, runs)
    ]


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(_md_escape(str(h)) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_md_escape(_report_cell(cell)) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def render_run_comparison_markdown(comparison: Dict[str, Any]) -> str:
    """Render a shareable Markdown report from :func:`compare_run_records`."""
    runs = list(comparison.get("runs") or [])
    labels = _run_column_labels(runs)
    lines = [
        "# Run comparison",
        "",
        f"Compared **{len(runs)}** runs. Baseline: "
        f"`{comparison.get('baseline_run_id', '')}`.",
        "",
        "## Runs",
        "",
        _markdown_table(
            ["Name", "Run ID", "Status", "Experiment", "Created"],
            [
                [
                    run.get("name"),
                    run.get("id"),
                    run.get("status"),
                    run.get("experiment_id"),
                    run.get("created_at"),
                ]
                for run in runs
            ],
        ),
        "",
        "## Metrics",
        "",
    ]

    metrics = list(comparison.get("metrics") or [])
    if not metrics:
        lines.append("_No metrics recorded on these runs._")
    else:
        lines.append("Latest recorded value of each metric, side-by-side.")
        lines.append("")
        metric_rows = []
        for entry in metrics:
            row = [entry.get("name")]
            values = entry.get("values") or {}
            deltas = entry.get("deltas") or {}
            for run in runs:
                row.append(values.get(run.get("id")))
            for run in runs[1:]:
                row.append(deltas.get(run.get("id")))
            metric_rows.append(row)
        delta_headers = [f"Δ {label}" for label in labels[1:]]
        lines.append(
            _markdown_table(["Metric", *labels, *delta_headers], metric_rows)
        )

    lines.extend(["", "## Parameters", ""])
    shared = dict(comparison.get("shared_params") or {})
    diffs = list(comparison.get("param_diffs") or [])
    if not shared and not diffs:
        lines.append("_No parameters recorded on these runs._")
    else:
        if shared:
            lines.append("Shared across every compared run:")
            lines.append("")
            lines.append(
                _markdown_table(
                    ["Parameter", "Value"],
                    [[name, shared[name]] for name in sorted(shared)],
                )
            )
            lines.append("")
        if diffs:
            lines.append("Parameter diffs:")
            lines.append("")
            diff_rows = [
                [entry.get("name"), *[
                    (entry.get("values") or {}).get(run.get("id")) for run in runs
                ]]
                for entry in diffs
            ]
            lines.append(_markdown_table(["Parameter", *labels], diff_rows))
        else:
            lines.append("All compared runs share the same parameter values.")
    lines.append("")
    return "\n".join(lines)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html_escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for cell in row:
            text = _report_cell(cell)
            css = ' class="missing"' if text == "—" else ""
            cells.append(f"<td{css}>{html_escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table>\n<thead><tr>"
        + head
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody>\n</table>"
    )


def render_run_comparison_html(comparison: Dict[str, Any]) -> str:
    """Render a standalone HTML report from :func:`compare_run_records`."""
    runs = list(comparison.get("runs") or [])
    labels = _run_column_labels(runs)
    baseline = html_escape(str(comparison.get("baseline_run_id") or ""))
    run_table = _html_table(
        ["Name", "Run ID", "Status", "Experiment", "Created"],
        [
            [
                run.get("name"),
                run.get("id"),
                run.get("status"),
                run.get("experiment_id"),
                run.get("created_at"),
            ]
            for run in runs
        ],
    )

    metrics = list(comparison.get("metrics") or [])
    if not metrics:
        metrics_block = "<p><em>No metrics recorded on these runs.</em></p>"
    else:
        metric_rows = []
        for entry in metrics:
            row = [entry.get("name")]
            values = entry.get("values") or {}
            deltas = entry.get("deltas") or {}
            for run in runs:
                row.append(values.get(run.get("id")))
            for run in runs[1:]:
                row.append(deltas.get(run.get("id")))
            metric_rows.append(row)
        delta_headers = [f"Δ {label}" for label in labels[1:]]
        metrics_block = (
            "<p>Latest recorded value of each metric, side-by-side.</p>\n"
            + _html_table(["Metric", *labels, *delta_headers], metric_rows)
        )

    shared = dict(comparison.get("shared_params") or {})
    diffs = list(comparison.get("param_diffs") or [])
    param_parts: List[str] = []
    if not shared and not diffs:
        param_parts.append("<p><em>No parameters recorded on these runs.</em></p>")
    else:
        if shared:
            param_parts.append("<p>Shared across every compared run:</p>")
            param_parts.append(
                _html_table(
                    ["Parameter", "Value"],
                    [[name, shared[name]] for name in sorted(shared)],
                )
            )
        if diffs:
            param_parts.append("<p>Parameter diffs:</p>")
            diff_rows = [
                [
                    entry.get("name"),
                    *[
                        (entry.get("values") or {}).get(run.get("id"))
                        for run in runs
                    ],
                ]
                for entry in diffs
            ]
            param_parts.append(_html_table(["Parameter", *labels], diff_rows))
        else:
            param_parts.append(
                "<p>All compared runs share the same parameter values.</p>"
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Run comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.75rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  td.missing {{ color: #888; }}
  code {{ font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Run comparison</h1>
<p>Compared <strong>{len(runs)}</strong> runs. Baseline: <code>{baseline}</code>.</p>
<h2>Runs</h2>
{run_table}
<h2>Metrics</h2>
{metrics_block}
<h2>Parameters</h2>
{chr(10).join(param_parts)}
</body>
</html>
"""


def render_run_comparison(comparison: Dict[str, Any], fmt: str = "markdown") -> str:
    """Render a comparison as ``markdown`` or ``html``."""
    normalized = str(fmt).strip().lower()
    if normalized in ("markdown", "md"):
        return render_run_comparison_markdown(comparison)
    if normalized in ("html", "htm"):
        return render_run_comparison_html(comparison)
    raise ValueError(f"unsupported comparison format: {fmt!r}")


def write_run_comparison_report(
    comparison: Dict[str, Any],
    destination: str,
    fmt: str = "markdown",
) -> str:
    """Write a Markdown or HTML comparison report to ``destination``."""
    text = render_run_comparison(comparison, fmt)
    if not text.endswith("\n"):
        text += "\n"
    target = Path(destination)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target)
