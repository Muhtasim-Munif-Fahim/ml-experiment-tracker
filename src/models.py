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


def _t_two_sided_p(t_statistic: float, degrees_of_freedom: float) -> float:
    """Two-sided Student-t p-value for a (possibly fractional) df."""

    if not degrees_of_freedom > 0 or math.isnan(t_statistic):
        return 1.0
    if math.isinf(t_statistic):
        return 0.0
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic * t_statistic)
    return _betai(degrees_of_freedom / 2.0, 0.5, x)


def student_t_critical(confidence: float, degrees_of_freedom: float) -> float:
    """Two-sided critical value ``t`` with ``P(|T| > t) = 1 - confidence``.

    Found by bisection on the incomplete-beta tail, so no scipy is needed.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if not degrees_of_freedom > 0:
        raise ValueError("degrees_of_freedom must be positive")
    target = 1.0 - confidence
    lo, hi = 0.0, 1.0
    while _t_two_sided_p(hi, degrees_of_freedom) > target:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - only for absurd inputs
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_two_sided_p(mid, degrees_of_freedom) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def summarize_values(values: Sequence[float], *, confidence: float = 0.95) -> dict:
    """Mean, sample std, standard error and a Student-t confidence interval.

    With a single value the spread is undefined: ``std``, ``sem`` and the
    interval bounds are ``None``.
    """
    data = [float(v) for v in values]
    if not data:
        raise ValueError("values must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    n = len(data)
    mean = sum(data) / n
    if n < 2:
        return {"n": n, "mean": mean, "std": None, "sem": None, "ci_low": None, "ci_high": None}
    variance = sum((v - mean) ** 2 for v in data) / (n - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(n)
    half = student_t_critical(confidence, n - 1) * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


def welch_t_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alpha: float = 0.05,
) -> Optional[dict]:
    """Welch's unequal-variance t-test of ``mean(a) - mean(b)``.

    Returns the difference, t statistic, Welch–Satterthwaite degrees of
    freedom, two-sided p-value, a ``1 - alpha`` confidence interval for the
    difference and a ``significant`` flag. Returns ``None`` when either
    sample has fewer than two values. Two zero-variance samples give
    ``p_value`` 1.0 when the means match and 0.0 otherwise.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    xs = [float(v) for v in a]
    ys = [float(v) for v in b]
    if len(xs) < 2 or len(ys) < 2:
        return None
    sa = summarize_values(xs)
    sb = summarize_values(ys)
    diff = sa["mean"] - sb["mean"]
    va = sa["std"] ** 2 / sa["n"]
    vb = sb["std"] ** 2 / sb["n"]
    se = math.sqrt(va + vb)
    if se == 0.0:
        p_value = 1.0 if diff == 0.0 else 0.0
        return {
            "mean_a": sa["mean"],
            "mean_b": sb["mean"],
            "difference": diff,
            "t_statistic": 0.0 if diff == 0.0 else math.copysign(math.inf, diff),
            "degrees_of_freedom": float(sa["n"] + sb["n"] - 2),
            "p_value": p_value,
            "ci_low": diff,
            "ci_high": diff,
            "significant": p_value < alpha,
        }
    t_stat = diff / se
    df = (va + vb) ** 2 / (va**2 / (sa["n"] - 1) + vb**2 / (sb["n"] - 1))
    p_value = _t_two_sided_p(t_stat, df)
    half = student_t_critical(1.0 - alpha, df) * se
    return {
        "mean_a": sa["mean"],
        "mean_b": sb["mean"],
        "difference": diff,
        "t_statistic": t_stat,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "ci_low": diff - half,
        "ci_high": diff + half,
        "significant": p_value < alpha,
    }


def _hashable_param(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def group_runs_by_params(
    runs: Sequence[Any],
    metric_name: str,
    *,
    ignore: Sequence[str] = ("seed",),
    group_by: Optional[Sequence[str]] = None,
    maximize: bool = True,
    confidence: float = 0.95,
) -> List[dict]:
    """Aggregate repeated runs (e.g. several seeds) of the same configuration.

    Runs are grouped by their (flattened) params, leaving out every key in
    ``ignore`` (``seed`` by default), or by exactly the keys in ``group_by``
    when it is given. Each run contributes its latest value of
    ``metric_name``; runs that never logged it are skipped. Every group gets
    ``n``, ``mean``, ``std``, ``sem`` and a Student-t ``confidence`` interval.
    Groups are ordered best-first by mean (``maximize`` picks the
    direction). Each group after the first also carries a Welch t-test
    against the best group (``vs_best``), which shows whether the gap is
    larger than the seed-to-seed noise.
    """
    if not metric_name:
        raise ValueError("metric_name is required")
    ignored = set(ignore or ())
    keys = list(group_by) if group_by else None
    buckets: Dict[str, dict] = {}
    for run in runs:
        view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
        values = [
            m.get("value")
            for m in view.get("metrics", []) or []
            if m.get("name") == metric_name and m.get("value") is not None
        ]
        if not values:
            continue
        params = flatten_params(view.get("params") or {})
        if keys is not None:
            config = {k: params.get(k) for k in keys}
        else:
            config = {k: v for k, v in sorted(params.items()) if k not in ignored}
        key = _hashable_param(config)
        bucket = buckets.setdefault(key, {"params": config, "run_ids": [], "values": []})
        bucket["run_ids"].append(view.get("id"))
        bucket["values"].append(float(values[-1]))

    groups: List[dict] = []
    for bucket in buckets.values():
        stats = summarize_values(bucket["values"], confidence=confidence)
        groups.append({"params": bucket["params"], "run_ids": bucket["run_ids"],
                       "values": bucket["values"], **stats})
    groups.sort(key=lambda g: (-g["mean"] if maximize else g["mean"], _hashable_param(g["params"])))
    for rank, group in enumerate(groups, start=1):
        group["rank"] = rank
        if rank == 1:
            group["vs_best"] = None
        else:
            group["vs_best"] = welch_t_test(
                groups[0]["values"], group["values"], alpha=1.0 - confidence
            )
    return groups


def _average_ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rank correlation (Pearson on average ranks; ``None`` if undefined)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson_correlation(_average_ranks(xs), _average_ranks(ys))


def _f_survival(f_statistic: float, df_between: int, df_within: int) -> float:
    """Upper-tail probability of the F distribution (pure Python)."""
    if df_between <= 0 or df_within <= 0:
        return 1.0
    if f_statistic <= 0.0:
        return 1.0
    if math.isinf(f_statistic):
        return 0.0
    x = df_within / (df_within + df_between * f_statistic)
    return _betai(df_within / 2.0, df_between / 2.0, x)


def one_way_anova(groups: Sequence[Sequence[float]]) -> Optional[dict]:
    """One-way ANOVA with eta-squared and omega-squared effect sizes.

    ``eta_squared = SS_between / SS_total`` is the share of metric variance
    explained by the grouping. ``omega_squared`` corrects eta-squared's upward
    bias with small groups and is clipped at 0. Returns ``None`` with fewer
    than two non-empty groups, no within-group degrees of freedom, or zero
    total variance.
    """
    clean = [[float(v) for v in g] for g in groups if len(g) > 0]
    k = len(clean)
    n = sum(len(g) for g in clean)
    if k < 2 or n <= k:
        return None
    grand = sum(sum(g) for g in clean) / n
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in clean)
    ss_within = sum(sum((v - sum(g) / len(g)) ** 2 for v in g) for g in clean)
    ss_total = ss_between + ss_within
    if ss_total <= 0.0:
        return None
    df_between = k - 1
    df_within = n - k
    ms_within = ss_within / df_within
    if ms_within == 0.0:
        f_statistic = math.inf
    else:
        f_statistic = (ss_between / df_between) / ms_within
    omega = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    return {
        "f_statistic": f_statistic,
        "p_value": _f_survival(f_statistic, df_between, df_within),
        "df_between": df_between,
        "df_within": df_within,
        "eta_squared": ss_between / ss_total,
        "omega_squared": max(0.0, omega),
    }


def _quantile_bins(values: List[float], max_levels: int) -> List[int]:
    """Equal-count bin index per value; tied values always share a bin."""
    ranks = _average_ranks(values)
    n = len(values)
    return [min(max_levels - 1, int((r - 1.0) * max_levels / n)) for r in ranks]


def parameter_importance(
    runs: Sequence[Any],
    metric_name: str,
    *,
    ignore: Sequence[str] = ("seed",),
    max_levels: int = 10,
) -> List[dict]:
    """Rank hyperparameters by the share of metric variance they explain.

    Each run contributes its latest ``metric_name`` value and its flattened
    params. For every parameter, runs are grouped by the parameter's value,
    and a one-way ANOVA gives ``eta_squared``, ``omega_squared``, ``F`` and a
    p-value. Unlike Pearson correlation, this also works for categorical
    parameters (optimizer, activation) and for numeric ones whose effect is
    not monotone. Numeric parameters with more than ``max_levels`` distinct
    values are put into ``max_levels`` equal-count bins first, and they also
    report a ``spearman`` rank correlation. Runs that do not set a parameter
    are left out of that parameter's analysis. Entries are sorted by
    ``omega_squared`` and then ``eta_squared``, both descending.
    """
    if not metric_name:
        raise ValueError("metric_name is required")
    if isinstance(max_levels, bool) or not isinstance(max_levels, int) or max_levels < 2:
        raise ValueError("max_levels must be an integer >= 2")
    ignored = set(ignore or ())
    observations: Dict[str, List[tuple]] = {}
    for run in runs:
        view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
        values = [
            m.get("value")
            for m in view.get("metrics", []) or []
            if m.get("name") == metric_name and m.get("value") is not None
        ]
        if not values:
            continue
        latest = float(values[-1])
        for name, value in flatten_params(view.get("params") or {}).items():
            if name in ignored or value is None:
                continue
            observations.setdefault(name, []).append((value, latest))

    results: List[dict] = []
    for name, pairs in observations.items():
        numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v, _ in pairs
        )
        ys = [y for _, y in pairs]
        if numeric:
            xs = [float(v) for v, _ in pairs]
            distinct = sorted(set(xs))
            if len(distinct) > max_levels:
                labels: List[Any] = _quantile_bins(xs, max_levels)
                binned = True
            else:
                labels = xs
                binned = False
        else:
            labels = [_hashable_param(v) for v, _ in pairs]
            binned = False
        buckets: Dict[Any, List[float]] = {}
        raw: Dict[Any, List[Any]] = {}
        for label, (value, y) in zip(labels, pairs):
            buckets.setdefault(label, []).append(y)
            raw.setdefault(label, []).append(value)
        anova = one_way_anova(list(buckets.values()))
        if anova is None:
            continue
        level_rows = []
        for label in sorted(buckets, key=lambda l: (str(type(l)), l)):
            vals = raw[label]
            if binned:
                level = {"min": min(vals), "max": max(vals)}
            else:
                level = vals[0]
            level_rows.append({
                "value": level,
                "n": len(buckets[label]),
                "mean": sum(buckets[label]) / len(buckets[label]),
            })
        results.append({
            "param_name": name,
            "kind": "numeric" if numeric else "categorical",
            "binned": binned,
            "levels": len(buckets),
            "run_count": len(pairs),
            **anova,
            "spearman": spearman_correlation([float(v) for v, _ in pairs], ys) if numeric else None,
            "level_means": level_rows,
        })
    results.sort(key=lambda r: (-r["omega_squared"], -r["eta_squared"], r["param_name"]))
    return results



def _regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularized incomplete gamma ``Q(a, x)`` (Numerical Recipes)."""
    if x <= 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - gln))
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi_square_survival(statistic: float, degrees_of_freedom: float) -> float:
    """Upper-tail probability of the chi-square distribution (pure Python)."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    return _regularized_gamma_q(degrees_of_freedom / 2.0, max(0.0, statistic) / 2.0)


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _range_cdf(q: float, k: int) -> float:
    """CDF of the range of ``k`` iid standard normals (studentized range, df=inf)."""
    if q <= 0.0:
        return 0.0
    # Simpson's rule on z in [-8, 8]
    n, lo, hi = 800, -8.0, 8.0
    h = (hi - lo) / n
    total = 0.0
    for i in range(n + 1):
        z = lo + i * h
        f = _normal_pdf(z) * (_normal_cdf(z + q) - _normal_cdf(z)) ** (k - 1)
        total += f * (1 if i in (0, n) else (4 if i % 2 else 2))
    return min(1.0, k * total * h / 3.0)


def nemenyi_critical_value(k: int, alpha: float = 0.05) -> float:
    """Nemenyi ``q_alpha``: the studentized-range quantile (df=inf) over sqrt(2).

    Matches Table 5 of Demšar (2006), e.g. ``k=3, alpha=0.05 -> 2.343``,
    and is computed numerically, so any number of configurations works.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    target = 1.0 - alpha
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _range_cdf(mid, k) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0 / math.sqrt(2.0)


def friedman_comparison(
    runs: Sequence[Any],
    metric_name: str,
    *,
    block_param: str = "seed",
    ignore: Sequence[str] = ("seed",),
    group_by: Optional[Sequence[str]] = None,
    maximize: bool = True,
    alpha: float = 0.05,
) -> Optional[dict]:
    """Friedman test with a Nemenyi post-hoc over matched blocks (Demšar, 2006).

    Compares several configurations evaluated on the same *blocks* -- seeds,
    datasets or CV folds, given by the ``block_param`` parameter -- using
    ranks instead of raw metric values, so it needs no normality assumption
    and is not dominated by one block with a large spread.

    Configurations are formed as in :func:`group_runs_by_params` (params
    minus ``ignore`` and ``block_param``, or exactly ``group_by``). Each run
    contributes its latest ``metric_name``; repeated runs of one
    configuration in one block are averaged. Only blocks in which *every*
    configuration has a value are used (a complete design). Within each
    block configurations are ranked 1 (best) to ``k``, ties sharing their
    average rank.

    Returns ``None`` with fewer than two configurations or two complete
    blocks. Otherwise a dict with:

    * ``chi_square`` / ``p_value``: tie-corrected Friedman statistic,
      chi-square with ``k - 1`` degrees of freedom;
    * ``iman_davenport_f`` / ``iman_davenport_p_value``: the less
      conservative F form ``(N-1) chi2 / (N(k-1) - chi2)`` (``None`` with
      p-value 0 when every block ranks the configurations identically);
    * ``critical_difference``: Nemenyi ``q_alpha * sqrt(k(k+1) / (6N))``;
      two configurations differ significantly when their average ranks
      differ by more than this;
    * ``configurations`` sorted by average rank, each with ``params``,
      ``average_rank``, ``mean`` and ``worse_than_best`` (rank gap to the
      best exceeds the critical difference);
    * ``pairwise``: every pair's rank difference and significance;
    * ``blocks`` used and ``dropped_blocks`` (incomplete).
    """
    if not metric_name:
        raise ValueError("metric_name is required")
    if not block_param:
        raise ValueError("block_param is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    ignored = set(ignore or ()) | {block_param}
    keys = [k for k in group_by if k != block_param] if group_by else None
    configs: Dict[str, dict] = {}
    cells: Dict[str, Dict[str, List[float]]] = {}
    block_labels: Dict[str, Any] = {}
    for run in runs:
        view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
        values = [
            m.get("value")
            for m in view.get("metrics", []) or []
            if m.get("name") == metric_name and m.get("value") is not None
        ]
        if not values:
            continue
        params = flatten_params(view.get("params") or {})
        if block_param not in params or params[block_param] is None:
            continue
        if keys is not None:
            config = {k: params.get(k) for k in keys}
        else:
            config = {k: v for k, v in sorted(params.items()) if k not in ignored}
        ckey = _hashable_param(config)
        bkey = _hashable_param(params[block_param])
        configs.setdefault(ckey, config)
        block_labels.setdefault(bkey, params[block_param])
        cells.setdefault(ckey, {}).setdefault(bkey, []).append(float(values[-1]))

    k = len(configs)
    if k < 2:
        return None
    config_keys = sorted(configs)
    complete = [b for b in block_labels if all(b in cells[c] for c in config_keys)]
    complete.sort()
    n_blocks = len(complete)
    if n_blocks < 2:
        return None

    rank_sums = [0.0] * k
    value_sums = [0.0] * k
    tie_term = 0.0
    for b in complete:
        row = [sum(cells[c][b]) / len(cells[c][b]) for c in config_keys]
        for j, v in enumerate(row):
            value_sums[j] += v
        ranks = _average_ranks([-v if maximize else v for v in row])
        for j, r in enumerate(ranks):
            rank_sums[j] += r
        counts: Dict[float, int] = {}
        for v in row:
            counts[v] = counts.get(v, 0) + 1
        tie_term += sum(t ** 3 - t for t in counts.values())

    avg = [s / n_blocks for s in rank_sums]
    chi2 = 12.0 * n_blocks / (k * (k + 1)) * (sum(r * r for r in avg) - k * (k + 1) ** 2 / 4.0)
    correction = 1.0 - tie_term / (n_blocks * (k ** 3 - k))
    if correction <= 0.0:
        chi2 = 0.0  # every block is a complete tie
    else:
        chi2 /= correction
    df1 = k - 1
    df2 = (k - 1) * (n_blocks - 1)
    denom = n_blocks * (k - 1) - chi2
    # perfect agreement across blocks makes the F statistic infinite; report
    # it as None (JSON-safe) with p-value 0
    f_stat = None if denom <= 0.0 else (n_blocks - 1) * chi2 / denom
    cd = nemenyi_critical_value(k, alpha) * math.sqrt(k * (k + 1) / (6.0 * n_blocks))

    order = sorted(range(k), key=lambda j: (avg[j], config_keys[j]))
    best = order[0]
    rows = [
        {
            "params": configs[config_keys[j]],
            "average_rank": avg[j],
            "mean": value_sums[j] / n_blocks,
            "rank_gap_to_best": avg[j] - avg[best],
            "worse_than_best": (avg[j] - avg[best]) > cd,
        }
        for j in order
    ]
    pairwise = []
    for a_pos in range(k):
        for b_pos in range(a_pos + 1, k):
            ja, jb = order[a_pos], order[b_pos]
            diff = avg[jb] - avg[ja]
            pairwise.append({
                "a": a_pos,
                "b": b_pos,
                "rank_difference": diff,
                "significant": diff > cd,
            })
    return {
        "metric": metric_name,
        "block_param": block_param,
        "maximize": maximize,
        "k": k,
        "n_blocks": n_blocks,
        "blocks": [block_labels[b] for b in complete],
        "dropped_blocks": [block_labels[b] for b in sorted(block_labels) if b not in complete],
        "chi_square": chi2,
        "df": df1,
        "p_value": chi_square_survival(chi2, df1) if chi2 > 0 else 1.0,
        "iman_davenport_f": f_stat,
        "iman_davenport_df": [df1, df2],
        "iman_davenport_p_value": (
            1.0 if chi2 <= 0 else (0.0 if f_stat is None else _f_survival(f_stat, df1, df2))
        ),
        "alpha": alpha,
        "critical_difference": cd,
        "configurations": rows,
        "pairwise": pairwise,
    }


def _t_cdf(x: float, degrees_of_freedom: float) -> float:
    """Student-t CDF via the incomplete-beta tail (pure Python)."""

    if math.isinf(x):
        return 1.0 if x > 0 else 0.0
    half_tail = 0.5 * _t_two_sided_p(abs(x), degrees_of_freedom)
    return 1.0 - half_tail if x >= 0 else half_tail


def correlated_t_test(
    differences: Sequence[float],
    *,
    test_fraction: float,
    rope: float = 0.0,
    confidence: float = 0.95,
) -> dict:
    """Corrected resampled t-test and Bayesian correlated t-test on CV differences.

    ``differences`` are per-fold (or per-repetition-and-fold) score
    differences ``a - b`` from the *same* resampling splits. Training sets of
    different folds overlap, so the differences are correlated and the naive
    paired t-test is badly overconfident. With ``rho = test_fraction`` (the
    share of the data in each test fold, ``1 / K`` for K-fold CV), the
    Nadeau & Bengio (2003) correction inflates the variance to
    ``(1 / n + rho / (1 - rho)) * s^2``.

    The Bayesian correlated t-test of Corani & Benavoli (2015) uses the same
    scale: the posterior of the mean difference is a Student-t with
    ``n - 1`` degrees of freedom, location ``mean`` and that scale. Given a
    region of practical equivalence ``[-rope, rope]`` it returns
    ``p_a_better`` (``P(mu > rope)``), ``p_rope`` and ``p_b_better``
    (``P(mu < -rope)``).

    Requires at least two differences and ``0 < test_fraction < 1``. When all
    differences are identical the posterior collapses onto their value.
    """
    values = [float(v) for v in differences]
    n = len(values)
    if n < 2:
        raise ValueError("at least two differences are required")
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise ValueError("differences must be finite")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if not rope >= 0.0:
        raise ValueError("rope must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    df = n - 1
    factor = 1.0 / n + test_fraction / (1.0 - test_fraction)
    scale = math.sqrt(factor * var)
    naive_scale = math.sqrt(var / n)
    if scale > 0.0:
        t_stat = mean / scale
        p_value = _t_two_sided_p(t_stat, df)
        naive_p = _t_two_sided_p(mean / naive_scale, df)
        half = student_t_critical(confidence, df) * scale
        upper = 1.0 - _t_cdf((rope - mean) / scale, df)
        lower = _t_cdf((-rope - mean) / scale, df)
    else:
        t_stat = None
        p_value = naive_p = 1.0 if mean == 0.0 else 0.0
        half = 0.0
        upper = 1.0 if mean > rope else 0.0
        lower = 1.0 if mean < -rope else 0.0
    upper, lower = max(0.0, min(1.0, upper)), max(0.0, min(1.0, lower))
    return {
        "n": n,
        "mean_difference": mean,
        "std": math.sqrt(var),
        "test_fraction": test_fraction,
        "correction_factor": factor,
        "scale": scale,
        "df": df,
        "t_statistic": t_stat,
        "p_value": p_value,
        "naive_p_value": naive_p,
        "confidence": confidence,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "rope": rope,
        "p_a_better": upper,
        "p_rope": max(0.0, 1.0 - upper - lower),
        "p_b_better": lower,
    }


def cv_bayesian_comparison(
    runs: Sequence[Any],
    metric_name: str,
    *,
    block_param: str = "fold",
    ignore: Sequence[str] = ("fold", "seed"),
    group_by: Optional[Sequence[str]] = None,
    maximize: bool = True,
    test_fraction: Optional[float] = None,
    rope: float = 0.01,
    confidence: float = 0.95,
    decision_threshold: float = 0.95,
) -> Optional[dict]:
    """Compare configurations to the best one over shared CV folds.

    Configurations are formed as in :func:`friedman_comparison` and matched
    on the ``block_param`` value (fold id; repeated runs in one cell are
    averaged). Only blocks shared by every configuration are used. The
    configuration with the best mean is the reference; for every other one
    the per-fold differences ``best - other`` (sign-flipped when
    ``maximize=False`` so positive always favours the reference) are passed
    to :func:`correlated_t_test`. ``test_fraction`` defaults to
    ``1 / n_blocks`` (plain K-fold CV).

    Each comparison carries a ``decision``: ``"best_better"`` /
    ``"equivalent"`` / ``"other_better"`` when that posterior probability
    reaches ``decision_threshold``, else ``"undecided"``. Returns ``None``
    with fewer than two configurations or two shared blocks.
    """
    if not metric_name:
        raise ValueError("metric_name is required")
    if not block_param:
        raise ValueError("block_param is required")
    if test_fraction is not None and not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if not rope >= 0.0:
        raise ValueError("rope must be non-negative")
    if not 0.5 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must be in (0.5, 1)")
    ignored = set(ignore or ()) | {block_param}
    keys = [k for k in group_by if k != block_param] if group_by else None
    configs: Dict[str, dict] = {}
    cells: Dict[str, Dict[str, List[float]]] = {}
    block_labels: Dict[str, Any] = {}
    for run in runs:
        view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
        values = [
            m.get("value")
            for m in view.get("metrics", []) or []
            if m.get("name") == metric_name and m.get("value") is not None
        ]
        if not values:
            continue
        params = flatten_params(view.get("params") or {})
        if block_param not in params or params[block_param] is None:
            continue
        if keys is not None:
            config = {k: params.get(k) for k in keys}
        else:
            config = {k: v for k, v in sorted(params.items()) if k not in ignored}
        ckey = _hashable_param(config)
        bkey = _hashable_param(params[block_param])
        configs.setdefault(ckey, config)
        block_labels.setdefault(bkey, params[block_param])
        cells.setdefault(ckey, {}).setdefault(bkey, []).append(float(values[-1]))

    if len(configs) < 2:
        return None
    config_keys = sorted(configs)
    complete = sorted(b for b in block_labels if all(b in cells[c] for c in config_keys))
    n_blocks = len(complete)
    if n_blocks < 2:
        return None
    rho = test_fraction if test_fraction is not None else 1.0 / n_blocks
    if rho >= 1.0:  # pragma: no cover - n_blocks >= 2 keeps rho <= 0.5
        raise ValueError("test_fraction must be in (0, 1)")
    table = {
        c: [sum(cells[c][b]) / len(cells[c][b]) for b in complete] for c in config_keys
    }
    sign = 1.0 if maximize else -1.0
    means = {c: sum(v) / n_blocks for c, v in table.items()}
    best = sorted(config_keys, key=lambda c: (-sign * means[c], c))[0]
    comparisons = []
    for c in config_keys:
        if c == best:
            continue
        diffs = [sign * (a - b) for a, b in zip(table[best], table[c])]
        res = correlated_t_test(diffs, test_fraction=rho, rope=rope, confidence=confidence)
        probs = {
            "best_better": res["p_a_better"],
            "equivalent": res["p_rope"],
            "other_better": res["p_b_better"],
        }
        label, prob = max(probs.items(), key=lambda kv: kv[1])
        comparisons.append(
            {
                "params": configs[c],
                "mean": means[c],
                "mean_difference": res["mean_difference"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "t_statistic": res["t_statistic"],
                "p_value": res["p_value"],
                "naive_p_value": res["naive_p_value"],
                "p_best_better": res["p_a_better"],
                "p_equivalent": res["p_rope"],
                "p_other_better": res["p_b_better"],
                "decision": label if prob >= decision_threshold else "undecided",
            }
        )
    comparisons.sort(key=lambda r: (-r["p_best_better"], -r["mean_difference"]))
    return {
        "metric": metric_name,
        "block_param": block_param,
        "maximize": maximize,
        "n_blocks": n_blocks,
        "blocks": [block_labels[b] for b in complete],
        "dropped_blocks": [block_labels[b] for b in sorted(block_labels) if b not in complete],
        "test_fraction": rho,
        "rope": rope,
        "confidence": confidence,
        "decision_threshold": decision_threshold,
        "best": {"params": configs[best], "mean": means[best]},
        "comparisons": comparisons,
    }


def _stepped_history(run: Any, metric_name: str) -> List[tuple]:
    """``(step, value)`` pairs of one metric, sorted by step (unstepped points dropped).

    When a step was logged more than once the last value wins.
    """
    view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    by_step: Dict[int, float] = {}
    for metric in view.get("metrics", []) or []:
        if metric.get("name") != metric_name:
            continue
        step, value = metric.get("step"), metric.get("value")
        if step is None or value is None:
            continue
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            continue
        by_step[int(step)] = value
    return sorted(by_step.items())


def successive_halving_replay(
    runs: Sequence[Any],
    metric_name: str,
    *,
    maximize: bool = True,
    eta: float = 3.0,
    min_step: Optional[int] = None,
    rungs: Optional[Sequence[int]] = None,
) -> Optional[dict]:
    """Replay finished runs under successive halving (SHA / ASHA's synchronous core).

    "Would early stopping have killed my best run, and what would it have
    saved?" -- answered from curves you already logged. Every run with a
    stepped history of ``metric_name`` starts in the bracket. At each rung
    step ``r`` (default ``min_step * eta**k`` below the longest run, with
    ``min_step`` defaulting to the earliest logged step), the surviving runs
    that reached ``r`` are ranked by their latest value at or before ``r``
    and the top ``max(1, floor(n / eta))`` are promoted; the rest are stopped
    at ``r``. Runs whose curve ends before a rung simply finish there.

    Returns ``None`` with fewer than two runs that have stepped values.
    Otherwise a mapping with:

    * ``rungs`` -- per rung: ``step``, ``candidates``, ``promoted`` and
      ``stopped`` run ids, the promotion ``cutoff`` value, and
      ``spearman_with_final`` (rank correlation of the rung value with the
      run's final value; low values mean the rung metric is a poor proxy).
    * ``runs`` -- per run: ``final_value`` / ``final_step`` as logged,
      ``stopped_at`` (rung step or ``None``) and ``steps_used`` under SHA.
    * ``best_run_id`` / ``best_final_value`` -- the best final value
      actually logged; ``best_run_survived`` and ``best_surviving_run_id``;
      ``regret`` -- how much worse the best never-stopped run ended (>= 0).
    * ``budget_full``, ``budget_used`` (sum of steps) and
      ``budget_saved_fraction``.
    """
    if not metric_name:
        raise ValueError("metric_name is required")
    if not eta > 1.0 or math.isinf(eta):
        raise ValueError("eta must be a finite number greater than 1")
    if min_step is not None and min_step < 1:
        raise ValueError("min_step must be a positive integer")

    histories: Dict[str, List[tuple]] = {}
    names: Dict[str, str] = {}
    for run in runs:
        view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
        history = _stepped_history(view, metric_name)
        if history:
            run_id = str(view.get("id"))
            histories[run_id] = history
            names[run_id] = str(view.get("name", ""))
    if len(histories) < 2:
        return None

    max_step = max(h[-1][0] for h in histories.values())
    if rungs is not None:
        rung_steps = sorted({int(r) for r in rungs})
        if not rung_steps or rung_steps[0] < 1:
            raise ValueError("rungs must be positive steps")
    else:
        first = min_step if min_step is not None else max(1, min(h[0][0] for h in histories.values()))
        rung_steps = []
        k = 0
        while True:
            step = int(round(first * eta ** k))
            if step >= max_step:
                break
            if not rung_steps or step > rung_steps[-1]:
                rung_steps.append(step)
            k += 1

    def value_at(history: List[tuple], step: int) -> Optional[float]:
        latest = None
        for s, v in history:
            if s > step:
                break
            latest = v
        return latest

    final = {rid: h[-1] for rid, h in histories.items()}
    alive = set(histories)
    stopped_at: Dict[str, Optional[int]] = {rid: None for rid in histories}
    rung_rows: List[dict] = []
    for step in rung_steps:
        candidates = [
            rid
            for rid in sorted(alive)
            if histories[rid][-1][0] >= step and value_at(histories[rid], step) is not None
        ]
        if not candidates:
            continue
        values = {rid: value_at(histories[rid], step) for rid in candidates}
        order = sorted(
            candidates,
            key=lambda rid: (-values[rid] if maximize else values[rid], rid),
        )
        keep = max(1, int(math.floor(len(order) / eta + 1e-12)))
        promoted, stopped = order[:keep], order[keep:]
        for rid in stopped:
            alive.discard(rid)
            stopped_at[rid] = step
        rho = None
        if len(candidates) >= 3:
            rho = spearman_correlation(
                [values[rid] for rid in candidates], [final[rid][1] for rid in candidates]
            )
        rung_rows.append(
            {
                "step": step,
                "candidates": len(candidates),
                "promoted": promoted,
                "stopped": stopped,
                "cutoff": values[promoted[-1]],
                "spearman_with_final": None if rho is None else round(rho, 6),
            }
        )

    def better(a: float, b: float) -> bool:
        return a > b if maximize else a < b

    run_rows = []
    budget_full = budget_used = 0
    for rid in sorted(histories, key=lambda r: (stopped_at[r] is not None, r)):
        final_step, final_value = final[rid]
        used = final_step if stopped_at[rid] is None else min(final_step, stopped_at[rid])
        budget_full += final_step
        budget_used += used
        run_rows.append(
            {
                "run_id": rid,
                "name": names[rid],
                "final_step": final_step,
                "final_value": final_value,
                "stopped_at": stopped_at[rid],
                "steps_used": used,
            }
        )

    best_id = None
    for rid in sorted(histories):
        if best_id is None or better(final[rid][1], final[best_id][1]):
            best_id = rid
    survivors = [rid for rid in sorted(histories) if stopped_at[rid] is None]
    best_surviving = None
    for rid in survivors:
        if best_surviving is None or better(final[rid][1], final[best_surviving][1]):
            best_surviving = rid
    regret = abs(final[best_id][1] - final[best_surviving][1])
    return {
        "metric": metric_name,
        "maximize": maximize,
        "eta": eta,
        "run_count": len(histories),
        "rungs": rung_rows,
        "runs": run_rows,
        "survivors": survivors,
        "best_run_id": best_id,
        "best_final_value": final[best_id][1],
        "best_run_survived": stopped_at[best_id] is None,
        "best_surviving_run_id": best_surviving,
        "regret": regret,
        "budget_full": budget_full,
        "budget_used": budget_used,
        "budget_saved_fraction": round(1.0 - budget_used / budget_full, 6) if budget_full else 0.0,
    }


def best_so_far_curve(
    runs: Sequence[Any],
    metric_name: str,
    *,
    maximize: bool = True,
) -> dict:
    """Build the best-so-far / simple-regret curve for an experiment metric.

    Runs that recorded ``metric_name`` are ordered by ``created_at`` (falling
    back to run id). Walking that order, the curve tracks the best metric seen
    so far and the simple regret against the eventual best. Useful for HPO /
    sweep dashboards: after ``t`` trials, how good is the incumbent, and how
    much is left on the table?

    Returns a mapping with ``points`` (one row per contributing run),
    ``final_best``, ``final_regret`` (always 0), ``run_count``, and
    ``maximize``.
    """
    if not metric_name:
        raise ValueError("metric_name is required")

    rows: List[dict] = []
    for run in runs:
        view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
        values = [
            m.get("value")
            for m in view.get("metrics", []) or []
            if m.get("name") == metric_name and m.get("value") is not None
        ]
        if not values:
            continue
        step = None
        for metric in view.get("metrics", []) or []:
            if metric.get("name") == metric_name:
                step = metric.get("step")
        rows.append(
            {
                "run_id": view.get("id"),
                "name": view.get("name"),
                "created_at": view.get("created_at") or "",
                "value": float(values[-1]),
                "step": step,
            }
        )

    rows.sort(key=lambda r: (str(r["created_at"]), str(r["run_id"] or "")))
    if not rows:
        return {
            "metric_name": metric_name,
            "maximize": bool(maximize),
            "run_count": 0,
            "final_best": None,
            "final_regret": None,
            "points": [],
        }

    values = [r["value"] for r in rows]
    eventual_best = max(values) if maximize else min(values)
    best_so_far = None
    points: List[dict] = []
    for index, row in enumerate(rows, start=1):
        value = row["value"]
        if best_so_far is None:
            best_so_far = value
            improved = True
        elif maximize and value > best_so_far:
            best_so_far = value
            improved = True
        elif (not maximize) and value < best_so_far:
            best_so_far = value
            improved = True
        else:
            improved = False
        if maximize:
            regret = eventual_best - best_so_far
        else:
            regret = best_so_far - eventual_best
        points.append(
            {
                "index": index,
                "run_id": row["run_id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "value": value,
                "step": row["step"],
                "best_so_far": best_so_far,
                "simple_regret": regret,
                "improved": improved,
            }
        )

    return {
        "metric_name": metric_name,
        "maximize": bool(maximize),
        "run_count": len(points),
        "final_best": eventual_best,
        "final_regret": 0.0,
        "points": points,
    }


_CURVE_MODELS = ("pow3", "exp3")


def _curve_history(run: Any, metric_name: str) -> List[tuple]:
    """``(step, value)`` pairs for one metric, steps defaulting to 1..n."""
    view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    points: List[tuple] = []
    index = 0
    for metric in view.get("metrics", []) or []:
        if metric.get("name") != metric_name or metric.get("value") is None:
            continue
        index += 1
        value = float(metric["value"])
        if not math.isfinite(value):
            continue
        step = metric.get("step")
        points.append((float(step) if step is not None else float(index), value))
    # Keep the last value logged at each step, ordered by step.
    by_step: Dict[float, float] = {}
    for step, value in points:
        by_step[step] = value
    return sorted(by_step.items())


def _fit_linear(xs: Sequence[float], ys: Sequence[float]) -> tuple:
    """Least-squares ``y = a + b * x``; returns ``(a, b, sse)``."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, sum((y - my) ** 2 for y in ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse


def _fit_curve_model(steps: Sequence[float], values: Sequence[float], model: str) -> dict:
    """Profile least squares over the rate ``c`` on a log grid.

    ``pow3``: ``y = a + b * t ** (-c)`` (Domhan et al., 2015).
    ``exp3``: ``y = a + b * exp(-c * t / t_max)``.
    For a fixed ``c`` both are linear in ``(a, b)``, so the fit is a 1-D grid
    search with closed-form inner solves: deterministic and dependency-free.
    """
    t_max = max(steps)
    best = None
    for i in range(121):
        c = 10 ** (-2 + 3.0 * i / 120)  # 0.01 .. 10
        if model == "pow3":
            xs = [t ** (-c) for t in steps]
        else:
            xs = [math.exp(-c * t / t_max) for t in steps]
        a, b, sse = _fit_linear(xs, values)
        if best is None or sse < best["sse"] - 1e-15:
            best = {"model": model, "a": a, "b": b, "c": c, "sse": sse}
    return best


def _curve_predict(fit: dict, step: float, t_max: float) -> float:
    if fit["model"] == "pow3":
        return fit["a"] + fit["b"] * step ** (-fit["c"])
    return fit["a"] + fit["b"] * math.exp(-fit["c"] * step / t_max)


def learning_curve_extrapolation(
    run: Any,
    metric_name: str,
    target_step: float,
    *,
    model: str = "auto",
    min_points: int = 4,
) -> dict:
    """Fit a saturating learning curve to one run and extrapolate it.

    Fits ``pow3`` (``a + b * t^-c``) and/or ``exp3`` (``a + b * exp(-c t)``)
    to the run's ``(step, value)`` history and predicts the metric at
    ``target_step`` (e.g. the planned final epoch). ``model="auto"`` keeps the
    family with the lower residual sum of squares. The fitted ``a`` is the
    asymptote (the value the curve approaches with unlimited steps).

    Returns ``status="insufficient_data"`` (no prediction) when fewer than
    ``min_points`` distinct steps were logged. Otherwise returns the fitted
    parameters, ``predicted_value``, ``asymptote``, ``rmse``, ``r2``,
    ``last_step``/``last_value`` and ``predicted_change`` from the last value.
    """
    if not metric_name:
        raise ValueError("metric_name is required")
    if model != "auto" and model not in _CURVE_MODELS:
        raise ValueError(f"model must be 'auto' or one of {_CURVE_MODELS}")
    if isinstance(min_points, bool) or int(min_points) < 3:
        raise ValueError("min_points must be an integer >= 3")
    target = float(target_step)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_step must be a positive number")

    view = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    history = _curve_history(view, metric_name)
    base = {
        "run_id": view.get("id"),
        "name": view.get("name"),
        "metric_name": metric_name,
        "target_step": target,
        "n_points": len(history),
    }
    if len(history) < int(min_points):
        return {**base, "status": "insufficient_data", "predicted_value": None}
    steps = [s for s, _ in history]
    values = [v for _, v in history]
    if min(steps) <= 0:
        raise ValueError("metric steps must be positive for curve fitting")

    families = _CURVE_MODELS if model == "auto" else (model,)
    fits = [_fit_curve_model(steps, values, family) for family in families]
    fit = min(fits, key=lambda f: f["sse"])
    t_max = max(steps)
    predicted = _curve_predict(fit, target, t_max)
    mean_v = sum(values) / len(values)
    sst = sum((v - mean_v) ** 2 for v in values)
    rmse = math.sqrt(fit["sse"] / len(values))
    r2 = 1.0 - fit["sse"] / sst if sst > 0 else 1.0
    return {
        **base,
        "status": "ok",
        "model": fit["model"],
        "params": {"a": fit["a"], "b": fit["b"], "c": fit["c"]},
        "asymptote": fit["a"],
        "predicted_value": predicted,
        "last_step": steps[-1],
        "last_value": values[-1],
        "predicted_change": predicted - values[-1],
        "rmse": rmse,
        "r2": r2,
    }


def learning_curve_forecast(
    runs: Sequence[Any],
    metric_name: str,
    target_step: float,
    *,
    maximize: bool = True,
    model: str = "auto",
    min_points: int = 4,
    margin: float = 0.0,
) -> dict:
    """Extrapolate every run's curve and flag runs unlikely to beat the incumbent.

    The incumbent is the best predicted value at ``target_step`` across runs
    (so finished and in-flight runs compete on the same footing). A run gets
    ``terminate_hint=True`` when its predicted value is worse than the
    incumbent by more than ``margin`` -- the learning-curve early-termination
    heuristic of Domhan, Springenberg and Hutter (IJCAI 2015), here with a
    point forecast instead of a posterior.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    rows = [
        learning_curve_extrapolation(
            run, metric_name, target_step, model=model, min_points=min_points
        )
        for run in runs
    ]
    rows = [r for r in rows if r["n_points"] > 0]
    fitted = [r for r in rows if r["status"] == "ok"]
    incumbent = None
    if fitted:
        chooser = max if maximize else min
        incumbent = chooser(fitted, key=lambda r: r["predicted_value"])
    for row in rows:
        if row["status"] != "ok" or incumbent is None:
            row["terminate_hint"] = False
            row["gap_to_incumbent"] = None
            continue
        if maximize:
            gap = incumbent["predicted_value"] - row["predicted_value"]
        else:
            gap = row["predicted_value"] - incumbent["predicted_value"]
        row["gap_to_incumbent"] = gap
        row["terminate_hint"] = bool(gap > margin)
    rows.sort(
        key=lambda r: (
            r["status"] != "ok",
            -(r["predicted_value"] or 0.0) if maximize else (r["predicted_value"] or 0.0),
            str(r["run_id"] or ""),
        )
    )
    return {
        "metric_name": metric_name,
        "maximize": bool(maximize),
        "target_step": float(target_step),
        "margin": float(margin),
        "incumbent_run_id": incumbent["run_id"] if incumbent else None,
        "incumbent_predicted_value": incumbent["predicted_value"] if incumbent else None,
        "run_count": len(rows),
        "fitted_count": len(fitted),
        "terminate_count": sum(1 for r in rows if r["terminate_hint"]),
        "runs": rows,
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
