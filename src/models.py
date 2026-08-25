"""ML Experiment Tracker - Core Models"""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


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
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Artifact:
    name: str
    artifact_type: ArtifactType
    path: str
    size_bytes: int
    checksum_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    error: Optional[str] = None

    def log_param(self, name: str, value: Any) -> None:
        self.params[name] = value

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
        self.finished_at = datetime.utcnow()
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
        return run


@dataclass
class Experiment:
    name: str
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    archived: bool = False
    tags: List[str] = field(default_factory=list)
    runs: List[Run] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_run(self, run: Run) -> None:
        self.runs.append(run)
        self.updated_at = datetime.utcnow()

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
