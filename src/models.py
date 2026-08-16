"""ML Experiment Tracker - Core Models"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


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


@dataclass
class Artifact:
    name: str
    artifact_type: ArtifactType
    path: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.artifact_type.value,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    error: Optional[str] = None

    def log_param(self, name: str, value: Any) -> None:
        self.params[name] = value

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.metrics.append(Metric(name=name, value=value, step=step))

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

    def log_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

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
            "metrics": [{"name": m.name, "value": m.value, "step": m.step, "timestamp": m.timestamp.isoformat()} for m in self.metrics],
            "artifacts": [{"name": a.name, "type": a.artifact_type.value, "path": a.path, "size": a.size_bytes} for a in self.artifacts],
            "tags": self.tags,
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
        return run


@dataclass
class Experiment:
    name: str
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        exp = cls(id=data["id"], name=data["name"], description=data.get("description", ""), tags=data.get("tags", []))
        exp.created_at = datetime.fromisoformat(data["created_at"])
        exp.updated_at = datetime.fromisoformat(data["updated_at"])
        exp.runs = [Run.from_dict(r) for r in data.get("runs", [])]
        return exp
