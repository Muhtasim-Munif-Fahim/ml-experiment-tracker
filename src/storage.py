"""Storage backends for experiment artifacts and metadata."""

from __future__ import annotations

import os
import shutil
import json
import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable, List, Optional

from .models import AlertRule, Run, Experiment, Artifact

EXPERIMENT_BUNDLE_VERSION = 1


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_experiment(self, exp_data: dict) -> None:
        pass

    @abstractmethod
    def load_experiment(self, exp_id: str) -> Optional[dict]:
        pass

    @abstractmethod
    def list_experiments(self, include_archived: bool = False) -> List[dict]:
        pass

    @abstractmethod
    def save_run(self, run_data: dict) -> None:
        pass

    @abstractmethod
    def load_run(self, run_id: str) -> Optional[dict]:
        pass

    @abstractmethod
    def list_runs(self, experiment_id: str) -> List[dict]:
        pass

    @abstractmethod
    def save_artifact(self, artifact_id: str, file_path: Path) -> str:
        pass

    @abstractmethod
    def load_artifact(self, artifact_path: str) -> bytes:
        pass


class LocalStorageBackend:
    """Local filesystem storage backend."""

    def __init__(self, base_path: Path = Path("./mlruns")):
        self.base_path = Path(base_path)
        self.experiments_dir = self.base_path / "experiments"
        self.artifacts_dir = self.base_path / "artifacts"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _exp_path(self, exp_id: str) -> Path:
        return self.experiments_dir / f"{exp_id}.json"

    def _run_path(self, run_id: str) -> Path:
        return self.experiments_dir / "runs" / f"{run_id}.json"

    def _artifact_path(self, artifact_id: str) -> Path:
        return self.artifacts_dir / artifact_id

    def save_experiment(self, exp_data: dict) -> None:
        path = self._exp_path(exp_data["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(exp_data, f, indent=2, default=str)

    def load_experiment(self, exp_id: str) -> Optional[dict]:
        path = self._exp_path(exp_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_experiments(self, include_archived: bool = False) -> List[dict]:
        experiments = []
        for path in self.experiments_dir.glob("*.json"):
            with open(path) as f:
                experiment = json.load(f)
            if not include_archived and experiment.get("archived"):
                continue
            experiments.append(experiment)
        return experiments

    def set_experiment_archived(self, exp_id: str, archived: bool = True) -> Optional[dict]:
        """Mark an experiment archived or active without deleting its records."""
        experiment = self.load_experiment(exp_id)
        if experiment is None:
            return None
        experiment["archived"] = bool(archived)
        experiment["updated_at"] = datetime.utcnow().isoformat()
        self.save_experiment(experiment)
        return experiment

    def save_run(self, run_data: dict) -> None:
        path = self._run_path(run_data["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(run_data, f, indent=2, default=str)

    def load_run(self, run_id: str) -> Optional[dict]:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def delete_run(self, run_id: str) -> bool:
        """Remove a run record, returning True if it existed."""
        path = self._run_path(run_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_runs(self, experiment_id: str) -> List[dict]:
        runs_dir = self.experiments_dir / "runs"
        if not runs_dir.exists():
            return []
        runs = []
        for path in runs_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
                if data.get("experiment_id") == experiment_id:
                    runs.append(data)
        return runs

    @staticmethod
    def _validate_metric_filters(
        metric_name: Optional[str],
        min_metric: Optional[float],
        max_metric: Optional[float],
    ) -> None:
        if (min_metric is not None or max_metric is not None) and not metric_name:
            raise ValueError("metric_name is required for metric bounds")
        if min_metric is not None and max_metric is not None and min_metric > max_metric:
            raise ValueError("min_metric cannot exceed max_metric")

    @staticmethod
    def _run_matches_filters(
        run: dict,
        *,
        allowed_statuses: set,
        required_tags: dict,
        fragment: Optional[str],
        metric_name: Optional[str],
        min_metric: Optional[float],
        max_metric: Optional[float],
    ) -> bool:
        if allowed_statuses and run.get("status") not in allowed_statuses:
            return False
        run_tags = run.get("tags", {})
        if not isinstance(run_tags, dict) or any(
            run_tags.get(key) != value for key, value in required_tags.items()
        ):
            return False
        if fragment and fragment not in str(run.get("name", "")).casefold():
            return False
        if metric_name is not None:
            metric_values = [
                metric.get("value")
                for metric in run.get("metrics", [])
                if metric.get("name") == metric_name
            ]
            if not metric_values:
                return False
            latest_metric = float(metric_values[-1])
            if min_metric is not None and latest_metric < min_metric:
                return False
            if max_metric is not None and latest_metric > max_metric:
                return False
        return True

    def query_runs(
        self,
        experiment_id: str,
        *,
        statuses: Optional[List[str]] = None,
        tags: Optional[dict[str, str]] = None,
        name_contains: Optional[str] = None,
        metric_name: Optional[str] = None,
        min_metric: Optional[float] = None,
        max_metric: Optional[float] = None,
    ) -> List[dict]:
        """Filter runs of one experiment by metadata and latest metric value."""

        self._validate_metric_filters(metric_name, min_metric, max_metric)
        allowed_statuses = set(statuses or [])
        required_tags = tags or {}
        fragment = name_contains.casefold() if name_contains else None
        matches = [
            run
            for run in self.list_runs(experiment_id)
            if self._run_matches_filters(
                run,
                allowed_statuses=allowed_statuses,
                required_tags=required_tags,
                fragment=fragment,
                metric_name=metric_name,
                min_metric=min_metric,
                max_metric=max_metric,
            )
        ]
        return sorted(
            matches,
            key=lambda run: str(run.get("created_at", "")),
            reverse=True,
        )

    def search_runs(
        self,
        *,
        statuses: Optional[List[str]] = None,
        tags: Optional[dict[str, str]] = None,
        name_contains: Optional[str] = None,
        metric_name: Optional[str] = None,
        min_metric: Optional[float] = None,
        max_metric: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Search runs across every experiment, newest first, paginated.

        Returns ``{"total": <all matches>, "runs": [<requested page>]}`` so
        callers can page without losing count.
        """
        self._validate_metric_filters(metric_name, min_metric, max_metric)
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        allowed_statuses = set(statuses or [])
        required_tags = tags or {}
        fragment = name_contains.casefold() if name_contains else None
        matches = []
        for experiment in self.list_experiments(include_archived=True):
            for run in self.list_runs(experiment["id"]):
                if self._run_matches_filters(
                    run,
                    allowed_statuses=allowed_statuses,
                    required_tags=required_tags,
                    fragment=fragment,
                    metric_name=metric_name,
                    min_metric=min_metric,
                    max_metric=max_metric,
                ):
                    matches.append(run)
        matches.sort(key=lambda run: str(run.get("created_at", "")), reverse=True)
        return {"total": len(matches), "runs": matches[offset : offset + limit]}

    def save_artifact(self, artifact_id: str, file_path: Path) -> str:
        dest = self.artifacts_dir / artifact_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        return str(dest)

    def load_artifact(self, artifact_path: str) -> bytes:
        with open(artifact_path, "rb") as f:
            return f.read()

    def artifact_checksum(self, artifact_path: str) -> str:
        """Return the SHA-256 checksum of a stored artifact."""

        digest = hashlib.sha256()
        with open(artifact_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_artifact(self, artifact_path: str, expected_sha256: str) -> bool:
        """Verify artifact bytes against a previously recorded checksum."""

        normalized = expected_sha256.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("expected_sha256 must be a 64-character hex digest")
        return self.artifact_checksum(artifact_path) == normalized

    def run_leaderboard(
        self,
        experiment_id: str,
        metric_name: str,
        *,
        maximize: bool = True,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Rank runs of one experiment by their latest value of a metric.

        Returns a list of ``{"run_id", "name", "value", "step"}`` entries
        ordered by metric value so the best run comes first. Runs without any
        recorded value for the metric are omitted.
        """
        if not metric_name:
            raise ValueError("metric_name is required")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        ranked = []
        for run in self.list_runs(experiment_id):
            values = [
                metric.get("value")
                for metric in run.get("metrics", [])
                if metric.get("name") == metric_name
            ]
            if not values:
                continue
            step = None
            for metric in run.get("metrics", []):
                if metric.get("name") == metric_name:
                    step = metric.get("step")
            ranked.append(
                {
                    "run_id": run.get("id"),
                    "name": run.get("name"),
                    "value": float(values[-1]),
                    "step": step,
                }
            )
        ranked.sort(key=lambda entry: entry["value"], reverse=maximize)
        if limit is not None:
            ranked = ranked[:limit]
        return ranked

    def add_note(self, run_id: str, body: str) -> dict:
        """Append a markdown note to a run and return the stored note."""
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("note body must be a non-empty string")
        note = {
            "id": f"note_{uuid.uuid4().hex[:8]}",
            "body": body,
            "created_at": datetime.utcnow().isoformat(),
        }
        run_data.setdefault("notes", []).append(note)
        self.save_run(run_data)
        return note

    def list_notes(self, run_id: str) -> List[dict]:
        """Return the notes attached to a run in creation order."""
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        return list(run_data.get("notes", []))

    def update_note(self, run_id: str, note_id: str, body: str) -> Optional[dict]:
        """Replace a note's body; returns None when the note does not exist."""
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("note body must be a non-empty string")
        for note in run_data.get("notes", []):
            if note.get("id") == note_id:
                note["body"] = body
                self.save_run(run_data)
                return note
        return None

    def delete_note(self, run_id: str, note_id: str) -> bool:
        """Remove one note, returning True if it existed."""
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        notes = run_data.get("notes", [])
        remaining = [note for note in notes if note.get("id") != note_id]
        if len(remaining) == len(notes):
            return False
        run_data["notes"] = remaining
        self.save_run(run_data)
        return True

    def save_alert_rule(self, exp_id: str, rule: dict) -> dict:
        """Attach a metric threshold rule to an experiment."""
        experiment = self.load_experiment(exp_id)
        if experiment is None:
            raise KeyError(f"experiment not found: {exp_id}")
        experiment.setdefault("alert_rules", []).append(rule)
        self.save_experiment(experiment)
        return rule

    def list_alert_rules(self, exp_id: str) -> List[dict]:
        """Return the alert rules configured for an experiment."""
        experiment = self.load_experiment(exp_id)
        if experiment is None:
            raise KeyError(f"experiment not found: {exp_id}")
        return list(experiment.get("alert_rules", []))

    def delete_alert_rule(self, exp_id: str, rule_id: str) -> bool:
        """Remove one alert rule, returning True if it existed."""
        experiment = self.load_experiment(exp_id)
        if experiment is None:
            raise KeyError(f"experiment not found: {exp_id}")
        rules = experiment.get("alert_rules", [])
        remaining = [rule for rule in rules if rule.get("id") != rule_id]
        if len(remaining) == len(rules):
            return False
        experiment["alert_rules"] = remaining
        self.save_experiment(experiment)
        return True

    def apply_alert_rules(
        self, run_data: dict, metric_points: Iterable[dict]
    ) -> List[dict]:
        """Evaluate freshly logged metric points against the experiment's rules.

        Appends one alert entry per triggered rule to the run record before
        persisting it. Returns only the alerts created by this call.
        """
        rules = [
            AlertRule.from_dict(raw)
            for raw in self.list_alert_rules(run_data["experiment_id"])
        ]
        if not rules:
            return []
        triggered = []
        for point in metric_points:
            value = float(point["value"])
            for rule in rules:
                if rule.metric_name == point.get("name") and rule.matches(value):
                    triggered.append(
                        {
                            "rule_id": rule.id,
                            "metric_name": point.get("name"),
                            "comparator": rule.comparator,
                            "threshold": rule.threshold,
                            "value": value,
                            "step": point.get("step"),
                            "triggered_at": datetime.utcnow().isoformat(),
                        }
                    )
        if triggered:
            run_data.setdefault("alerts", []).extend(triggered)
            run_data["updated_at"] = datetime.utcnow().isoformat()
            self.save_run(run_data)
        return triggered

    def export_experiment(self, exp_id: str, destination: Path) -> Path:
        """Export experiment metadata and all run records as one JSON bundle."""

        experiment = self.load_experiment(exp_id)
        if experiment is None:
            raise KeyError(f"experiment not found: {exp_id}")
        payload = {
            "schema_version": EXPERIMENT_BUNDLE_VERSION,
            "experiment": experiment,
            "runs": self.list_runs(exp_id),
        }
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return target

    def import_experiment(self, source: Path) -> str:
        """Import a bundle while refusing to overwrite existing records."""

        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read experiment bundle '{path}': {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("experiment bundle must contain a JSON object")
        if payload.get("schema_version") != EXPERIMENT_BUNDLE_VERSION:
            raise ValueError(
                "unsupported experiment bundle version: "
                f"{payload.get('schema_version')!r}"
            )
        experiment = payload.get("experiment")
        runs = payload.get("runs")
        if not isinstance(experiment, dict) or not experiment.get("id"):
            raise ValueError("experiment bundle is missing experiment.id")
        if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
            raise ValueError("experiment bundle runs must be a list of objects")

        exp_id = str(experiment["id"])
        if self.load_experiment(exp_id) is not None:
            raise FileExistsError(f"experiment already exists: {exp_id}")
        for run in runs:
            if run.get("experiment_id") != exp_id or not run.get("id"):
                raise ValueError("every bundled run must belong to experiment.id")
            if self.load_run(str(run["id"])) is not None:
                raise FileExistsError(f"run already exists: {run['id']}")

        self.save_experiment(experiment)
        for run in runs:
            self.save_run(run)
        return exp_id


class S3StorageBackend:
    """S3-compatible storage backend (stub)."""

    def __init__(self, bucket: str, prefix: str = "mlruns/"):
        self.bucket = bucket
        self.prefix = prefix
        # Implementation would use boto3

    def save_experiment(self, exp_data: dict) -> None:
        raise NotImplementedError

    def load_experiment(self, exp_id: str) -> Optional[dict]:
        raise NotImplementedError

    def list_experiments(self, include_archived: bool = False) -> List[dict]:
        raise NotImplementedError

    def save_run(self, run_data: dict) -> None:
        raise NotImplementedError

    def load_run(self, run_id: str) -> Optional[dict]:
        raise NotImplementedError

    def list_runs(self, experiment_id: str) -> List[dict]:
        raise NotImplementedError

    def save_artifact(self, artifact_id: str, file_path: Path) -> str:
        raise NotImplementedError

    def load_artifact(self, artifact_path: str) -> bytes:
        raise NotImplementedError


class StorageFactory:
    @staticmethod
    def create(backend: str = "local", **kwargs) -> "StorageBackend":
        if backend == "local":
            return LocalStorageBackend(**kwargs)
        elif backend == "s3":
            return S3StorageBackend(**kwargs)
        else:
            raise ValueError(f"Unknown storage backend: {backend}")
