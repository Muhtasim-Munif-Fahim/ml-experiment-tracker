"""Storage backends for experiment artifacts and metadata."""

from __future__ import annotations

import csv
import os
import shutil
import json
import hashlib
import uuid
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, List, Optional

from .models import (
    AlertRule,
    Artifact,
    ArtifactType,
    Experiment,
    Run,
    pearson_correlation,
    standardize_series,
)

EXPERIMENT_BUNDLE_VERSION = 1
RUN_SNAPSHOT_VERSION = 1


def _median(sorted_values: List[float]) -> float:
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _describe_metric_values(values: List[float]) -> dict:
    """Descriptive statistics over a flat list of numeric observations."""

    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
        }
    mean = sum(values) / count
    if count >= 2:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    else:
        variance = 0.0
    ordered = sorted(values)
    return {
        "count": count,
        "mean": mean,
        "std": variance ** 0.5,
        "min": ordered[0],
        "max": ordered[-1],
        "median": _median(ordered),
    }


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
        experiment["updated_at"] = datetime.now(timezone.utc).isoformat()
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

    def move_run(self, run_id: str, target_exp_id: str) -> dict:
        """Move a run to a different experiment, returning the updated run.

        The run's ``experiment_id`` is rewritten and both the source and
        destination experiment records are updated in-place: the source
        drops the run id and the destination appends it. The run record is
        re-persisted under its existing ``run_id``. Raises ``KeyError``
        when either the run or the destination experiment cannot be found;
        raises ``ValueError`` when the run already belongs to the target
        experiment (no-op).
        """
        run = self.load_run(run_id)
        if run is None:
            raise KeyError(f"run not found: {run_id}")
        target = self.load_experiment(target_exp_id)
        if target is None:
            raise KeyError(f"experiment not found: {target_exp_id}")

        source_exp_id = run.get("experiment_id")
        if source_exp_id == target_exp_id:
            raise ValueError(
                f"run {run_id} already belongs to experiment {target_exp_id}"
            )

        if source_exp_id:
            source = self.load_experiment(source_exp_id)
            if source is not None and isinstance(source.get("runs"), list):
                source["runs"] = [
                    r for r in source["runs"]
                    if not (isinstance(r, dict) and r.get("id") == run_id)
                ]
                source["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.save_experiment(source)

        run["experiment_id"] = target_exp_id
        run["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_run(run)

        if isinstance(target.get("runs"), list):
            if not any(
                isinstance(r, dict) and r.get("id") == run_id
                for r in target["runs"]
            ):
                target["runs"].append({
                    "id": run["id"],
                    "name": run.get("name"),
                    "status": run.get("status"),
                    "created_at": run.get("created_at"),
                })
        target["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_experiment(target)

        return run

    def duplicate_run(
        self,
        run_id: str,
        *,
        name: Optional[str] = None,
        include_metrics: bool = False,
    ) -> dict:
        """Copy a run into a fresh record with a new id.

        The copy keeps the source run's experiment, params and tags; when
        ``include_metrics`` is set the recorded metric history is copied too.
        Artifacts are not duplicated because their bytes are owned by the
        original run record. The new run starts in the running state.
        """
        source = self.load_run(run_id)
        if source is None:
            raise KeyError(f"run not found: {run_id}")
        if not isinstance(include_metrics, bool):
            raise ValueError("include_metrics must be a boolean")
        copy = {
            "id": str(uuid.uuid4()),
            "experiment_id": source.get("experiment_id"),
            "name": name or f"{source.get('name', 'run')} (copy)",
            "parent_run_id": run_id,
            "status": "running",
            "params": dict(source.get("params", {}) or {}),
            "tags": dict(source.get("tags", {}) or {}),
            "metrics": (
                [dict(metric) for metric in source.get("metrics", [])]
                if include_metrics
                else []
            ),
            "artifacts": [],
            "alerts": [],
            "notes": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error": None,
        }
        self.save_run(copy)
        return copy

    def import_runs(self, experiment_id: str, run_specs: List[dict]) -> List[dict]:
        """Create several runs from explicit specs in a single call.

        Every spec must carry a non-empty ``name`` and may include ``params``
        and ``tags`` mappings. Runs receive fresh ids and start in the running
        state, so the call is safe to retry and never overwrites records.
        """
        if self.load_experiment(experiment_id) is None:
            raise KeyError(f"experiment not found: {experiment_id}")
        if not isinstance(run_specs, list) or not run_specs:
            raise ValueError("run_specs must be a non-empty list")
        if len(run_specs) > 1000:
            raise ValueError("cannot import more than 1000 runs at once")

        created = []
        for spec in run_specs:
            if not isinstance(spec, dict):
                raise ValueError("every run spec must be an object")
            name = spec.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("every run spec must have a non-empty name")
            params = spec.get("params") or {}
            tags = spec.get("tags") or {}
            if not isinstance(params, dict):
                raise ValueError(f"params for run {name!r} must be a mapping")
            if not isinstance(tags, dict):
                raise ValueError(f"tags for run {name!r} must be a mapping")
            run = {
                "id": str(uuid.uuid4()),
                "experiment_id": experiment_id,
                "name": name,
                "parent_run_id": None,
                "status": "running",
                "params": dict(params),
                "tags": dict(tags),
                "metrics": [],
                "artifacts": [],
                "alerts": [],
                "notes": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "error": None,
            }
            self.save_run(run)
            created.append(run)
        return created

    def list_run_tags(self, run_id: str) -> dict:
        """Return the key/value tags attached to a run."""
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        return dict(run_data.get("tags", {}))

    def set_run_tag(self, run_id: str, name: str, value: str) -> dict:
        """Set one key/value tag on a run, replacing any existing value."""
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tag name must be a non-empty string")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("tag value must be a non-empty string")
        tags = run_data.get("tags", {})
        if not isinstance(tags, dict):
            raise ValueError("run tags are not a mapping")
        tags[name] = value
        run_data["tags"] = tags
        run_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_run(run_data)
        return {"name": name, "value": value}

    def delete_run_tag(self, run_id: str, name: str) -> bool:
        """Remove one tag from a run, returning True if it existed."""
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        tags = run_data.get("tags", {})
        if not isinstance(tags, dict) or name not in tags:
            return False
        del tags[name]
        run_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_run(run_data)
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

    def storage_stats(self) -> dict:
        """Count experiments, runs and artifact records plus on-disk usage."""
        experiments = self.list_experiments(include_archived=True)
        run_count = 0
        artifact_count = 0
        artifact_bytes = 0
        for experiment in experiments:
            for run in self.list_runs(experiment["id"]):
                run_count += 1
                for artifact in run.get("artifacts", []):
                    artifact_count += 1
                    artifact_bytes += int(
                        artifact.get("size_bytes", artifact.get("size", 0))
                    )
        store_bytes = sum(
            path.stat().st_size for path in self.artifacts_dir.glob("*") if path.is_file()
        )
        return {
            "experiment_count": len(experiments),
            "run_count": run_count,
            "artifact_count": artifact_count,
            "artifact_bytes": artifact_bytes,
            "artifact_store_bytes": store_bytes,
        }

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

    def experiment_artifacts(self, exp_id: str) -> List[dict]:
        """Collect artifact records across every run of an experiment.

        Runs are ordered newest first, mirroring ``query_runs``. Each entry
        carries the owning run id and name alongside the artifact record.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")
        runs = sorted(
            self.list_runs(exp_id),
            key=lambda run: str(run.get("created_at", "")),
            reverse=True,
        )
        inventory = []
        for run in runs:
            for artifact in run.get("artifacts", []):
                entry = dict(artifact)
                entry["run_id"] = run.get("id")
                entry["run_name"] = run.get("name")
                inventory.append(entry)
        return inventory
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

    def export_run_artifacts_zip(self, run_id: str, destination: Path) -> Path:
        """Bundle every stored artifact of a run into one zip archive.

        Missing artifact files are skipped so the archive never fails on a
        stale record. When two artifacts share a name, the later arcname is
        prefixed with its artifact id. A ``manifest.json`` entry lists each
        artifact's metadata alongside its bytes.
        """
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")

        used_names = set()
        entries = []
        for artifact in run_data.get("artifacts", []):
            file_path = Path(artifact["path"])
            if not file_path.exists():
                continue
            arcname = artifact.get("name") or file_path.name
            if arcname in used_names:
                arcname = f"{artifact.get('artifact_id', 'artifact')}-{arcname}"
            used_names.add(arcname)
            entries.append((arcname, file_path, artifact))

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(
                    [
                        {
                            "artifact_id": artifact.get("artifact_id"),
                            "name": artifact.get("name"),
                            "type": artifact.get("type"),
                            "size_bytes": artifact.get(
                                "size_bytes", artifact.get("size")
                            ),
                            "checksum_sha256": artifact.get("checksum_sha256"),
                        }
                        for _, _, artifact in entries
                    ],
                    indent=2,
                ),
            )
            for arcname, file_path, _artifact in entries:
                archive.write(file_path, arcname)
        return target

    def update_artifact(
        self, run_id: str, artifact_ref: str, updates: dict
    ) -> Optional[dict]:
        """Update editable metadata of one artifact in place.

        Accepts ``name``, ``artifact_type``, and ``metadata`` changes without
        touching the stored bytes. Returns the updated artifact entry, or None
        when the artifact cannot be found.
        """
        run_data = self.load_run(run_id)
        if run_data is None:
            raise KeyError(f"run not found: {run_id}")
        for artifact in run_data.get("artifacts", []):
            if (
                artifact.get("artifact_id") != artifact_ref
                and artifact.get("name") != artifact_ref
            ):
                continue
            if "name" in updates:
                name = updates["name"]
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("artifact name must be a non-empty string")
                artifact["name"] = name
            if "artifact_type" in updates:
                try:
                    artifact["type"] = ArtifactType(updates["artifact_type"]).value
                except ValueError as exc:
                    raise ValueError(
                        f"unknown artifact type: {updates['artifact_type']!r}"
                    ) from exc
            if "metadata" in updates:
                if not isinstance(updates["metadata"], dict):
                    raise ValueError("artifact metadata must be a mapping")
                artifact["metadata"] = updates["metadata"]
            run_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.save_run(run_data)
            return artifact
        return None

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
            "created_at": datetime.now(timezone.utc).isoformat(),
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
                            "triggered_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
        if triggered:
            run_data.setdefault("alerts", []).extend(triggered)
            run_data["updated_at"] = datetime.now(timezone.utc).isoformat()
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


    def build_run_snapshot(self, run_id: str) -> dict:
        """Assemble a self-contained JSON snapshot of one run."""
        run = self.load_run(run_id)
        if run is None:
            raise KeyError(f"run not found: {run_id}")
        experiment = self.load_experiment(run.get("experiment_id", ""))
        return {
            "schema_version": RUN_SNAPSHOT_VERSION,
            "snapshot_type": "run",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": (
                {"id": experiment["id"], "name": experiment.get("name", "")}
                if experiment
                else None
            ),
            "run": run,
            "artifact_manifest": [
                {
                    "name": artifact.get("name"),
                    "type": artifact.get("type"),
                    "size_bytes": artifact.get("size_bytes", artifact.get("size")),
                    "checksum_sha256": artifact.get("checksum_sha256"),
                }
                for artifact in run.get("artifacts", [])
            ],
        }

    def export_run_snapshot(self, run_id: str, destination: Path) -> Path:
        """Write one run's snapshot as a standalone JSON file."""
        payload = self.build_run_snapshot(run_id)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return target

    def export_experiment_artifacts_zip(self, exp_id: str, destination: Path) -> Path:
        """Bundle every stored artifact of an experiment into one zip archive.

        Missing artifact files are skipped so the archive never fails on a
        stale record. When two artifacts share a name, the arcname is prefixed
        with ``<run_id>-<artifact_id>-`` to keep them distinct. A
        ``manifest.json`` entry lists each artifact's metadata alongside its
        bytes and owning run.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")

        runs = self.list_runs(exp_id)
        used_names = set()
        entries = []
        for run in runs:
            for artifact in run.get("artifacts", []):
                file_path = Path(artifact["path"])
                if not file_path.exists():
                    continue
                arcname = artifact.get("name") or file_path.name
                prefixed = f"{run.get('id', 'run')}-{artifact.get('artifact_id', 'artifact')}-{arcname}"
                if arcname in used_names:
                    arcname = prefixed
                used_names.add(arcname)
                entries.append(
                    {
                        "arcname": arcname,
                        "file_path": file_path,
                        "run_id": run.get("id"),
                        "run_name": run.get("name"),
                        "artifact": artifact,
                    }
                )

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(
                    [
                        {
                            "run_id": entry["run_id"],
                            "run_name": entry["run_name"],
                            "artifact_id": entry["artifact"].get("artifact_id"),
                            "name": entry["artifact"].get("name"),
                            "type": entry["artifact"].get("type"),
                            "size_bytes": entry["artifact"].get(
                                "size_bytes", entry["artifact"].get("size")
                            ),
                            "checksum_sha256": entry["artifact"].get("checksum_sha256"),
                        }
                        for entry in entries
                    ],
                    indent=2,
                ),
            )
            for entry in entries:
                archive.write(entry["file_path"], entry["arcname"])
        return target

    def experiment_metric_pivot(
        self, exp_id: str, metric_names: Optional[List[str]] = None
    ) -> dict:
        """Build a wide-format metric table for every run of one experiment.

        Returns ``{"columns": [...], "rows": [[...], ...]}`` where each row
        starts with ``run_id`` and ``run_name`` followed by the latest value
        for each requested metric. When ``metric_names`` is omitted, every
        metric observed across the experiment is included.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")

        runs = sorted(
            self.list_runs(exp_id),
            key=lambda run: str(run.get("created_at", "")),
            reverse=True,
        )
        if not runs:
            return {"columns": ["run_id", "run_name"], "rows": []}

        observed = {}
        for run in runs:
            for metric in run.get("metrics", []):
                name = metric.get("name")
                if name and name not in observed:
                    observed[name] = True
        if metric_names is None:
            metric_names = sorted(observed.keys())
        else:
            metric_names = [name for name in metric_names if name in observed]

        columns = ["run_id", "run_name"] + metric_names
        rows = []
        for run in runs:
            latest: Dict[str, Optional[float]] = {name: None for name in metric_names}
            for metric in run.get("metrics", []):
                name = metric.get("name")
                if name in latest:
                    latest[name] = float(metric.get("value", 0) or 0)
            row = [run.get("id"), run.get("name")] + [latest[name] for name in metric_names]
            rows.append(row)
        return {"columns": columns, "rows": rows}

    def experiment_parameter_correlation(
        self,
        exp_id: str,
        metric_name: str,
    ) -> List[dict]:
        """Rank parameters by Pearson correlation with a target metric.

        For every numeric parameter configured across the experiment's runs,
        pairs the parameter value with each run's latest value of
        ``metric_name`` and computes the Pearson coefficient. Returns entries
        ``{"param_name", "correlation", "run_count"}`` sorted by absolute
        correlation descending; parameters with fewer than two paired
        observations or undefined correlation are omitted. Raises ``KeyError``
        when the experiment does not exist.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")

        paired: Dict[str, List[tuple]] = {}
        for run in self.list_runs(exp_id):
            target = [
                metric.get("value")
                for metric in run.get("metrics", [])
                if metric.get("name") == metric_name
            ]
            if not target:
                continue
            latest = float(target[-1])
            params = run.get("params", {}) or {}
            for name, value in params.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                paired.setdefault(name, []).append((float(value), latest))

        results: List[dict] = []
        for name, observations in paired.items():
            if len(observations) < 2:
                continue
            xs = [pair[0] for pair in observations]
            ys = [pair[1] for pair in observations]
            r = pearson_correlation(xs, ys)
            if r is None:
                continue
            results.append(
                {
                    "param_name": name,
                    "correlation": r,
                    "run_count": len(observations),
                }
            )
        results.sort(key=lambda entry: abs(entry["correlation"]), reverse=True)
        return results

    def experiment_metric_long(
        self,
        exp_id: str,
        *,
        metric_names: Optional[List[str]] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[dict]:
        """Return every metric observation in an experiment as long-form rows.

        Each row carries ``run_id``, ``run_name``, ``metric_name``, ``step``,
        ``value`` and ``timestamp`` so plotting libraries can render the
        full time-series across runs. ``metric_names`` restricts the output
        to a chosen subset (unknown names are silently dropped); an inclusive
        ``start_step`` / ``end_step`` range further narrows the result.
        Rows are sorted by (run_name, metric_name, step) so the output is
        reproducible across calls.

        Distinct from :meth:`experiment_metric_pivot`, which returns the
        latest value per run in wide form. This helper exposes every logged
        observation so consumers can plot training curves, not just the
        final accuracy.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")
        if start_step is not None and end_step is not None and start_step > end_step:
            raise ValueError("start_step must not exceed end_step")

        runs = self.list_runs(exp_id)
        if metric_names is not None:
            wanted = {str(name) for name in metric_names}
        else:
            wanted = None

        rows: List[dict] = []
        for run in runs:
            run_id = run.get("id")
            run_name = run.get("name")
            for metric in run.get("metrics", []):
                name = metric.get("name")
                if wanted is not None and name not in wanted:
                    continue
                step = metric.get("step")
                if start_step is not None and step is not None and step < start_step:
                    continue
                if end_step is not None and step is not None and step > end_step:
                    continue
                rows.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "metric_name": name,
                    "step": step,
                    "value": metric.get("value"),
                    "timestamp": metric.get("timestamp"),
                })

        rows.sort(key=lambda row: (
            str(row.get("run_name") or ""),
            str(row.get("metric_name") or ""),
            row.get("step") if row.get("step") is not None else -1,
        ))
        return rows

    def experiment_metric_baseline(
        self,
        exp_id: str,
        metric_name: str,
    ) -> dict:
        """Compute experiment-wide descriptive statistics for one metric.

        Aggregates every recorded observation of ``metric_name`` across all
        runs of the experiment and returns ``count``, ``mean``, sample
        ``std``, ``min``, ``max`` and ``median``. Raises ``KeyError`` when the
        experiment is missing; a metric with no observations yields zero count
        and ``None`` statistics.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")
        values: List[float] = []
        for run in self.list_runs(exp_id):
            for metric in run.get("metrics", []):
                if metric.get("name") == metric_name:
                    try:
                        values.append(float(metric.get("value")))
                    except (TypeError, ValueError):
                        continue
        return _describe_metric_values(values)

    def standardize_run_metric(
        self,
        exp_id: str,
        run_id: str,
        metric_name: str,
        outlier_threshold: float = 2.0,
    ) -> dict:
        """Z-score standardize one run's metric series against the experiment baseline.

        Returns ``{"metric_name", "baseline": {...}, "points": [...]}`` where
        each point carries ``step``, ``value`` and ``zscore``; points whose
        absolute z-score exceeds ``outlier_threshold`` are flagged
        ``is_outlier``. The baseline is derived from every observation of the
        metric across all runs of the experiment. Raises ``KeyError`` for a
        missing experiment or run.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")
        run = self.load_run(run_id)
        if run is None:
            raise KeyError(f"run not found: {run_id}")

        baseline = self.experiment_metric_baseline(exp_id, metric_name)
        mean = baseline["mean"]
        std = baseline["std"]
        steps: List[Any] = []
        values: List[float] = []
        for metric in run.get("metrics", []):
            if metric.get("name") != metric_name:
                continue
            try:
                values.append(float(metric.get("value")))
            except (TypeError, ValueError):
                continue
            steps.append(metric.get("step"))

        if mean is None or std is None:
            zscores = [0.0 for _ in values]
        else:
            zscores = standardize_series(values, mean, std)

        points: List[dict] = []
        for step, value, zscore in zip(steps, values, zscores):
            points.append(
                {
                    "step": step,
                    "value": value,
                    "zscore": zscore,
                    "is_outlier": abs(zscore) > outlier_threshold,
                }
            )
        return {
            "metric_name": metric_name,
            "baseline": baseline,
            "points": points,
        }

    def experiment_snapshot(
        self,
        exp_id: str,
        *,
        metric_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """One row per run with the latest value of each requested metric.

        Each row carries ``run_id``, ``run_name``, ``status``,
        ``created_at``, ``updated_at``, ``finished_at``, ``error``,
        ``metric_count``, ``artifact_count``, ``tag_count``, ``note_count``,
        ``params`` (JSON), and a column per requested metric. ``metric_names``
        defaults to every metric observed across the experiment so the
        CSV output is wide-form. Raises ``KeyError`` when the experiment
        is missing.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")

        runs = self.list_runs(exp_id)
        observed: dict[str, bool] = {}
        for run in runs:
            for metric in run.get("metrics", []):
                name = metric.get("name")
                if name:
                    observed[name] = True
        if metric_names is None:
            chosen = sorted(observed.keys())
        else:
            chosen = [str(name) for name in metric_names if name in observed]

        latest_values: dict[str, dict[str, object]] = {}
        for run in runs:
            run_id = run.get("id")
            latest_values[run_id] = {}
            for metric in run.get("metrics", []):
                name = metric.get("name")
                if name in chosen:
                    latest_values[run_id][name] = metric.get("value")

        rows: list[dict[str, object]] = []
        for run in runs:
            run_id = run.get("id")
            row: dict[str, object] = {
                "run_id": run_id,
                "run_name": run.get("name"),
                "status": run.get("status"),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
                "finished_at": run.get("finished_at"),
                "error": run.get("error"),
                "metric_count": len(run.get("metrics", [])),
                "artifact_count": len(run.get("artifacts", [])),
                "tag_count": len(run.get("tags", {})),
                "note_count": len(run.get("notes", [])),
                "params": json.dumps(run.get("params", {}), default=str),
            }
            for metric_name in chosen:
                value = latest_values[run_id].get(metric_name)
                row[metric_name] = value if value is not None else ""
            rows.append(row)
        return rows



    def export_run_notes_csv(
        self,
        run_id: str,
        destination: str,
    ) -> str:
        """Write a run's notes to ``destination`` as a CSV file.

        The CSV carries one row per note with ``run_id``, ``body``,
        ``created_at``, ``author``. The run must exist (KeyError
        otherwise). The destination's parent directory is created as
        needed. Returns the absolute path to the written CSV.
        """
        run = self.load_run(run_id)
        if run is None:
            raise KeyError(f"run not found: {run_id}")
        target = Path(destination)
        if target.parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run_id", "body", "created_at", "author"]
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for note in run.get("notes", []):
                if not isinstance(note, dict):
                    continue
                writer.writerow({
                    "run_id": run_id,
                    "body": str(note.get("body", "")),
                    "created_at": str(note.get("created_at", "")),
                    "author": str(note.get("author", "")),
                })
        return str(target)


    def search_runs_sorted(
        self,
        *,
        experiment_id: Optional[str] = None,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        descending: bool = True,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """List runs filtered by ``experiment_id`` and ``status``, sorted.

        ``sort_by`` accepts any field present on the run record; unknown
        fields fall back to ``created_at``. ``descending=True`` puts the
        newest runs first. ``limit`` truncates the result without
        raising. Returns the run records (one dict per run).
        """
        candidates: list[dict] = []
        if experiment_id is not None:
            candidates = list(self.list_runs(experiment_id))
        else:
            seen: set[str] = set()
            for exp in self.list_experiments(include_archived=True):
                for run in self.list_runs(exp.get("id")):
                    if run.get("id") in seen:
                        continue
                    seen.add(run.get("id"))
                    candidates.append(run)
        if status is not None:
            wanted = str(status)
            candidates = [run for run in candidates if run.get("status") == wanted]
        valid_sort_keys = {"created_at", "updated_at", "finished_at", "name", "status"}
        if sort_by not in valid_sort_keys:
            sort_by = "created_at"
        candidates.sort(
            key=lambda run: str(run.get(sort_by) or ""),
            reverse=bool(descending),
        )
        if limit is not None and limit >= 0:
            return candidates[: int(limit)]
        return candidates


    def experiment_timeline(
        self,
        exp_id: str,
    ) -> List[dict]:
        """Return the merged lifecycle events of every run in an experiment.

        Each entry is a dict with ``run_id``, ``run_name``, ``event``
        (one of "created", "updated", "finished", "metric", "artifact",
        "note", "tag"), ``timestamp``, and ``detail``. The list is
        sorted by ``timestamp`` ascending so callers can render it as a
        waterfall chart or feed. Unknown experiments raise ``KeyError``.
        """
        if self.load_experiment(exp_id) is None:
            raise KeyError(f"experiment not found: {exp_id}")

        events: list[dict[str, object]] = []
        for run in self.list_runs(exp_id):
            run_id = run.get("id")
            run_name = run.get("name")
            for event_name, timestamp in (
                ("created", run.get("created_at")),
                ("updated", run.get("updated_at")),
                ("finished", run.get("finished_at")),
            ):
                if not timestamp:
                    continue
                events.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "event": event_name,
                    "timestamp": str(timestamp),
                    "detail": "" if event_name != "finished" else str(run.get("error") or ""),
                })
            for metric in run.get("metrics", []) or []:
                events.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "event": "metric",
                    "timestamp": str(metric.get("timestamp") or run.get("created_at") or ""),
                    "detail": f"{metric.get('name')}={metric.get('value')}",
                })
            for artifact in run.get("artifacts", []) or []:
                events.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "event": "artifact",
                    "timestamp": str(artifact.get("created_at") or run.get("created_at") or ""),
                    "detail": str(artifact.get("name") or artifact.get("path") or ""),
                })
            for note in run.get("notes", []) or []:
                events.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "event": "note",
                    "timestamp": str(note.get("created_at") or ""),
                    "detail": str(note.get("body") or "")[:120],
                })
            for tag_name, tag_value in (run.get("tags") or {}).items():
                events.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "event": "tag",
                    "timestamp": str(run.get("updated_at") or run.get("created_at") or ""),
                    "detail": f"{tag_name}={tag_value}",
                })
        events.sort(key=lambda event: (str(event.get("timestamp") or ""), str(event.get("event") or "")))
        return events


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
