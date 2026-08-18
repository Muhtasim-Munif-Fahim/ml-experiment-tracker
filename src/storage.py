"""Storage backends for experiment artifacts and metadata."""

from __future__ import annotations

import os
import shutil
import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, List, Optional

from .models import Run, Experiment, Artifact

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
    def list_experiments(self) -> List[dict]:
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

    def list_experiments(self) -> List[dict]:
        exps = []
        for path in self.experiments_dir.glob("*.json"):
            with open(path) as f:
                exps.append(json.load(f))
        return exps

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

    def query_runs(
        self,
        experiment_id: str,
        *,
        statuses: Optional[List[str]] = None,
        tags: Optional[dict[str, str]] = None,
        name_contains: Optional[str] = None,
    ) -> List[dict]:
        """Filter stored runs by status, exact tags, and a name fragment."""

        allowed_statuses = set(statuses or [])
        required_tags = tags or {}
        fragment = name_contains.casefold() if name_contains else None
        matches = []
        for run in self.list_runs(experiment_id):
            if allowed_statuses and run.get("status") not in allowed_statuses:
                continue
            run_tags = run.get("tags", {})
            if not isinstance(run_tags, dict) or any(
                run_tags.get(key) != value for key, value in required_tags.items()
            ):
                continue
            if fragment and fragment not in str(run.get("name", "")).casefold():
                continue
            matches.append(run)
        return sorted(
            matches,
            key=lambda run: str(run.get("created_at", "")),
            reverse=True,
        )

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

    def list_experiments(self) -> List[dict]:
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
