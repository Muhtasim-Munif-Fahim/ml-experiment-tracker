"""Storage backends for experiment artifacts and metadata."""

from __future__ import annotations

import os
import shutil
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, List, Optional

from .models import Run, Experiment, Artifact


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

    def save_artifact(self, artifact_id: str, file_path: Path) -> str:
        dest = self.artifacts_dir / artifact_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        return str(dest)

    def load_artifact(self, artifact_path: str) -> bytes:
        with open(artifact_path, "rb") as f:
            return f.read()


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