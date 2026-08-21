"""CLI Client for ML Experiment Tracker"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class ExperimentTrackerClient:
    """Client for interacting with ML Experiment Tracker server."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    # Experiment methods
    def create_experiment(self, name: str, description: str = "", tags: List[str] = None) -> dict:
        data = {"name": name, "description": description, "tags": tags or []}
        return self._request("POST", "/experiments/", json=data)

    def get_experiment(self, exp_id: str) -> dict:
        return self._request("GET", f"/experiments/{exp_id}")

    def experiment_summary(self, exp_id: str) -> dict:
        """Fetch lifecycle and aggregate metric totals for an experiment."""

        return self._request("GET", f"/experiments/{exp_id}/summary")

    def list_experiments(self, limit: int = 100, offset: int = 0) -> List[dict]:
        return self._request("GET", "/experiments/", params={"limit": limit, "offset": offset})

    def update_experiment(self, exp_id: str, updates: dict) -> dict:
        return self._request("PATCH", f"/experiments/{exp_id}", json=updates)

    def delete_experiment(self, exp_id: str) -> None:
        self._request("DELETE", f"/experiments/{exp_id}")

    # Run methods
    def create_run(self, exp_id: str, name: str, params: dict = None, tags: dict = None) -> dict:
        data = {"name": name, "params": params or {}, "tags": tags or {}}
        return self._request("POST", f"/experiments/{exp_id}/runs/", json=data)

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/runs/{run_id}")

    def list_runs(
        self,
        exp_id: str,
        limit: int = 50,
        status: str = None,
        metric: str = None,
        min_metric: float = None,
        max_metric: float = None,
    ) -> List[dict]:
        params = {"limit": limit}
        if status:
            params["status"] = status
        if metric:
            params["metric"] = metric
        if min_metric is not None:
            params["min_metric"] = min_metric
        if max_metric is not None:
            params["max_metric"] = max_metric
        return self._request("GET", f"/experiments/{exp_id}/runs/", params=params)

    def update_run(self, run_id: str, updates: dict) -> dict:
        return self._request("PATCH", f"/runs/{run_id}", json=updates)

    def log_param(self, run_id: str, name: str, value: Any) -> dict:
        return self._request("POST", f"/runs/{run_id}/params", json={"name": name, "value": value})

    def log_metric(self, run_id: str, name: str, value: float, step: int = None) -> dict:
        return self._request("POST", f"/runs/{run_id}/metrics", json={"name": name, "value": value, "step": step})

    def log_metrics(self, run_id: str, metrics: dict, step: int = None) -> dict:
        """Log several metrics in one request at a shared step."""

        return self._request(
            "POST",
            f"/runs/{run_id}/metrics/batch",
            json={"metrics": metrics, "step": step},
        )

    def metric_history(self, run_id: str, metric_name: str) -> List[dict]:
        """Fetch the recorded series for one run metric."""

        return self._request("GET", f"/runs/{run_id}/metrics/{metric_name}")

    def run_leaderboard(
        self,
        exp_id: str,
        metric: str,
        maximize: bool = True,
        limit: int = 10,
    ) -> List[dict]:
        """Rank runs of an experiment by the latest value of one metric."""

        return self._request(
            "GET",
            f"/experiments/{exp_id}/leaderboard",
            params={"metric": metric, "maximize": maximize, "limit": limit},
        )

    def upload_artifact(self, run_id: str, name: str, artifact_type: str, file_path: str, metadata: dict = None) -> dict:
        import mimetypes
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, mimetypes.guess_type(file_path)[0] or "application/octet-stream")}
            data = {"name": name, "artifact_type": artifact_type, "metadata": json.dumps(metadata or {})}
            return self._request("POST", f"/runs/{run_id}/artifacts", data=data, files=files)

    def list_artifacts(self, run_id: str) -> List[dict]:
        return self._request("GET", f"/runs/{run_id}/artifacts/")


class ExperimentContext:
    """Context manager for running an experiment."""

    def __init__(self, client: "ExperimentTrackerClient", name: str, description: str = "", tags: List[str] = None, params: dict = None):
        self.client = client
        self.exp_name = name
        self.exp_description = description
        self.exp_tags = tags or []
        self.params = params or {}
        self.experiment = None
        self.run = None

    def __enter__(self):
        exp = self.client.create_experiment(self.exp_name, self.exp_description, self.exp_tags)
        self.experiment = exp
        run = self.client.create_run(exp["id"], name=f"run-{datetime.utcnow().isoformat()}", params=self.params)
        self.run = run
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.client.update_run(self.run["id"], {"status": "failed", "error": str(exc_val)})
        else:
            self.client.update_run(self.run["id"], {"status": "completed"})
        return False

    def log_param(self, name: str, value: Any):
        self.client.log_param(self.run["id"], name, value)

    def log_metric(self, name: str, value: float, step: int = None):
        self.client.log_metric(self.run["id"], name, value, step)

    def log_metrics(self, metrics: dict, step: int = None):
        self.client.log_metrics(self.run["id"], metrics, step)

    def log_artifact(self, name: str, artifact_type: str, file_path: str, metadata: dict = None):
        self.client.upload_artifact(self.run["id"], name, artifact_type, file_path, metadata)

    def __getattr__(self, name):
        return getattr(self.client, name)


class ExperimentTrackerClient:
    def track(self, name: str, description: str = "", tags: List[str] = None, params: dict = None) -> ExperimentContext:
        return ExperimentContext(self, name=name, description=description, tags=tags, params=params)


def get_tracker(base_url: str = "http://localhost:8000") -> ExperimentTrackerClient:
    """Convenience function to create a tracker client."""
    return ExperimentTrackerClient(base_url=base_url)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "track":
        client = get_tracker()
        with client.track("my-experiment", params={"lr": 0.01}) as exp:
            exp.log_param("lr", 0.01)
            exp.log_metric("loss", 0.5, step=1)
            exp.log_metric("loss", 0.3, step=2)
            exp.log_metric("loss", 0.1, step=3)
    else:
        print("Usage: python -m ml_experiment_tracker.client track")
