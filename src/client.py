"""CLI Client for ML Experiment Tracker"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class PaginatedList(list):
    """A list page that also carries the server-reported total count.

    Behaves exactly like a plain list so existing callers keep working,
    while ``total`` exposes how many records match the query in full.
    """

    def __init__(self, items: List[dict], total: Optional[int] = None):
        super().__init__(items)
        self.total = int(total) if total is not None else len(items)


class ExperimentTrackerClient:
    """Client for interacting with ML Experiment Tracker server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or not 0 <= max_retries <= 10
        ):
            raise ValueError("max_retries must be an integer between 0 and 10")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _retry_delay(self, response: "requests.Response", attempt: int) -> float:
        """Seconds to wait before the next attempt, from Retry-After or backoff."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
            try:
                parsed = parsedate_to_datetime(retry_after)
                delta = (parsed - datetime.now(timezone.utc)).total_seconds()
                return max(0.0, delta)
            except (TypeError, ValueError, OverflowError):
                pass
        return self.backoff_factor * (2 ** attempt)

    def _request_raw(self, method: str, endpoint: str, **kwargs) -> "requests.Response":
        """Send a request, retrying transient 5xx responses with capped backoff."""
        url = f"{self.base_url}{endpoint}"
        attempt = 0
        while True:
            response = self.session.request(method, url, **kwargs)
            if response.status_code < 500 or attempt >= self.max_retries:
                return response
            time.sleep(self._retry_delay(response, attempt))
            attempt += 1

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        response = self._request_raw(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()

    def _request_page(self, method: str, endpoint: str, **kwargs) -> PaginatedList:
        response = self._request_raw(method, endpoint, **kwargs)
        response.raise_for_status()
        return PaginatedList(response.json(), total=response.headers.get("X-Total-Count"))

    def health(self) -> dict:
        """Fetch liveness status plus current storage statistics."""
        return self._request("GET", "/health")

    # Experiment methods
    def create_experiment(self, name: str, description: str = "", tags: List[str] = None) -> dict:
        data = {"name": name, "description": description, "tags": tags or []}
        return self._request("POST", "/experiments/", json=data)

    def get_experiment(self, exp_id: str) -> dict:
        return self._request("GET", f"/experiments/{exp_id}")

    def experiment_summary(self, exp_id: str) -> dict:
        """Fetch lifecycle and aggregate metric totals for an experiment."""

        return self._request("GET", f"/experiments/{exp_id}/summary")

    def list_experiments(
        self, limit: int = 100, offset: int = 0, include_archived: bool = False
    ) -> PaginatedList:
        params = {"limit": limit, "offset": offset, "include_archived": include_archived}
        return self._request_page("GET", "/experiments/", params=params)

    def experiment_artifacts(self, exp_id: str, limit: int = 50, offset: int = 0) -> dict:
        """Page through artifacts recorded across an experiment's runs."""
        params = {"limit": limit, "offset": offset}
        return self._request("GET", f"/experiments/{exp_id}/artifacts", params=params)

    def archive_experiment(self, exp_id: str) -> dict:
        """Soft-archive an experiment so default listings skip it."""
        return self._request("POST", f"/experiments/{exp_id}/archive")

    def unarchive_experiment(self, exp_id: str) -> dict:
        """Restore a soft-archived experiment to active."""
        return self._request("POST", f"/experiments/{exp_id}/unarchive")

    def update_experiment(self, exp_id: str, updates: dict) -> dict:
        return self._request("PATCH", f"/experiments/{exp_id}", json=updates)

    def delete_experiment(self, exp_id: str) -> None:
        self._request("DELETE", f"/experiments/{exp_id}")

    # Run methods
    def create_run(self, exp_id: str, name: str, params: dict = None, tags: dict = None) -> dict:
        data = {"name": name, "params": params or {}, "tags": tags or {}}
        return self._request("POST", f"/experiments/{exp_id}/runs/", json=data)

    def create_sweep(
        self,
        exp_id: str,
        param_grid: Dict[str, List[Any]],
        base_params: Dict[str, Any] = None,
        base_tags: Dict[str, str] = None,
        name_template: str = "sweep-{index}",
        base_name: str = None,
    ) -> List[dict]:
        """Create multiple runs from a parameter grid (grid search)."""
        data = {
            "param_grid": param_grid,
            "base_params": base_params or {},
            "base_tags": base_tags or {},
            "name_template": name_template,
        }
        if base_name:
            data["base_name"] = base_name
        return self._request("POST", f"/experiments/{exp_id}/runs/sweep", json=data)

    def import_runs(self, exp_id: str, runs: List[dict]) -> List[dict]:
        """Create several runs from explicit specs in a single request."""
        return self._request(
            "POST", f"/experiments/{exp_id}/runs/import", json={"runs": runs}
        )

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/runs/{run_id}")

    def list_runs(
        self,
        exp_id: str,
        limit: int = 50,
        offset: int = 0,
        status: str = None,
        metric: str = None,
        min_metric: float = None,
        max_metric: float = None,
    ) -> PaginatedList:
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if metric:
            params["metric"] = metric
        if min_metric is not None:
            params["min_metric"] = min_metric
        if max_metric is not None:
            params["max_metric"] = max_metric
        return self._request_page("GET", f"/experiments/{exp_id}/runs/", params=params)

    def update_run(self, run_id: str, updates: dict) -> dict:
        return self._request("PATCH", f"/runs/{run_id}", json=updates)

    def set_run_status(self, run_id: str, status: str, error: Optional[str] = None) -> dict:
        """Move a run through the declared lifecycle; illegal moves fail with 409."""
        payload: Dict[str, Any] = {"status": status}
        if error is not None:
            payload["error"] = error
        return self._request("PATCH", f"/runs/{run_id}", json=payload)

    def search_runs(
        self,
        status: str = None,
        name_contains: str = None,
        metric: str = None,
        min_metric: float = None,
        max_metric: float = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Find runs across every experiment, newest first, paginated."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if name_contains:
            params["name_contains"] = name_contains
        if metric:
            params["metric"] = metric
        if min_metric is not None:
            params["min_metric"] = min_metric
        if max_metric is not None:
            params["max_metric"] = max_metric
        return self._request("GET", "/runs/search", params=params)

    def compare_runs(self, exp_id: str, baseline_run_id: str, candidate_run_id: str) -> dict:
        """Compare parameters and latest metric values between two runs."""
        return self._request("POST", f"/experiments/{exp_id}/runs/compare", json={
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id
        })

    def search_runs_csv(
        self,
        status: str = None,
        name_contains: str = None,
        metric: str = None,
        min_metric: float = None,
        max_metric: float = None,
        limit: int = 50,
        offset: int = 0,
        destination: Optional[str] = None,
    ) -> str:
        """Fetch one search page as CSV text, optionally saving it to a file."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if name_contains:
            params["name_contains"] = name_contains
        if metric:
            params["metric"] = metric
        if min_metric is not None:
            params["min_metric"] = min_metric
        if max_metric is not None:
            params["max_metric"] = max_metric
        response = self._request_raw("GET", "/runs/search.csv", params=params)
        response.raise_for_status()
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return response.text

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its metadata record."""

        self._request("DELETE", f"/runs/{run_id}")

    def duplicate_run(
        self, run_id: str, name: str = None, include_metrics: bool = False
    ) -> dict:
        """Copy a run into a fresh record, optionally including metric history."""
        payload = {"include_metrics": include_metrics}
        if name:
            payload["name"] = name
        return self._request("POST", f"/runs/{run_id}/duplicate", json=payload)

    def move_run(self, run_id: str, target_experiment_id: str) -> dict:
        """Move a run into a different experiment.

        Returns the rewritten run (with the new ``experiment_id``) so the
        caller can confirm the move succeeded.
        """
        return self._request(
            "POST",
            f"/runs/{run_id}/move",
            json={"target_experiment_id": target_experiment_id},
        )

    def list_run_tags(self, run_id: str) -> dict:
        """Fetch the key/value tags attached to a run."""
        return self._request("GET", f"/runs/{run_id}/tags")

    def set_run_tag(self, run_id: str, name: str, value: str) -> dict:
        """Set or replace one key/value tag on a run."""
        return self._request("PUT", f"/runs/{run_id}/tags/{name}", json={"value": value})

    def delete_run_tag(self, run_id: str, name: str) -> dict:
        """Remove one tag from a run; missing tags raise 404."""
        return self._request("DELETE", f"/runs/{run_id}/tags/{name}")

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

    def metric_history_csv(self, run_id: str, destination: Optional[str] = None) -> str:
        """Fetch a run's full metric history as CSV text, optionally saving it."""
        response = self._request_raw("GET", f"/runs/{run_id}/metrics.csv")
        response.raise_for_status()
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return response.text

    def downsample_metric(self, run_id: str, metric_name: str, points: int) -> List[dict]:
        """Fetch a metric series reduced to ``points`` shape-preserving samples."""
        return self._request(
            "GET",
            f"/runs/{run_id}/metrics/{metric_name}/downsample",
            params={"points": points},
        )

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

    def run_leaderboard_csv(
        self,
        exp_id: str,
        metric: str,
        maximize: bool = True,
        limit: int = 10,
        destination: Optional[str] = None,
    ) -> str:
        """Fetch the metric leaderboard as CSV text, optionally saving it."""
        response = self._request_raw(
            "GET",
            f"/experiments/{exp_id}/leaderboard.csv",
            params={"metric": metric, "maximize": maximize, "limit": limit},
        )
        response.raise_for_status()
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return response.text

    def upload_artifact(self, run_id: str, name: str, artifact_type: str, file_path: str, metadata: dict = None) -> dict:
        import mimetypes
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, mimetypes.guess_type(file_path)[0] or "application/octet-stream")}
            data = {"name": name, "artifact_type": artifact_type, "metadata": json.dumps(metadata or {})}
            return self._request("POST", f"/runs/{run_id}/artifacts", data=data, files=files)

    def list_artifacts(self, run_id: str, limit: int = 50, offset: int = 0) -> PaginatedList:
        return self._request_page(
            "GET",
            f"/runs/{run_id}/artifacts/",
            params={"limit": limit, "offset": offset},
        )

    def download_artifact(self, run_id: str, artifact_ref: str, destination: Optional[str] = None) -> bytes:
        """Download stored artifact bytes, optionally saving them to a path."""
        response = self._request_raw("GET", f"/runs/{run_id}/artifacts/{artifact_ref}")
        response.raise_for_status()
        if destination:
            Path(destination).write_bytes(response.content)
        return response.content

    def update_artifact(self, run_id: str, artifact_ref: str, updates: dict) -> dict:
        """Update an artifact's name, type, or metadata without re-uploading."""
        return self._request(
            "PATCH", f"/runs/{run_id}/artifacts/{artifact_ref}", json=updates
        )

    def download_run_artifacts_zip(
        self, run_id: str, destination: Optional[str] = None
    ) -> bytes:
        """Download every stored artifact of a run as one zip archive."""
        response = self._request_raw("GET", f"/runs/{run_id}/artifacts.zip")
        response.raise_for_status()
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return response.content

    def download_experiment_artifacts_zip(
        self, exp_id: str, destination: Optional[str] = None
    ) -> bytes:
        """Download every stored artifact of an experiment as one zip archive."""
        response = self._request_raw("GET", f"/experiments/{exp_id}/artifacts.zip")
        response.raise_for_status()
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return response.content

    def experiment_metric_pivot_csv(
        self,
        exp_id: str,
        metric_names: Optional[List[str]] = None,
        destination: Optional[str] = None,
    ) -> str:
        """Fetch a wide-format CSV of the latest metric values for every run."""
        params = {}
        if metric_names:
            params["metric_names"] = ",".join(metric_names)
        response = self._request_raw(
            "GET", f"/experiments/{exp_id}/pivot.csv", params=params
        )
        response.raise_for_status()
        text = response.text
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def experiment_metric_history_csv(
        self,
        exp_id: str,
        metric_names: Optional[List[str]] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        destination: Optional[str] = None,
    ) -> str:
        """Fetch the long-form metric time-series CSV for an experiment.

        Each row carries ``run_id``, ``run_name``, ``metric_name``, ``step``,
        ``value`` and ``timestamp``. ``metric_names`` restricts to a chosen
        subset and ``start_step`` / ``end_step`` clip the inclusive step range.
        Returns the CSV text and optionally writes it to ``destination``.
        """
        params: dict = {}
        if metric_names:
            params["metric_names"] = ",".join(metric_names)
        if start_step is not None:
            params["start_step"] = str(start_step)
        if end_step is not None:
            params["end_step"] = str(end_step)
        response = self._request_raw(
            "GET", f"/experiments/{exp_id}/metrics.history.csv", params=params
        )
        response.raise_for_status()
        text = response.text
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def experiment_snapshot_csv(
        self,
        exp_id: str,
        metric_names: Optional[List[str]] = None,
        destination: Optional[str] = None,
    ) -> str:
        """Fetch a wide-form per-run snapshot CSV for an experiment."""
        params: dict = {}
        if metric_names:
            params["metric_names"] = ",".join(metric_names)
        response = self._request_raw(
            "GET", f"/experiments/{exp_id}/snapshot.csv", params=params
        )
        response.raise_for_status()
        text = response.text
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text




    def export_run_notes_csv(
        self,
        run_id: str,
        destination: Optional[str] = None,
    ) -> str:
        """Fetch the CSV of a single run's notes."""
        response = self._request_raw(
            "GET", f"/runs/{run_id}/notes.csv"
        )
        response.raise_for_status()
        text = response.text
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def search_runs_sorted(
        self,
        *,
        experiment_id: Optional[str] = None,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        descending: bool = True,
        limit: Optional[int] = None,
    ) -> dict:
        """List runs with optional filters and a multi-field sort.

        The ``/search/runs`` endpoint returns ``{"runs": [...], "total": N, "count": N}``.
        ``experiment_id`` and ``status`` filter the result; ``sort_by`` and
        ``descending`` control the ordering; ``limit`` truncates the result.
        """
        params: dict = {
            "sort_by": sort_by,
            "descending": "true" if descending else "false",
        }
        if experiment_id is not None:
            params["experiment_id"] = experiment_id
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = str(int(limit))
        response = self._request_raw("GET", "/search/runs", params=params)
        response.raise_for_status()
        return response.json()

    def experiment_timeline(self, exp_id: str) -> dict:
        """Return the merged run-lifecycle events for an experiment."""
        response = self._request_raw(
            "GET", f"/experiments/{exp_id}/timeline.json"
        )
        response.raise_for_status()
        return response.json()

    def track(self, name: str, description: str = "", tags: List[str] = None, params: dict = None) -> ExperimentContext:
        """Open an experiment context that creates a run on entry."""
        return ExperimentContext(self, name=name, description=description, tags=tags, params=params)

    def create_alert_rule(self, exp_id: str, metric_name: str, comparator: str, threshold: float) -> dict:
        """Register a metric threshold rule evaluated as metrics are logged."""
        return self._request(
            "POST",
            f"/experiments/{exp_id}/alert-rules",
            json={"metric_name": metric_name, "comparator": comparator, "threshold": threshold},
        )

    def list_alert_rules(self, exp_id: str) -> List[dict]:
        return self._request("GET", f"/experiments/{exp_id}/alert-rules")

    def delete_alert_rule(self, exp_id: str, rule_id: str) -> dict:
        return self._request("DELETE", f"/experiments/{exp_id}/alert-rules/{rule_id}")

    def run_alerts(self, run_id: str) -> List[dict]:
        """Fetch alerts recorded when logged metrics breached thresholds."""
        return self._request("GET", f"/runs/{run_id}/alerts")

    def create_note(self, run_id: str, body: str) -> dict:
        """Attach a freeform markdown note to a run."""
        return self._request("POST", f"/runs/{run_id}/notes", json={"body": body})

    def list_notes(self, run_id: str) -> List[dict]:
        return self._request("GET", f"/runs/{run_id}/notes")

    def update_note(self, run_id: str, note_id: str, body: str) -> dict:
        return self._request("PUT", f"/runs/{run_id}/notes/{note_id}", json={"body": body})

    def delete_note(self, run_id: str, note_id: str) -> dict:
        return self._request("DELETE", f"/runs/{run_id}/notes/{note_id}")

    def run_snapshot(self, run_id: str) -> dict:
        """Fetch a self-contained JSON snapshot of one run."""
        return self._request("GET", f"/runs/{run_id}/snapshot")

    def export_run_snapshot(self, run_id: str, destination: str) -> str:
        """Save one run's snapshot to a local JSON file and return the path."""
        snapshot = self.run_snapshot(run_id)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(path)


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
