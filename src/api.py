"""REST API for ML Experiment Tracker"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Form, Request, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .models import (
    AlertRule,
    Experiment,
    Run,
    RunStatus,
    Param,
    Metric,
    Artifact,
    ArtifactType,
    lttb_downsample,
    parse_run_status,
    validate_status_transition,
)
from .storage import StorageFactory, LocalStorageBackend


class ExperimentCreate(BaseModel):
    name: str
    description: str = ""
    tags: List[str] = []


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class RunCreate(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)


class RunUpdate(BaseModel):
    status: Optional[str] = None
    error: Optional[str] = None


class RunCompareRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str


class RunDuplicateRequest(BaseModel):
    name: Optional[str] = None
    include_metrics: bool = False


class RunMoveRequest(BaseModel):
    target_experiment_id: str


class SweepCreate(BaseModel):
    name_template: str = 'sweep-{index}'
    param_grid: Dict[str, List[Any]]
    base_params: Dict[str, Any] = Field(default_factory=dict)
    base_tags: Dict[str, str] = Field(default_factory=dict)
    base_name: Optional[str] = None


class RunImportRequest(BaseModel):
    runs: List[Dict[str, Any]] = Field(min_length=1)


class ParamCreate(BaseModel):
    name: str
    value: Any


class MetricCreate(BaseModel):
    name: str
    value: float
    step: Optional[int] = None


class MetricBatchCreate(BaseModel):
    metrics: Dict[str, float] = Field(min_length=1)
    step: Optional[int] = None


class AlertRuleCreate(BaseModel):
    metric_name: str
    comparator: str
    threshold: float


class NoteCreate(BaseModel):
    body: str


class TagValue(BaseModel):
    value: str


class ArtifactUpdate(BaseModel):
    name: Optional[str] = None
    artifact_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


app = FastAPI(title="ML Experiment Tracker API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


storage = StorageFactory.create("local", base_path="./mlruns")
API_TOKEN_ENV_VAR = "MLTRACKER_API_TOKEN"


@app.middleware("http")
async def require_api_token(request: Request, call_next):
    """Reject requests lacking the bearer token when one is configured."""
    expected = os.environ.get(API_TOKEN_ENV_VAR)
    if not expected or request.url.path == "/health":
        return await call_next(request)
    if request.headers.get("authorization") != f"Bearer {expected}":
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid API token"},
        )
    return await call_next(request)


def render_csv(columns: List[str], rows: List[List[Any]]) -> str:
    """Serialize table rows to CSV text using RFC 4180 quoting."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


# Experiment endpoints
@app.post("/experiments/", response_model=dict)
def create_experiment(exp: ExperimentCreate):
    experiment = Experiment(name=exp.name, description=exp.description, tags=exp.tags)
    storage.save_experiment(experiment.to_dict())
    return experiment.to_dict()


@app.get("/experiments/", response_model=List[dict])
def list_experiments(
    limit: int = 100,
    offset: int = 0,
    include_archived: bool = False,
    response: Response = None,
):
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    exps = storage.list_experiments(include_archived=include_archived)
    response.headers["X-Total-Count"] = str(len(exps))
    return exps[offset:offset + limit]


@app.get("/experiments/{exp_id}", response_model=dict)
def get_experiment(exp_id: str):
    exp = storage.load_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@app.get("/experiments/{exp_id}/summary", response_model=dict)
def experiment_summary(exp_id: str):
    exp_data = storage.load_experiment(exp_id)
    if not exp_data:
        raise HTTPException(status_code=404, detail="Experiment not found")
    exp_data = dict(exp_data)
    exp_data["runs"] = storage.list_runs(exp_id)
    return Experiment.from_dict(exp_data).summary()
@app.get("/experiments/{exp_id}/artifacts", response_model=dict)
def experiment_artifacts(
    exp_id: str, limit: int = 50, offset: int = 0, response: Response = None
):
    """Page through artifacts recorded across every run of an experiment."""
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    try:
        inventory = storage.experiment_artifacts(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    response.headers["X-Total-Count"] = str(len(inventory))
    page = inventory[offset : offset + limit]
    return {
        "total": len(inventory),
        "artifacts": [
            {
                "run_id": entry.get("run_id"),
                "run_name": entry.get("run_name"),
                "artifact_id": entry.get("artifact_id"),
                "name": entry.get("name"),
                "type": entry.get("type"),
                "size_bytes": entry.get("size_bytes", entry.get("size")),
                "sha256_prefix": str(entry.get("checksum_sha256") or "")[:12],
                "created_at": entry.get("created_at"),
            }
            for entry in page
        ],
    }


@app.get("/experiments/{exp_id}/artifacts.zip")
def download_experiment_artifacts_zip(exp_id: str):
    """Bundle every stored artifact of an experiment into a downloadable zip archive."""
    archive_path = (
        Path(artifact_staging_dir())
        / f"experiment_{exp_id}_artifacts_{uuid.uuid4().hex[:8]}.zip"
    )
    try:
        written = storage.export_experiment_artifacts_zip(exp_id, archive_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    experiment = storage.load_experiment(exp_id)
    filename = f"experiment-{experiment.get('name', exp_id)}-artifacts.zip"
    return FileResponse(
        written,
        media_type="application/zip",
        filename=filename,
    )


@app.get("/experiments/{exp_id}/pivot.csv")
def experiment_metric_pivot(exp_id: str, metric_names: Optional[str] = None):
    """Return a wide-format CSV of the latest metric values for every run."""
    try:
        requested = metric_names.split(",") if metric_names else None
        table = storage.experiment_metric_pivot(exp_id, requested)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    return Response(
        content=render_csv(table["columns"], table["rows"]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=experiment-{exp_id}-pivot.csv"
        },
    )


@app.get("/experiments/{exp_id}/metrics.history.csv")
def experiment_metric_history(
    exp_id: str,
    metric_names: Optional[str] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
):
    """Return the full long-form metric time-series for every run of an experiment.

    Each row carries ``run_id``, ``run_name``, ``metric_name``, ``step``,
    ``value`` and ``timestamp`` so plotting libraries can render training
    curves across runs. ``metric_names`` accepts a comma-separated list of
    metric names to include; ``start_step`` / ``end_step`` clip the inclusive
    step range. The CSV stays RFC-4180 compatible (CRLF line endings, every
    field quoted) so it round-trips through pandas without surprises.
    """
    try:
        requested = metric_names.split(",") if metric_names else None
        rows = storage.experiment_metric_long(
            exp_id,
            metric_names=requested,
            start_step=start_step,
            end_step=end_step,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    columns = ["run_id", "run_name", "metric_name", "step", "value", "timestamp"]
    row_lists = [
        [row.get(col) for col in columns]
        for row in rows
    ]
    return Response(
        content=render_csv(columns, row_lists),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment; filename=experiment-{exp_id}-metrics-history.csv"
            )
        },
    )


@app.get("/runs/{run_id}/notes.csv")
def export_run_notes(run_id: str, destination: Optional[str] = None):
    """Export one run's notes as a CSV file."""
    from fastapi.responses import FileResponse
    import tempfile
    if destination is None:
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False, dir=tempfile.gettempdir()
        )
        destination = tmp.name
        tmp.close()
    try:
        written_path = storage.export_run_notes_csv(run_id, destination)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    return FileResponse(
        written_path,
        media_type="text/csv",
        filename=f"run-{run_id}-notes.csv",
    )




@app.get("/experiments/{exp_id}/timeline.json")
def experiment_timeline(exp_id: str):
    """Sorted lifecycle events of every run in an experiment."""
    try:
        events = storage.experiment_timeline(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    return {"events": events, "count": len(events)}




@app.get("/experiments/{exp_id}/snapshot.csv")
def experiment_snapshot(
    exp_id: str,
    metric_names: Optional[str] = None,
):
    """Wide-form snapshot of every run: status, counts, latest metric values."""
    try:
        requested = metric_names.split(",") if metric_names else None
        rows = storage.experiment_snapshot(exp_id, metric_names=requested)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    if not rows:
        columns = ["run_id", "run_name", "status", "metric_count"]
    else:
        columns = [
            "run_id", "run_name", "status", "created_at", "updated_at",
            "finished_at", "error", "metric_count", "artifact_count",
            "tag_count", "note_count", "params",
        ]
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
    row_lists = [[row.get(col, "") for col in columns] for row in rows]
    return Response(
        content=render_csv(columns, row_lists),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=experiment-{exp_id}-snapshot.csv"
        },
    )





@app.get("/search/runs")
def search_runs_endpoint(
    experiment_id: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = "created_at",
    descending: bool = True,
    limit: Optional[int] = None,
):
    """List runs with optional filters and a multi-field sort."""
    runs = storage.search_runs_sorted(
        experiment_id=experiment_id,
        status=status,
        sort_by=sort_by,
        descending=descending,
        limit=limit,
    )
    return {
        "runs": runs,
        "count": len(runs),
        "total": len(runs),
    }




@app.patch("/experiments/{exp_id}", response_model=dict)
def update_experiment(exp_id: str, updates: ExperimentUpdate):
    exp = storage.load_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    for key, value in updates.model_dump(exclude_unset=True).items():
        exp[key] = value
    exp["updated_at"] = datetime.utcnow().isoformat()
    storage.save_experiment(exp)
    return exp


@app.post("/experiments/{exp_id}/archive", response_model=dict)
def archive_experiment(exp_id: str):
    """Soft-archive an experiment so default listings skip it."""
    experiment = storage.set_experiment_archived(exp_id, archived=True)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.post("/experiments/{exp_id}/unarchive", response_model=dict)
def unarchive_experiment(exp_id: str):
    experiment = storage.set_experiment_archived(exp_id, archived=False)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.delete("/experiments/{exp_id}")
def delete_experiment(exp_id: str):
    raise HTTPException(status_code=501, detail="Not implemented")


# Run endpoints
@app.post("/experiments/{exp_id}/runs/", response_model=dict)
def create_run(exp_id: str, run: RunCreate):
    exp = storage.load_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    run = Run(experiment_id=exp_id, name=run.name, params=run.params, tags=run.tags)
    storage.save_run(run.to_dict())
    return run.to_dict()


@app.post("/experiments/{exp_id}/runs/sweep", response_model=List[dict])
def create_sweep_runs(exp_id: str, sweep: SweepCreate):
    """Create multiple runs from a parameter grid (grid search)."""
    exp = storage.load_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    import itertools
    
    # Generate all combinations from the parameter grid
    param_names = list(sweep.param_grid.keys())
    param_values = list(sweep.param_grid.values())
    
    if not param_names:
        raise HTTPException(status_code=400, detail="param_grid must not be empty")
    
    combinations = list(itertools.product(*param_values))
    if len(combinations) > 1000:
        raise HTTPException(status_code=400, detail="Too many combinations (max 1000)")
    
    created_runs = []
    for index, combo in enumerate(combinations):
        run_params = dict(sweep.base_params)
        for name, value in zip(param_names, combo):
            run_params[name] = value
        
        run_tags = dict(sweep.base_tags)
        run_tags['sweep_index'] = str(index)
        
        run_name = sweep.base_name or sweep.name_template.format(index=index)
        
        run = Run(
            experiment_id=exp_id,
            name=run_name,
            params=run_params,
            tags=run_tags,
        )
        storage.save_run(run.to_dict())
        created_runs.append(run.to_dict())
    
    return created_runs


@app.post("/experiments/{exp_id}/runs/import", response_model=List[dict])
def import_runs(exp_id: str, request: RunImportRequest):
    """Create several runs from explicit specs in one request."""
    try:
        return storage.import_runs(exp_id, request.runs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/experiments/{exp_id}/runs/", response_model=List[dict])
def list_runs(
    exp_id: str,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    metric: Optional[str] = None,
    min_metric: Optional[float] = None,
    max_metric: Optional[float] = None,
    response: Response = None,
):
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    try:
        runs = storage.query_runs(
            exp_id,
            statuses=[status] if status else None,
            metric_name=metric,
            min_metric=min_metric,
            max_metric=max_metric,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.headers["X-Total-Count"] = str(len(runs))
    return runs[offset:offset + limit]


@app.get("/runs/search", response_model=dict)
def search_runs(
    status: Optional[str] = None,
    name_contains: Optional[str] = None,
    metric: Optional[str] = None,
    min_metric: Optional[float] = None,
    max_metric: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
    response: Response = None,
):
    """Find runs across every experiment, newest first, paginated."""
    try:
        result = storage.search_runs(
            statuses=[status] if status else None,
            name_contains=name_contains,
            metric_name=metric,
            min_metric=min_metric,
            max_metric=max_metric,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.headers["X-Total-Count"] = str(result["total"])
    return result


@app.get("/runs/search.csv")
def search_runs_csv(
    status: Optional[str] = None,
    name_contains: Optional[str] = None,
    metric: Optional[str] = None,
    min_metric: Optional[float] = None,
    max_metric: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Download one cross-experiment search page as CSV with fixed columns."""
    try:
        result = storage.search_runs(
            statuses=[status] if status else None,
            name_contains=name_contains,
            metric_name=metric,
            min_metric=min_metric,
            max_metric=max_metric,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    columns = ["experiment_id", "run_id", "name", "status", "created_at", "updated_at"]
    fields = ["experiment_id", "id", "name", "status", "created_at", "updated_at"]
    rows = [[run.get(field) for field in fields] for run in result["runs"]]
    return Response(
        content=render_csv(columns, rows),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="run-search.csv"'},
    )


@app.get("/runs/{run_id}", response_model=dict)
def get_run(run_id: str):
    run = storage.load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.patch("/runs/{run_id}", response_model=dict)
def update_run(run_id: str, updates: RunUpdate):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    changes = updates.model_dump(exclude_unset=True)
    if "status" in changes:
        try:
            current_status = parse_run_status(run_data.get("status", RunStatus.RUNNING.value))
            target_status = parse_run_status(changes["status"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            validate_status_transition(current_status, target_status)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        changes["status"] = target_status.value
        if (
            target_status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.ABORTED)
            and not run_data.get("finished_at")
        ):
            changes["finished_at"] = datetime.utcnow().isoformat()
    for key, value in changes.items():
        run_data[key] = value
    run_data["updated_at"] = datetime.utcnow().isoformat()
    storage.save_run(run_data)
    return run_data


@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    if not storage.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"message": "Run deleted"}


@app.post("/runs/{run_id}/duplicate", response_model=dict)
def duplicate_run(run_id: str, request: RunDuplicateRequest):
    """Copy a run into a fresh record with a new id, optionally with metrics."""
    try:
        return storage.duplicate_run(
            run_id, name=request.name, include_metrics=request.include_metrics
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.post("/runs/{run_id}/move", response_model=dict)
def move_run(run_id: str, request: RunMoveRequest):
    """Move a run from its current experiment into another experiment.

    The source experiment drops the run id and the destination experiment
    gains it; the run record itself is rewritten with the new
    ``experiment_id``. Returns the updated run record so the caller can
    confirm the move.
    """
    try:
        return storage.move_run(run_id, request.target_experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/runs/{run_id}/tags", response_model=dict)
def list_run_tags(run_id: str):
    """Return the key/value tags attached to a run."""
    try:
        return storage.list_run_tags(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.put("/runs/{run_id}/tags/{tag_name}", response_model=dict)
def set_run_tag(run_id: str, tag_name: str, body: TagValue):
    """Set or replace one key/value tag on a run."""
    try:
        return storage.set_run_tag(run_id, tag_name, body.value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/runs/{run_id}/tags/{tag_name}")
def delete_run_tag(run_id: str, tag_name: str):
    try:
        deleted = storage.delete_run_tag(run_id, tag_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"message": "Tag deleted"}


@app.post("/runs/{run_id}/params", response_model=dict)
def log_param(run_id: str, param: ParamCreate):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    run_data.setdefault("params", {})[param.name] = param.value
    storage.save_run(run_data)
    return {"message": "Parameter logged"}


@app.post("/runs/{run_id}/metrics", response_model=dict)
def log_metric(run_id: str, metric: MetricCreate):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    point = metric.model_dump()
    run_data.setdefault("metrics", []).append(point)
    storage.save_run(run_data)
    alerts = storage.apply_alert_rules(run_data, [point])
    return {"message": "Metric logged", "alerts": alerts}


@app.post("/runs/{run_id}/metrics/batch", response_model=dict)
def log_metrics(run_id: str, batch: MetricBatchCreate):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    timestamp = datetime.utcnow().isoformat()
    points = [
        {
            "name": name,
            "value": value,
            "step": batch.step,
            "timestamp": timestamp,
        }
        for name, value in batch.metrics.items()
    ]
    run_data.setdefault("metrics", []).extend(points)
    storage.save_run(run_data)
    alerts = storage.apply_alert_rules(run_data, points)
    return {"message": "Metrics logged", "count": len(batch.metrics), "alerts": alerts}


@app.get("/runs/{run_id}/metrics/{metric_name}", response_model=List[dict])
def metric_history(run_id: str, metric_name: str):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    return [
        metric
        for metric in run_data.get("metrics", [])
        if metric.get("name") == metric_name
    ]


@app.get("/runs/{run_id}/metrics/{metric_name}/downsample", response_model=List[dict])
def downsample_metric_history(
    run_id: str,
    metric_name: str,
    points: int = Query(..., ge=2),
):
    """Return a long metric series reduced to ``points`` shape-preserving samples."""
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    series = [
        metric
        for metric in run_data.get("metrics", [])
        if metric.get("name") == metric_name
    ]
    return lttb_downsample(series, points)


@app.get("/runs/{run_id}/metrics.csv")
def run_metrics_csv(run_id: str):
    """Export every recorded metric point of a run as a flat CSV."""
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    rows = [
        [
            metric.get("name"),
            "" if metric.get("step") is None else metric.get("step"),
            metric.get("value"),
            metric.get("timestamp", ""),
        ]
        for metric in run_data.get("metrics", [])
    ]
    return Response(
        content=render_csv(["name", "step", "value", "timestamp"], rows),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="run-metrics.csv"'},
    )


@app.post("/experiments/{exp_id}/alert-rules", response_model=dict)
def create_alert_rule(exp_id: str, rule: AlertRuleCreate):
    """Register a metric threshold rule evaluated as metrics are logged."""
    if not storage.load_experiment(exp_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    try:
        alert_rule = AlertRule(
            metric_name=rule.metric_name,
            comparator=rule.comparator,
            threshold=rule.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return storage.save_alert_rule(exp_id, alert_rule.to_dict())


@app.get("/experiments/{exp_id}/alert-rules", response_model=List[dict])
def list_alert_rules(exp_id: str):
    try:
        return storage.list_alert_rules(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc


@app.delete("/experiments/{exp_id}/alert-rules/{rule_id}")
def delete_alert_rule(exp_id: str, rule_id: str):
    try:
        deleted = storage.delete_alert_rule(exp_id, rule_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return {"message": "Alert rule deleted"}


@app.get("/runs/{run_id}/alerts", response_model=List[dict])
def run_alerts(run_id: str):
    """Return alerts recorded on a run when logged metrics breached thresholds."""
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_data.get("alerts", [])


@app.get("/experiments/{exp_id}/leaderboard", response_model=List[dict])
def run_leaderboard(
    exp_id: str,
    metric: str = Query(...),
    maximize: bool = True,
    limit: int = Query(10, ge=1),
):
    """Rank runs of an experiment by the latest value of one metric."""
    try:
        return storage.run_leaderboard(
            exp_id, metric, maximize=maximize, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/experiments/{exp_id}/leaderboard.csv")
def run_leaderboard_csv(
    exp_id: str,
    metric: str = Query(...),
    maximize: bool = True,
    limit: int = Query(10, ge=1),
):
    """Download the metric leaderboard as CSV with one-based rank numbers."""
    try:
        ranked = storage.run_leaderboard(
            exp_id, metric, maximize=maximize, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rows = [
        [rank, entry["run_id"], entry["name"], entry["value"], entry["step"]]
        for rank, entry in enumerate(ranked, start=1)
    ]
    return Response(
        content=render_csv(["rank", "run_id", "name", "value", "step"], rows),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="leaderboard.csv"'},
    )


@app.post("/experiments/{exp_id}/runs/compare", response_model=dict)
def compare_runs_endpoint(exp_id: str, request: RunCompareRequest):
    """Compare parameters and latest metric values between two runs."""
    exp_data = storage.load_experiment(exp_id)
    if not exp_data:
        raise HTTPException(status_code=404, detail="Experiment not found")
    exp = Experiment.from_dict(exp_data)
    try:
        return exp.compare_runs(request.baseline_run_id, request.candidate_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/runs/{run_id}/artifacts", response_model=dict)
def upload_artifact(run_id: str, name: str = Form(...), artifact_type: str = Form(...), file: UploadFile = File(...), metadata: str = Form("{}")):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    content = file.file.read()
    tmp_path = Path(artifact_staging_dir()) / f"artifact_{name}_{uuid.uuid4().hex[:8]}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(content)
    artifact_id = f"artifact_{uuid.uuid4().hex[:8]}"
    artifact_path = storage.save_artifact(artifact_id, tmp_path)
    artifact = Artifact(
        name=name,
        artifact_type=ArtifactType(artifact_type),
        path=artifact_path,
        size_bytes=len(content),
        checksum_sha256=storage.artifact_checksum(artifact_path),
        metadata=json.loads(metadata)
    )
    entry = artifact.to_dict()
    entry["artifact_id"] = artifact_id
    run_data.setdefault("artifacts", []).append(entry)
    storage.save_run(run_data)
    return {
        "artifact_id": artifact_id,
        "path": artifact_path,
        "checksum_sha256": artifact.checksum_sha256,
    }


@app.get("/runs/{run_id}/artifacts/", response_model=List[dict])
def list_artifacts(
    run_id: str,
    limit: int = 50,
    offset: int = 0,
    response: Response = None,
):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    artifacts = run_data.get("artifacts", [])
    response.headers["X-Total-Count"] = str(len(artifacts))
    return artifacts[offset:offset + limit]


@app.get("/runs/{run_id}/artifacts/{artifact_ref}")
def download_artifact(run_id: str, artifact_ref: str):
    """Serve stored artifact bytes by artifact id (or name) with its content type."""
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    entry = next(
        (
            artifact
            for artifact in run_data.get("artifacts", [])
            if artifact.get("artifact_id") == artifact_ref
            or artifact.get("name") == artifact_ref
        ),
        None,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    file_path = Path(entry["path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    filename = entry.get("name") or file_path.name
    headers = {}
    if entry.get("checksum_sha256"):
        headers["X-Artifact-Sha256"] = entry["checksum_sha256"]
    return FileResponse(
        file_path,
        media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
        filename=filename,
        headers=headers,
    )


@app.patch("/runs/{run_id}/artifacts/{artifact_ref}", response_model=dict)
def update_artifact(run_id: str, artifact_ref: str, updates: ArtifactUpdate):
    """Update an artifact's name, type, or metadata without re-uploading bytes."""
    try:
        updated = storage.update_artifact(
            run_id, artifact_ref, updates.model_dump(exclude_unset=True)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return updated


@app.get("/runs/{run_id}/artifacts.zip")
def download_run_artifacts_zip(run_id: str):
    """Bundle every stored artifact of a run into a downloadable zip archive."""
    archive_path = (
        Path(artifact_staging_dir())
        / f"run_{run_id}_artifacts_{uuid.uuid4().hex[:8]}.zip"
    )
    try:
        written = storage.export_run_artifacts_zip(run_id, archive_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    return FileResponse(
        written,
        media_type="application/zip",
        filename=f"run-{run_id}-artifacts.zip",
    )


@app.post("/runs/{run_id}/notes", response_model=dict)
def create_note(run_id: str, note: NoteCreate):
    """Attach a freeform markdown note to a run."""
    try:
        return storage.add_note(run_id, note.body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/runs/{run_id}/notes", response_model=List[dict])
def list_notes(run_id: str):
    try:
        return storage.list_notes(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.put("/runs/{run_id}/notes/{note_id}", response_model=dict)
def update_note(run_id: str, note_id: str, note: NoteCreate):
    try:
        updated = storage.update_note(run_id, note_id, note.body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return updated


@app.delete("/runs/{run_id}/notes/{note_id}")
def delete_note(run_id: str, note_id: str):
    try:
        deleted = storage.delete_note(run_id, note_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"message": "Note deleted"}


@app.get("/runs/{run_id}/snapshot", response_model=dict)
def run_snapshot(run_id: str):
    """Return a self-contained JSON snapshot of one run."""
    try:
        return storage.build_run_snapshot(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


# Health check
@app.get("/health")
def health_check():
    """Report liveness plus current storage totals for readiness checks."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "storage": storage.storage_stats(),
    }


def artifact_staging_dir() -> str:
    import tempfile
    return tempfile.gettempdir()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
