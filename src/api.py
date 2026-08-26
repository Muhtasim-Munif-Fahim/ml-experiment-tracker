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
def list_experiments(limit: int = 100, offset: int = 0, include_archived: bool = False):
    exps = storage.list_experiments(include_archived=include_archived)
    return exps[offset:offset+limit]


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
def experiment_artifacts(exp_id: str, limit: int = 50, offset: int = 0):
    """Page through artifacts recorded across every run of an experiment."""
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    try:
        inventory = storage.experiment_artifacts(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
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


@app.get("/experiments/{exp_id}/runs/", response_model=List[dict])
def list_runs(
    exp_id: str,
    limit: int = 50,
    status: Optional[str] = None,
    metric: Optional[str] = None,
    min_metric: Optional[float] = None,
    max_metric: Optional[float] = None,
):
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
    return runs[:limit]


@app.get("/runs/search", response_model=dict)
def search_runs(
    status: Optional[str] = None,
    name_contains: Optional[str] = None,
    metric: Optional[str] = None,
    min_metric: Optional[float] = None,
    max_metric: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Find runs across every experiment, newest first, paginated."""
    try:
        return storage.search_runs(
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
def list_artifacts(run_id: str):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_data.get("artifacts", [])


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
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


def artifact_staging_dir() -> str:
    import tempfile
    return tempfile.gettempdir()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
