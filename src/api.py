"""REST API for ML Experiment Tracker"""

from __future__ import annotations

import json
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .models import AlertRule, Experiment, Run, RunStatus, Param, Metric, Artifact, ArtifactType
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


app = FastAPI(title="ML Experiment Tracker API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


storage = StorageFactory.create("local", base_path="./mlruns")


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
    for key, value in updates.model_dump(exclude_unset=True).items():
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
