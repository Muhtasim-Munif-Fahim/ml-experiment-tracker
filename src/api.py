"""REST API for ML Experiment Tracker"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .models import Experiment, Run, RunStatus, Param, Metric, Artifact, ArtifactType
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
def list_experiments(limit: int = 100, offset: int = 0):
    exps = storage.list_experiments()
    return exps[offset:offset+limit]


@app.get("/experiments/{exp_id}", response_model=dict)
def get_experiment(exp_id: str):
    exp = storage.load_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


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
    run_data.setdefault("metrics", []).append(metric.model_dump())
    storage.save_run(run_data)
    return {"message": "Metric logged"}


@app.post("/runs/{run_id}/metrics/batch", response_model=dict)
def log_metrics(run_id: str, batch: MetricBatchCreate):
    run_data = storage.load_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    timestamp = datetime.utcnow().isoformat()
    run_data.setdefault("metrics", []).extend(
        {
            "name": name,
            "value": value,
            "step": batch.step,
            "timestamp": timestamp,
        }
        for name, value in batch.metrics.items()
    )
    storage.save_run(run_data)
    return {"message": "Metrics logged", "count": len(batch.metrics)}


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
    run_data.setdefault("artifacts", []).append(artifact.to_dict())
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
