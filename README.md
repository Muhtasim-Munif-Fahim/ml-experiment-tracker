# ML Experiment Tracker

A lightweight experiment tracking and model registry system for ML workflows.

## Features

- Experiment logging and versioning
- Metric/parameter/hyperparameter tracking
- Model registry with versioning
- Artifact storage (local/S3/GCS)
- Web UI for experiment comparison
- REST API for integration
- Shareable Markdown/HTML run-comparison reports

## Quick Start

```bash
pip install -r requirements.txt
python run.py --help
```

`python run.py` (or `python run.py serve`) starts the FastAPI server.

## Compare runs

Compare two or more runs side-by-side (latest metrics plus parameter diffs) and export a Markdown or HTML report. This uses local storage only; S3 is not required.

### Library

```python
from src.models import Experiment, Run, compare_run_records, render_run_comparison, write_run_comparison_report

baseline = Run(experiment_id="exp", name="baseline", params={"lr": 0.1, "seed": 7})
baseline.log_metric("accuracy", 0.82, step=2)
candidate = Run(experiment_id="exp", name="tuned", params={"lr": 0.05, "seed": 7})
candidate.log_metric("accuracy", 0.87, step=2)

comparison = compare_run_records([baseline, candidate])
markdown = render_run_comparison(comparison, "markdown")
html = render_run_comparison(comparison, "html")
write_run_comparison_report(comparison, "reports/compare.md", fmt="markdown")

# In-memory experiment helper (same payload):
exp = Experiment(name="vision")
exp.add_run(baseline)
exp.add_run(candidate)
exp.compare_many([baseline.id, candidate.id])
```

Stored runs can be compared the same way through `LocalStorageBackend`:

```python
from src.storage import LocalStorageBackend

storage = LocalStorageBackend("./mlruns")
storage.compare_runs([run_a_id, run_b_id])
storage.export_run_comparison([run_a_id, run_b_id], "reports/compare.html", fmt="html")
```

### CLI

```bash
python run.py compare \
  --run-id <id-1> --run-id <id-2> \
  --format markdown \
  --output reports/compare.md \
  --storage ./mlruns
```

Omit `--output` to print the report to stdout. `--format html` writes a standalone HTML page.

### HTTP API and client

```http
POST /runs/compare
{"run_ids": ["id-1", "id-2"]}

GET /runs/compare.md?run_ids=id-1,id-2
GET /runs/compare.html?run_ids=id-1,id-2
```

```python
from src.client import ExperimentTrackerClient

client = ExperimentTrackerClient("http://localhost:8000")
client.compare_run_records([run_a_id, run_b_id])
client.export_run_comparison([run_a_id, run_b_id], destination="reports/compare.md")
```

The pairwise experiment helper `POST /experiments/{id}/runs/compare` is unchanged.

## Project Structure

```
src/
  models.py        # Core data models (Experiment, Run, Metric, Artifact)
  storage.py       # Storage backends (local filesystem, S3 stub)
  api.py           # FastAPI REST API
  client.py        # HTTP client + experiment context manager
  ui.py            # Streamlit dashboard
tests/
config.yaml
run.py
```

## Requirements

- Python 3.8+
- FastAPI, SQLAlchemy, pydantic
- Streamlit + Plotly (dashboard)
- requests (client)
- click (CLI)
