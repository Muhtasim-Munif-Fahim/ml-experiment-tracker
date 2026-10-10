# Nested params

`flatten_params` / `Run.log_params` expand nested dicts into dotted keys (e.g. `model.lr`).

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
- Per-step metric history export (CSV and JSON), alongside the wide pivot CSV
- Seed-aggregated run groups: mean ± Student-t CI per configuration, with a Welch t-test against the best group

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

## Metric history export

Runs already accept string key/value tags (`tags` on create, `GET`/`PUT`/`DELETE /runs/{id}/tags`, and `query_runs(..., tags={...})`). Comparison reports stay Markdown and HTML and are not filtered by a second tag system. This export is the per-step metric history that sits next to the wide pivot table.

Two views of the same logged metrics:

| View | Endpoint | Shape |
| --- | --- | --- |
| Pivot | `GET /experiments/{id}/pivot.csv` | One row per run. Columns are `run_id`, `run_name`, then the **latest** value of each metric. |
| History | `GET /experiments/{id}/metrics.history.csv` and `GET /experiments/{id}/metrics.history.json` | One row per logged step: `run_id`, `run_name`, `metric_name`, `step`, `value`, `timestamp`. |

History accepts `metric_names` (comma-separated), `start_step`, and `end_step`. An unknown experiment is 404. A range with `start_step` greater than `end_step` is 400. Rows are ordered by run name, metric name, then step.

The JSON body is:

```json
{
  "experiment_id": "exp-id",
  "columns": ["run_id", "run_name", "metric_name", "step", "value", "timestamp"],
  "rows": [
    {
      "run_id": "run-id",
      "run_name": "train-a",
      "metric_name": "accuracy",
      "step": 2,
      "value": 0.9,
      "timestamp": "2026-09-22T00:00:00+00:00"
    }
  ]
}
```

`columns` matches the CSV header, so the two files describe the same table.

```python
from src.client import ExperimentTrackerClient

client = ExperimentTrackerClient("http://localhost:8000")
client.experiment_metric_history_csv(
    exp_id, metric_names=["accuracy"], destination="reports/history.csv"
)
payload = client.experiment_metric_history_json(
    exp_id, start_step=1, end_step=10, destination="reports/history.json"
)
```

A single run's points are also available as `GET /runs/{id}/metrics.csv` (columns `name`, `step`, `value`, `timestamp`).

## Aggregate runs across seeds

One run per configuration can't tell a real improvement from seed noise.
`group_runs_by_params` (and the `run-groups` endpoint) group runs by their
flattened params, ignoring `seed` by default, or by an explicit `group_by`
list. Each group reports `n`, `mean`, `std`, `sem` and a Student-t
confidence interval of each run's latest metric value. Groups are ordered
best-first (`maximize`), and every group after the first carries `vs_best`,
a Welch unequal-variance t-test against the best group (difference, df,
p-value, CI, `significant`). All of it is pure Python, with no scipy.

```python
from src.models import group_runs_by_params, welch_t_test

groups = group_runs_by_params(runs, "accuracy", ignore=["seed"], confidence=0.95)
for g in groups:
    print(g["rank"], g["params"], g["n"], g["mean"], (g["ci_low"], g["ci_high"]),
          g["vs_best"] and g["vs_best"]["p_value"])
```

```bash
curl "http://localhost:8000/experiments/$EXP/run-groups?metric=accuracy&ignore=seed"
curl "http://localhost:8000/experiments/$EXP/run-groups?metric=loss&group_by=optimizer&maximize=false"
```

`ExperimentTrackerClient.run_groups(exp_id, "accuracy", group_by=["optimizer"])`
wraps the endpoint.

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
