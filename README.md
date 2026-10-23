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
- Best-so-far / simple-regret curves for sweep progress over time
- Learning-curve extrapolation (pow3 / exp3 saturating fits) with predicted final metric, asymptote and early-termination hints for runs unlikely to beat the incumbent
- Friedman rank test with Iman-Davenport F and Nemenyi critical difference for comparing configurations across matched seeds / datasets / folds
- Corrected resampled t-test (Nadeau-Bengio) and Bayesian correlated t-test with a ROPE (Corani-Benavoli) for comparing configurations across shared CV folds
- ANOVA hyperparameter importance (eta² / omega², F-test p-value, Spearman) for categorical and numeric params

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

## Friedman / Nemenyi comparison across seeds or datasets

When every configuration is run on the same seeds, datasets or CV folds,
`friedman_comparison` (endpoint `/experiments/{id}/friedman`) ranks the
configurations within each block (`block_param`, default `seed`) and runs
the Friedman test on the average ranks (tie-corrected chi-square plus the
Iman-Davenport F form), then the Nemenyi post-hoc: two configurations differ
when their average ranks differ by more than the critical difference
`q_alpha * sqrt(k(k+1) / (6N))` (Demšar, "Statistical Comparisons of
Classifiers over Multiple Data Sets", JMLR 2006). Ranks need no normality
assumption and are not swamped by one high-variance block. Only blocks where
every configuration has a value are used; the rest are listed in
`dropped_blocks`. `q_alpha` comes from the studentized range distribution,
computed numerically, so any number of configurations works.

```python
from src.models import friedman_comparison

res = friedman_comparison(runs, "accuracy", block_param="seed", alpha=0.05)
res["p_value"], res["critical_difference"]
for c in res["configurations"]:
    print(c["params"], c["average_rank"], c["worse_than_best"])
```

```bash
curl "http://localhost:8000/experiments/$EXP/friedman?metric=accuracy&block_param=dataset&ignore=seed,dataset"
```

`ExperimentTrackerClient.friedman_comparison(exp_id, "accuracy", block_param="dataset")`
wraps the endpoint.

## Bayesian comparison over CV folds

Fold scores from one cross-validation share most of their training data, so
the per-fold differences between two models are correlated and a plain paired
t-test is overconfident. `cv_bayesian_comparison` (endpoint
`/experiments/{id}/cv-bayesian`) matches configurations on `block_param`
(default `fold`), takes the configuration with the best mean as the reference
and, for each other configuration, runs on the per-fold differences:

- the **corrected resampled t-test** (Nadeau & Bengio, 2003), variance
  `(1/n + rho/(1-rho)) s^2` with `rho = test_fraction` (default `1/K`);
- the **Bayesian correlated t-test** (Corani & Benavoli, 2015): a Student-t
  posterior with the same scale, giving `p_best_better`, `p_equivalent`
  (inside the region of practical equivalence `[-rope, rope]`) and
  `p_other_better`, plus a `decision` once one of them reaches
  `decision_threshold` (default 0.95).

```python
from src.models import correlated_t_test, cv_bayesian_comparison

res = cv_bayesian_comparison(runs, "accuracy", block_param="fold", rope=0.01)
for c in res["comparisons"]:
    print(c["params"], c["mean_difference"], c["p_best_better"], c["p_equivalent"], c["decision"])

correlated_t_test([0.01, 0.02, -0.004, 0.015, 0.009], test_fraction=0.2, rope=0.005)
```

```bash
curl "http://localhost:8000/experiments/$EXP/cv-bayesian?metric=accuracy&rope=0.01"
```

`ExperimentTrackerClient.cv_bayesian_comparison(exp_id, "accuracy", rope=0.01)`
wraps the endpoint.

## Learning-curve extrapolation

`learning_curve_extrapolation(run, metric, target_step)` fits a saturating
curve to a run's `(step, value)` history -- `pow3` (`a + b * t^-c`, Domhan et
al. 2015) or `exp3` (`a + b * exp(-c t)`), `model="auto"` keeps the better
fit -- and predicts the metric at `target_step`. The fit is a deterministic
profile least-squares grid over `c` with closed-form `(a, b)`, so it needs no
extra dependencies. `a` is the asymptote.

`learning_curve_forecast(runs, metric, target_step, maximize=True, margin=0.0)`
does this for every run and sets `terminate_hint=True` on runs whose forecast
trails the best forecast by more than `margin`: candidates to stop early in a
sweep.

```python
from src.models import learning_curve_forecast

forecast = learning_curve_forecast(runs, "val_accuracy", target_step=50, margin=0.01)
[r["run_id"] for r in forecast["runs"] if r["terminate_hint"]]
```

```bash
curl "http://localhost:8000/experiments/$EXP/learning-curve-forecast?metric=val_accuracy&target_step=50&margin=0.01"
```

`ExperimentTrackerClient.learning_curve_forecast(exp_id, "val_accuracy", 50)` wraps the endpoint.

## Hyperparameter importance

Pearson correlation (`parameter-correlation`) only covers numeric params with
a linear effect. `parameter_importance` (and the `parameter-importance`
endpoint) instead runs a one-way ANOVA of each run's latest metric value,
grouped by each flattened param:

- `eta_squared`: the share of metric variance explained by the param.
- `omega_squared`: eta² with its small-sample bias corrected, clipped at 0.
  Rows are ranked by this value.
- `f_statistic` and `p_value`: the F-test, computed in pure Python with the
  existing incomplete-beta helper.
- `level_means`: the mean metric value for each param value.
- `spearman`: the rank correlation, for numeric params only.

This also handles categorical params (optimizer, activation) and
non-monotone effects, such as a learning rate whose best value is in the
middle. Numeric params with more than `max_levels` distinct values are put
into equal-count bins first. `seed` is ignored by default.

```python
from src.models import parameter_importance

for row in parameter_importance(runs, "accuracy", ignore=["seed"], max_levels=10):
    print(row["param_name"], row["kind"], round(row["omega_squared"], 3), row["p_value"])
```

```bash
curl "http://localhost:8000/experiments/$EXP/parameter-importance?metric=accuracy&ignore=seed&max_levels=10"
```

`ExperimentTrackerClient.parameter_importance(exp_id, "accuracy")` wraps the endpoint.

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
