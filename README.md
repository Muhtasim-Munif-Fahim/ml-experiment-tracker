# ML Experiment Tracker

A lightweight experiment tracking and model registry system for ML workflows.

## Features

- Experiment logging and versioning
- Metric/parameter/hyperparameter tracking
- Model registry with versioning
- Artifact storage (local/S3/GCS)
- Web UI for experiment comparison
- REST API for integration

## Quick Start

```bash
pip install -r requirements.txt
python run.py --help
```

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