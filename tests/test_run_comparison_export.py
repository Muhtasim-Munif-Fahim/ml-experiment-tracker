"""Tests for multi-run Markdown/HTML comparison export."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from run import main
from src.models import (
    Experiment,
    Run,
    RunStatus,
    compare_run_records,
    render_run_comparison,
    write_run_comparison_report,
)
from src.storage import LocalStorageBackend


def _fake_runs() -> tuple[Run, Run, Run]:
    baseline = Run(
        experiment_id="exp-vision",
        name="baseline",
        params={"lr": 0.1, "seed": 7, "optimizer": "adam"},
    )
    baseline.status = RunStatus.COMPLETED
    baseline.log_metric("accuracy", 0.80, step=1)
    baseline.log_metric("accuracy", 0.82, step=2)
    baseline.log_metric("loss", 0.40, step=2)

    tuned = Run(
        experiment_id="exp-vision",
        name="tuned",
        params={"lr": 0.05, "seed": 7, "optimizer": "adam"},
    )
    tuned.status = RunStatus.COMPLETED
    tuned.log_metric("accuracy", 0.87, step=2)
    tuned.log_metric("loss", 0.31, step=2)
    tuned.log_metric("f1", 0.84, step=2)

    wide = Run(
        experiment_id="exp-vision",
        name="wide",
        params={"lr": 0.05, "seed": 7, "batch_size": 64},
    )
    wide.log_metric("accuracy", 0.90, step=3)
    return baseline, tuned, wide


def test_compare_run_records_metrics_side_by_side_and_param_diffs():
    baseline, tuned, _ = _fake_runs()

    comparison = compare_run_records([baseline, tuned])

    assert comparison["run_ids"] == [baseline.id, tuned.id]
    assert comparison["baseline_run_id"] == baseline.id

    by_name = {entry["name"]: entry for entry in comparison["metrics"]}
    assert by_name["accuracy"]["values"][baseline.id] == pytest.approx(0.82)
    assert by_name["accuracy"]["values"][tuned.id] == pytest.approx(0.87)
    assert by_name["accuracy"]["deltas"][tuned.id] == pytest.approx(0.05)
    assert by_name["f1"]["values"][baseline.id] is None
    assert by_name["f1"]["values"][tuned.id] == pytest.approx(0.84)

    assert comparison["shared_params"] == {"optimizer": "adam", "seed": 7}
    diffs = {entry["name"]: entry["values"] for entry in comparison["param_diffs"]}
    assert diffs == {"lr": {baseline.id: 0.1, tuned.id: 0.05}}


def test_compare_run_records_supports_three_runs():
    baseline, tuned, wide = _fake_runs()
    comparison = compare_run_records([baseline, tuned, wide])
    assert comparison["run_ids"] == [baseline.id, tuned.id, wide.id]
    diffs = {entry["name"]: entry["values"] for entry in comparison["param_diffs"]}
    assert diffs["batch_size"][wide.id] == 64
    assert diffs["batch_size"][baseline.id] is None
    assert "optimizer" in diffs
    assert comparison["shared_params"] == {"seed": 7}


def test_compare_run_records_rejects_too_few_or_duplicate_ids():
    run = Run(experiment_id="e", name="only")
    with pytest.raises(ValueError, match="at least two"):
        compare_run_records([run])
    clone = Run(experiment_id="e", name="copy")
    clone.id = run.id
    with pytest.raises(ValueError, match="unique"):
        compare_run_records([run, clone])


def test_experiment_compare_many_uses_in_memory_runs():
    experiment = Experiment(name="vision")
    baseline, tuned, _ = _fake_runs()
    experiment.add_run(baseline)
    experiment.add_run(tuned)
    comparison = experiment.compare_many([tuned.id, baseline.id])
    assert comparison["baseline_run_id"] == tuned.id
    with pytest.raises(KeyError, match="not found"):
        experiment.compare_many([baseline.id, "missing"])


def test_compare_run_records_accepts_fake_stored_dicts():
    records = [
        {
            "id": "run-a",
            "name": "a",
            "status": "completed",
            "experiment_id": "exp-1",
            "params": {"lr": 0.1},
            "metrics": [{"name": "loss", "value": 0.4, "step": 1}],
        },
        {
            "id": "run-b",
            "name": "b",
            "status": "running",
            "experiment_id": "exp-1",
            "params": {"lr": 0.01, "wd": 1e-4},
            "metrics": [{"name": "loss", "value": 0.2, "step": 1}],
        },
    ]
    comparison = compare_run_records(records)
    assert comparison["metrics"][0]["values"] == {"run-a": 0.4, "run-b": 0.2}
    assert comparison["metrics"][0]["deltas"]["run-b"] == pytest.approx(-0.2)
    diffs = {entry["name"]: entry["values"] for entry in comparison["param_diffs"]}
    assert diffs["lr"] == {"run-a": 0.1, "run-b": 0.01}
    assert diffs["wd"]["run-a"] is None


def test_markdown_report_contains_tables_and_values():
    baseline, tuned, _ = _fake_runs()
    text = render_run_comparison(compare_run_records([baseline, tuned]), "markdown")
    assert text.startswith("# Run comparison")
    assert "| Metric | baseline | tuned |" in text
    assert "| accuracy |" in text
    assert "0.82" in text
    assert "0.87" in text
    assert "| lr |" in text
    assert "0.1" in text
    assert "0.05" in text
    assert "seed" in text


def test_html_report_is_standalone_and_escapes_content():
    left = Run(experiment_id="e", name="<script>alert(1)</script>", params={"note": "<b>x</b>"})
    left.log_metric("acc", 1.0, step=1)
    right = Run(experiment_id="e", name="safe", params={"note": "ok"})
    right.log_metric("acc", 2.0, step=1)
    html = render_run_comparison(compare_run_records([left, right]), "html")
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "<table>" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "2" in html


def test_render_run_comparison_rejects_unknown_format():
    baseline, tuned, _ = _fake_runs()
    with pytest.raises(ValueError, match="unsupported comparison format"):
        render_run_comparison(compare_run_records([baseline, tuned]), "pdf")


def test_write_run_comparison_report_creates_parent(tmp_path: Path):
    baseline, tuned, _ = _fake_runs()
    destination = tmp_path / "exports" / "compare.md"
    path = write_run_comparison_report(
        compare_run_records([baseline, tuned]), str(destination), fmt="markdown"
    )
    assert path == str(destination)
    assert destination.read_text(encoding="utf-8").startswith("# Run comparison")


def _seed_storage(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "mlruns")
    experiment = Experiment(name="vision")
    storage.save_experiment(experiment.to_dict())
    baseline, tuned, wide = _fake_runs()
    baseline.experiment_id = experiment.id
    tuned.experiment_id = experiment.id
    wide.experiment_id = experiment.id
    storage.save_run(baseline.to_dict())
    storage.save_run(tuned.to_dict())
    storage.save_run(wide.to_dict())
    return storage, baseline, tuned, wide


def test_storage_compare_and_export_from_fake_records(tmp_path: Path):
    storage, baseline, tuned, wide = _seed_storage(tmp_path)
    comparison = storage.compare_runs([baseline.id, tuned.id, wide.id])
    assert [run["name"] for run in comparison["runs"]] == ["baseline", "tuned", "wide"]

    destination = tmp_path / "out" / "report.html"
    written = storage.export_run_comparison(
        [baseline.id, tuned.id], str(destination), fmt="html"
    )
    assert written == str(destination)
    body = destination.read_text(encoding="utf-8")
    assert "<h1>Run comparison</h1>" in body
    assert "baseline" in body and "tuned" in body


def test_storage_compare_rejects_missing_and_short_lists(tmp_path: Path):
    storage, baseline, _, _ = _seed_storage(tmp_path)
    with pytest.raises(KeyError, match="run not found"):
        storage.compare_runs([baseline.id, "missing"])
    with pytest.raises(ValueError, match="at least two"):
        storage.compare_runs([baseline.id])
    with pytest.raises(ValueError, match="unique"):
        storage.compare_runs([baseline.id, baseline.id])


def test_api_compare_json_markdown_and_html(api):
    experiment = api.post("/experiments/", json={"name": "vision"}).json()
    baseline = api.post(
        f"/experiments/{experiment['id']}/runs/",
        json={"name": "baseline", "params": {"lr": 0.1, "seed": 7}},
    ).json()
    candidate = api.post(
        f"/experiments/{experiment['id']}/runs/",
        json={"name": "candidate", "params": {"lr": 0.05, "seed": 7}},
    ).json()
    api.post(
        f"/runs/{baseline['id']}/metrics",
        json={"name": "accuracy", "value": 0.82, "step": 2},
    )
    api.post(
        f"/runs/{candidate['id']}/metrics",
        json={"name": "accuracy", "value": 0.87, "step": 2},
    )

    payload = api.post(
        "/runs/compare", json={"run_ids": [baseline["id"], candidate["id"]]}
    )
    assert payload.status_code == 200
    body = payload.json()
    assert body["metrics"][0]["name"] == "accuracy"
    assert body["metrics"][0]["deltas"][candidate["id"]] == pytest.approx(0.05)
    assert body["shared_params"] == {"seed": 7}

    markdown = api.get(
        "/runs/compare.md",
        params={"run_ids": f"{baseline['id']},{candidate['id']}"},
    )
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert (
        'attachment; filename="run-comparison.md"'
        in markdown.headers["content-disposition"]
    )
    assert "| accuracy |" in markdown.text

    html = api.get(
        "/runs/compare.html",
        params={"run_ids": f"{baseline['id']},{candidate['id']}"},
    )
    assert html.status_code == 200
    assert html.headers["content-type"].startswith("text/html")
    assert "<table>" in html.text


def test_api_compare_validates_ids(api):
    too_few = api.post("/runs/compare", json={"run_ids": ["only-one"]})
    assert too_few.status_code == 422
    missing = api.get("/runs/compare.md", params={"run_ids": "a,b"})
    assert missing.status_code == 404
    short = api.get("/runs/compare.html", params={"run_ids": "only-one"})
    assert short.status_code == 400


def test_client_exports_comparison(tracker, tmp_path):
    experiment = tracker.create_experiment("vision")
    run_a = tracker.create_run(experiment["id"], "resnet-a", params={"lr": 0.1})
    run_b = tracker.create_run(experiment["id"], "resnet-b", params={"lr": 0.05})
    tracker.log_metric(run_a["id"], "accuracy", 0.5, step=1)
    tracker.log_metric(run_b["id"], "accuracy", 0.9, step=2)

    comparison = tracker.compare_run_records([run_a["id"], run_b["id"]])
    assert comparison["param_diffs"][0]["name"] == "lr"

    destination = tmp_path / "exports" / "compare.md"
    text = tracker.export_run_comparison(
        [run_a["id"], run_b["id"]], destination=str(destination), fmt="markdown"
    )
    assert destination.read_text(encoding="utf-8") == text
    assert "# Run comparison" in text
    html = tracker.export_run_comparison([run_a["id"], run_b["id"]], fmt="html")
    assert html.lstrip().startswith("<!DOCTYPE html>")


def test_cli_compare_writes_markdown_from_local_storage(tmp_path: Path):
    storage, baseline, tuned, _ = _seed_storage(tmp_path)
    destination = tmp_path / "report.md"
    result = CliRunner().invoke(
        main,
        [
            "compare",
            "--run-id",
            baseline.id,
            "--run-id",
            tuned.id,
            "--storage",
            str(tmp_path / "mlruns"),
            "--output",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    assert destination.exists()
    assert "# Run comparison" in destination.read_text(encoding="utf-8")
    assert str(destination) in result.output


def test_cli_compare_prints_html_to_stdout(tmp_path: Path):
    storage, baseline, tuned, _ = _seed_storage(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "compare",
            "--run-id",
            baseline.id,
            "--run-id",
            tuned.id,
            "--format",
            "html",
            "--storage",
            str(tmp_path / "mlruns"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.lstrip().startswith("<!DOCTYPE html>")


def test_cli_compare_requires_two_run_ids():
    result = CliRunner().invoke(main, ["compare", "--run-id", "only-one"])
    assert result.exit_code != 0
    assert "at least two" in result.output
