"""Shared fixtures for API and client layer tests."""

import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

import src.api as api_module
from src.client import ExperimentTrackerClient
from src.storage import LocalStorageBackend


@pytest.fixture()
def temp_storage(tmp_path):
    return LocalStorageBackend(tmp_path / "mlruns")


@pytest.fixture()
def api(monkeypatch, temp_storage):
    """TestClient wired to a throwaway storage backend."""
    monkeypatch.setattr(api_module, "storage", temp_storage)
    with TestClient(api_module.app) as test_client:
        yield test_client


@pytest.fixture()
def live_server(monkeypatch, temp_storage):
    """Run the API on an ephemeral local port for HTTP client tests."""
    monkeypatch.setattr(api_module, "storage", temp_storage)
    server = uvicorn.Server(
        uvicorn.Config(api_module.app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("test API server did not start")
    host, port = server.servers[0].sockets[0].getsockname()[:2]
    yield f"http://{host}:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def tracker(live_server):
    return ExperimentTrackerClient(base_url=live_server)
