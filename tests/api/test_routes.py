"""
Tests for API routes.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock

from src.api.app import app
from src.core.training_coordinator import TrainingCoordinator
from src.core.exceptions import TrainingError


@pytest.fixture
def mock_coordinator():
    """Create a mock training coordinator"""
    coordinator = Mock(spec=TrainingCoordinator)
    coordinator.start_training = AsyncMock(return_value="job-123")
    coordinator.stop_training = AsyncMock(return_value=True)
    coordinator.get_progress = AsyncMock(return_value={
        "id": "job-123",
        "model_id": "model-123",
        "status": "running",
        "progress": 0.5,
        "start_time": "2024-01-01T00:00:00",
        "end_time": None,
        "error": None
    })
    coordinator.list_jobs = AsyncMock(return_value=[{
        "id": "job-123",
        "model_id": "model-123",
        "status": "running",
        "progress": 0.5,
        "start_time": "2024-01-01T00:00:00",
        "end_time": None
    }])
    coordinator.get_job_logs = AsyncMock(return_value=["Log message 1", "Log message 2"])
    coordinator.cleanup_jobs = AsyncMock(return_value=1)
    return coordinator


@pytest.fixture
def client(mock_coordinator):
    """Create a test client"""
    app.dependency_overrides = {
        "src.api.routes.get_coordinator": lambda: mock_coordinator
    }
    return TestClient(app)


def test_root(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_start_training(client, mock_coordinator):
    """Test starting a training job"""
    config = {
        "model_id": "model-123",
        "cpu_cores": 2,
        "memory_mb": 1024,
        "gpu_devices": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "max_epochs": 100
    }
    
    response = client.post("/api/v1/training/jobs", json=config)
    assert response.status_code == 200
    assert response.json()["id"] == "job-123"
    assert response.json()["status"] == "running"


def test_start_training_error(client, mock_coordinator):
    """Test starting a training job with error"""
    mock_coordinator.start_training.side_effect = TrainingError("Failed to start training")
    
    config = {
        "model_id": "model-123"
    }
    
    response = client.post("/api/v1/training/jobs", json=config)
    assert response.status_code == 400
    assert "Failed to start training" in response.json()["error"]


def test_stop_training(client, mock_coordinator):
    """Test stopping a training job"""
    response = client.delete("/api/v1/training/jobs/job-123")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_stop_nonexistent_job(client, mock_coordinator):
    """Test stopping a nonexistent job"""
    mock_coordinator.stop_training.return_value = False
    
    response = client.delete("/api/v1/training/jobs/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_job_progress(client, mock_coordinator):
    """Test getting job progress"""
    response = client.get("/api/v1/training/jobs/job-123")
    assert response.status_code == 200
    assert response.json()["id"] == "job-123"
    assert response.json()["status"] == "running"
    assert response.json()["progress"] == 0.5


def test_get_nonexistent_job_progress(client, mock_coordinator):
    """Test getting progress for nonexistent job"""
    mock_coordinator.get_progress.side_effect = TrainingError("Job not found")
    
    response = client.get("/api/v1/training/jobs/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_list_jobs(client, mock_coordinator):
    """Test listing jobs"""
    response = client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert len(response.json()["jobs"]) == 1
    assert response.json()["jobs"][0]["id"] == "job-123"


def test_list_jobs_with_status(client, mock_coordinator):
    """Test listing jobs with status filter"""
    response = client.get("/api/v1/training/jobs?status=running")
    assert response.status_code == 200
    mock_coordinator.list_jobs.assert_called_with("running")


def test_get_job_logs(client, mock_coordinator):
    """Test getting job logs"""
    response = client.get("/api/v1/training/jobs/job-123/logs")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "Log message 1"


def test_get_nonexistent_job_logs(client, mock_coordinator):
    """Test getting logs for nonexistent job"""
    mock_coordinator.get_job_logs.side_effect = TrainingError("Job not found")
    
    response = client.get("/api/v1/training/jobs/nonexistent/logs")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_cleanup_jobs(client, mock_coordinator):
    """Test cleaning up jobs"""
    response = client.post("/api/v1/training/cleanup?max_age_days=7")
    assert response.status_code == 200
    assert response.json()["cleaned"] == 1
    assert response.json()["success"] is True


def test_cleanup_jobs_error(client, mock_coordinator):
    """Test cleaning up jobs with error"""
    mock_coordinator.cleanup_jobs.side_effect = TrainingError("Failed to cleanup")
    
    response = client.post("/api/v1/training/cleanup")
    assert response.status_code == 400
    assert "Failed to cleanup" in response.json()["error"] 