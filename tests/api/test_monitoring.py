import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from pyrai.api.server import app
from pyrai.core.state import app_state
from pyrai.models.user import User

@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)

@pytest.fixture
def mock_superuser():
    """Mock superuser fixture."""
    return User(
        id=1,
        email="admin@example.com",
        full_name="Admin User",
        is_active=True,
        is_superuser=True
    )

@pytest.mark.asyncio
async def test_get_metrics(client, mock_superuser):
    """Test metrics endpoint."""
    with patch("pyrai.api.auth.get_current_active_superuser", return_value=mock_superuser):
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4"

@pytest.mark.asyncio
async def test_get_system_metrics(client, mock_superuser):
    """Test system metrics endpoint."""
    with patch("pyrai.api.auth.get_current_active_superuser", return_value=mock_superuser):
        response = client.get("/monitoring/system")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "db_connections" in data
        assert "uptime_seconds" in data
        assert "active_tasks" in data

@pytest.mark.asyncio
async def test_get_model_metrics(client, mock_superuser):
    """Test model metrics endpoint."""
    mock_model = MagicMock(
        name="test_model",
        current_version="1.0.0",
        versions=["1.0.0"],
        last_inference="2024-01-01T00:00:00",
        last_training="2024-01-01T00:00:00"
    )
    
    with patch("pyrai.api.auth.get_current_active_superuser", return_value=mock_superuser), \
         patch("pyrai.core.state.AppState.get_model", return_value=mock_model):
        response = client.get("/monitoring/models/1")
        assert response.status_code == 200
        data = response.json()
        assert "inference_count" in data
        assert "training_duration" in data
        assert "versions" in data
        assert "last_inference" in data
        assert "last_training" in data

@pytest.mark.asyncio
async def test_get_task_metrics(client, mock_superuser):
    """Test task metrics endpoint."""
    mock_task = MagicMock(
        status="running",
        progress=50,
        get_duration=lambda: 60.0,
        memory_usage=1024,
        cpu_usage=50.0
    )
    
    with patch("pyrai.api.auth.get_current_active_superuser", return_value=mock_superuser), \
         patch("pyrai.core.state.AppState.get_task", return_value=mock_task):
        response = client.get("/monitoring/tasks/1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["progress"] == 50
        assert data["duration"] == 60.0
        assert data["memory_usage"] == 1024
        assert data["cpu_usage"] == 50.0

@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint."""
    with patch("pyrai.core.state.AppState.session") as mock_session, \
         patch("pyrai.core.state.AppState.redis") as mock_redis, \
         patch("pyrai.core.state.AppState.check_task_queue", return_value=True):
        
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock()
        mock_session.return_value = mock_session_ctx
        
        mock_redis_ctx = MagicMock()
        mock_redis_ctx.__aenter__.return_value.ping = AsyncMock()
        mock_redis.return_value = mock_redis_ctx
        
        response = client.get("/monitoring/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] is True
        assert data["cache"] is True
        assert data["task_queue"] is True
        assert "uptime" in data 