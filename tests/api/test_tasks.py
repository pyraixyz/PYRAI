import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from pyrai.api.server import app
from pyrai.models.user import User
from pyrai.models.task import Task
from pyrai.core.constants import TaskStatus

@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)

@pytest.fixture
def mock_user():
    """Mock user fixture."""
    return User(
        id=1,
        email="user@example.com",
        full_name="Test User",
        is_active=True
    )

@pytest.fixture
def mock_task():
    """Mock task fixture."""
    return Task(
        id=1,
        name="test_task",
        task_type="training",
        status=TaskStatus.RUNNING,
        user_id=1
    )

@pytest.mark.asyncio
async def test_list_tasks(client, mock_user, mock_task):
    """Test list tasks endpoint."""
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("sqlalchemy.ext.asyncio.AsyncSession.scalars", return_value=[mock_task]):
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_task"

@pytest.mark.asyncio
async def test_create_task(client, mock_user):
    """Test create task endpoint."""
    task_data = {
        "name": "new_task",
        "task_type": "training",
        "params": {"epochs": 10}
    }
    
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("pyrai.core.tasks.task_manager.create_task") as mock_create:
        mock_create.return_value = Task(**task_data, id=1, user_id=mock_user.id)
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new_task"
        assert data["user_id"] == mock_user.id

@pytest.mark.asyncio
async def test_get_task(client, mock_user, mock_task):
    """Test get task endpoint."""
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("sqlalchemy.ext.asyncio.AsyncSession.get", return_value=mock_task):
        response = client.get("/tasks/1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["name"] == "test_task"

@pytest.mark.asyncio
async def test_cancel_task(client, mock_user, mock_task):
    """Test cancel task endpoint."""
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("sqlalchemy.ext.asyncio.AsyncSession.get", return_value=mock_task), \
         patch("pyrai.core.tasks.task_manager.cancel_task", return_value=True):
        response = client.post("/tasks/1/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled" 