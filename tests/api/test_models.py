import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path

from pyrai.api.server import app
from pyrai.models.user import User
from pyrai.models.model import Model
from pyrai.core.constants import ModelType, ModelFramework, ModelStatus

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
def mock_model():
    """Mock model fixture."""
    return Model(
        id=1,
        name="test_model",
        description="Test model",
        model_type=ModelType.CLASSIFICATION,
        framework=ModelFramework.PYTORCH,
        status=ModelStatus.DEPLOYED,
        user_id=1
    )

@pytest.mark.asyncio
async def test_list_models(client, mock_user, mock_model):
    """Test list models endpoint."""
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("sqlalchemy.ext.asyncio.AsyncSession.scalars", return_value=[mock_model]):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_model"

@pytest.mark.asyncio
async def test_create_model(client, mock_user):
    """Test create model endpoint."""
    model_data = {
        "name": "new_model",
        "description": "New test model",
        "model_type": "classification",
        "framework": "pytorch"
    }
    
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user):
        response = client.post("/models", json=model_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new_model"
        assert data["user_id"] == mock_user.id

@pytest.mark.asyncio
async def test_get_model(client, mock_user, mock_model):
    """Test get model endpoint."""
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("sqlalchemy.ext.asyncio.AsyncSession.get", return_value=mock_model):
        response = client.get("/models/1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["name"] == "test_model"

@pytest.mark.asyncio
async def test_update_model(client, mock_user, mock_model):
    """Test update model endpoint."""
    update_data = {
        "name": "updated_model",
        "description": "Updated description"
    }
    
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("sqlalchemy.ext.asyncio.AsyncSession.get", return_value=mock_model):
        response = client.put("/models/1", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated_model"
        assert data["description"] == "Updated description"

@pytest.mark.asyncio
async def test_create_version(client, mock_user, mock_model):
    """Test create model version endpoint."""
    test_file = Path(__file__).parent / "test_data" / "model.pt"
    
    with patch("pyrai.api.auth.get_current_user", return_value=mock_user), \
         patch("sqlalchemy.ext.asyncio.AsyncSession.get", return_value=mock_model), \
         patch("pyrai.core.state.AppState.get_model_path", return_value=str(test_file)):
        
        files = {"file": ("model.pt", open(test_file, "rb"), "application/octet-stream")}
        response = client.post("/models/1/versions", files=files)
        assert response.status_code == 201
        data = response.json()
        assert data["model_id"] == 1
        assert "version" in data
        assert "file_path" in data 