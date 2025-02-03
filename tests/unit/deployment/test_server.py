import pytest
from fastapi.testclient import TestClient
from engine.deployment.server import ModelServer
from engine.core.versioning import VersionManager
from engine.backends.pytorch import PyTorchBackend

def test_model_server(temp_dir, sample_model_pytorch):
    """Test model server."""
    # Setup
    version_manager = VersionManager(str(temp_dir))
    backend = PyTorchBackend()
    
    # Create server
    server = ModelServer(
        version_manager=version_manager,
        backend=backend
    )
    
    # Create test client
    client = TestClient(server.app)
    
    # Save test model
    version = version_manager.save_version(
        model=sample_model_pytorch,
        model_id="test_model",
        description="Test model",
        metrics={"accuracy": 0.95},
        parameters={"lr": 0.001}
    )
    
    # Test prediction endpoint
    response = client.post(
        "/predict",
        json={
            "inputs": {"data": [[0.0] * 10]},
            "model_id": "test_model",
            "version_id": version.version_id
        }
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    
    # Test version listing
    response = client.get("/models/test_model/versions")
    assert response.status_code == 200
    assert len(response.json()) == 1
    
    # Test health check
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy" 