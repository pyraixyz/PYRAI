import pytest
import requests
import numpy as np
from engine.deployment.manager import DeploymentManager
from engine.deployment.config import DeploymentConfig
from engine.core.versioning import VersionManager
from engine.backends.pytorch import PyTorchBackend
from pyrai import Engine
import torch.nn as nn

@pytest.mark.integration
def test_model_deployment(temp_dir, sample_model_pytorch):
    """Test end-to-end model deployment."""
    # Setup components
    version_manager = VersionManager(str(temp_dir / 'models'))
    deployment_manager = DeploymentManager(
        version_manager=version_manager,
        config_path=str(temp_dir / 'config'),
        registry_url="localhost:5000"
    )
    
    # Save model version
    version = version_manager.save_version(
        model=sample_model_pytorch,
        model_id="test_model",
        description="Test deployment",
        metrics={"accuracy": 0.95},
        parameters={"lr": 0.001}
    )
    
    # Create deployment config
    config = DeploymentConfig(
        model_id="test_model",
        version_id=version.version_id,
        name="test-deployment",
        description="Test deployment",
        resources={
            "cpu": "1",
            "memory": "2Gi"
        },
        scaling={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_cpu_utilization": 80
        },
        monitoring={
            "enable_metrics": True,
            "enable_logging": True
        }
    )
    
    # Deploy model
    deployment_name = deployment_manager.deploy_model(config)
    
    # Wait for deployment to be ready
    import time
    time.sleep(30)  # Wait for pods to start
    
    # Test prediction endpoint
    response = requests.post(
        f"http://localhost/models/{deployment_name}/predict",
        json={
            "inputs": {"data": [[0.0] * 10]}
        }
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    
    # Cleanup
    deployment_manager.delete_deployment(deployment_name)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

@pytest.mark.integration
def test_full_workflow():
    """Test complete training and deployment workflow."""
    # Initialize engine
    engine = Engine(backend=PyTorchBackend())
    
    # Train model
    model = SimpleModel()
    train_data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    
    history = engine.train(
        model=model,
        train_data=train_data,
        labels=labels
    )
    
    assert "loss" in history
    
    # Deploy model
    deployment = engine.deploy(
        model_id="test_model",
        version="v1"
    )
    
    assert deployment.status == "running"
    
    # Make predictions
    inputs = torch.randn(1, 10)
    predictions = deployment.predict(inputs)
    
    assert predictions.shape == (1, 2)
    
    # Clean up
    deployment.delete() 