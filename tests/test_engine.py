import pytest
import torch
import numpy as np
from pyrai import Engine
from pyrai.config import Config
from pyrai.backends import PyTorchBackend
from pyrai.data import TensorDataset

class DummyModel(torch.nn.Module):
    """Dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def engine():
    """Create test engine instance."""
    config = Config({
        "training": {
            "batch_size": 32,
            "epochs": 2
        },
        "model": {
            "type": "pytorch",
            "device": "cpu"
        }
    })
    return Engine(backend=PyTorchBackend(), config=config)

@pytest.fixture
def dummy_data():
    """Create dummy dataset."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return TensorDataset(X, y)

def test_train(engine, dummy_data):
    """Test model training."""
    model = DummyModel()
    history = engine.train(
        model=model,
        train_data=dummy_data,
        val_data=dummy_data
    )
    
    assert isinstance(history, dict)
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 2

def test_predict(engine, dummy_data):
    """Test model prediction."""
    model = DummyModel()
    inputs = torch.randn(16, 10)
    
    predictions = engine.predict(model, inputs)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (16, 2)

def test_deploy(engine, dummy_data):
    """Test model deployment."""
    model = DummyModel()
    deployment = engine.deploy(
        model=model,
        model_id="test_model"
    )
    
    assert deployment.model_id == "test_model"
    assert deployment.status == "running"
    
    # Test prediction
    inputs = torch.randn(16, 10)
    predictions = deployment.predict(inputs)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (16, 2)
    
    # Test stop
    deployment.stop()
    assert deployment.status == "stopped" 