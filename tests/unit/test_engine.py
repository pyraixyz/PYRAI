import pytest
from pyrai import Engine
from pyrai.backends import PyTorchBackend

def test_engine_initialization():
    """Test Engine initialization."""
    engine = Engine(backend=PyTorchBackend())
    assert engine.backend is not None
    assert isinstance(engine.backend, PyTorchBackend)

def test_engine_train():
    """Test model training."""
    engine = Engine(backend=PyTorchBackend())
    
    # Mock training data
    model = MockModel()
    train_data = MockDataset()
    
    history = engine.train(
        model=model,
        train_data=train_data
    )
    
    assert isinstance(history, dict)
    assert "loss" in history
    assert "accuracy" in history

def test_engine_deploy():
    """Test model deployment."""
    engine = Engine(backend=PyTorchBackend())
    
    deployment = engine.deploy(
        model_id="test_model",
        version="v1"
    )
    
    assert deployment.name == "test_model-v1"
    assert deployment.endpoint is not None

@pytest.mark.asyncio
async def test_engine_predict():
    """Test model prediction."""
    engine = Engine(backend=PyTorchBackend())
    
    predictions = await engine.predict(
        model_id="test_model",
        version="v1",
        inputs={"data": [1, 2, 3]}
    )
    
    assert predictions is not None
    assert isinstance(predictions, dict)

class MockModel:
    """Mock model for testing."""
    def __init__(self):
        pass
    
    def train(self):
        return {"loss": 0.1, "accuracy": 0.95}

class MockDataset:
    """Mock dataset for testing."""
    def __init__(self):
        pass 