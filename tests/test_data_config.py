import pytest
import torch
import numpy as np
from pyrai.data import TensorDataset, DataLoader
from pyrai.config import Config

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001
        },
        "model": {
            "type": "pytorch",
            "device": "cpu"
        },
        "data": {
            "num_workers": 2,
            "pin_memory": False
        }
    }

def test_tensor_dataset():
    """Test tensor dataset."""
    # Create random data
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    
    dataset = TensorDataset(inputs, targets)
    
    assert len(dataset) == 100
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (10,)

def test_data_loader():
    """Test data loader."""
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    dataset = TensorDataset(inputs, targets)
    
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    
    assert len(loader) == 4  # 100/32 rounded up
    batch = next(iter(loader))
    assert len(batch) == 2  # (inputs, targets)
    assert batch[0].shape[0] <= 32  # batch size

def test_config_validation(sample_config):
    """Test configuration validation."""
    config = Config(sample_config)
    
    assert config.get("training.batch_size") == 32
    assert config.get("model.device") == "cpu"
    assert config.get("nonexistent", "default") == "default"

def test_config_update(sample_config):
    """Test configuration update."""
    config = Config(sample_config)
    
    updates = {
        "training": {
            "batch_size": 64,
            "learning_rate": 0.01
        }
    }
    
    config.update(updates)
    assert config.get("training.batch_size") == 64
    assert config.get("training.learning_rate") == 0.01
    assert config.get("training.epochs") == 10  # unchanged 