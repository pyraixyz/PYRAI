import pytest
import torch
import torch.nn as nn
from pathlib import Path
from pyrai.utils.model_utils import (
    save_model_info,
    load_model_info,
    get_model_size,
    get_model_flops
)

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 111 * 111, 10)
        
    def forward(self, x):
        x = self.pool(self.conv(x))
        x = x.view(-1, 64 * 111 * 111)
        return self.fc(x)

@pytest.fixture
def model():
    """Create test model."""
    return SimpleModel()

@pytest.fixture
def model_info():
    """Create test model info."""
    return {
        "name": "test_model",
        "version": "v1",
        "input_size": [1, 3, 224, 224],
        "output_size": [1, 10],
        "parameters": 1000000
    }

def test_save_load_model_info(tmp_path, model_info):
    """Test saving and loading model info."""
    info_path = tmp_path / "model_info.json"
    
    # Save info
    save_model_info(info_path, model_info)
    assert info_path.exists()
    
    # Load info
    loaded_info = load_model_info(info_path)
    assert loaded_info == model_info

def test_get_model_size(model):
    """Test model size calculation."""
    size = get_model_size(model)
    assert isinstance(size, int)
    assert size > 0

def test_get_model_flops(model):
    """Test FLOPs calculation."""
    input_size = (1, 3, 224, 224)
    flops = get_model_flops(model, input_size)
    
    assert isinstance(flops, int)
    assert flops > 0 