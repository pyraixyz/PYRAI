import pytest
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_model_pytorch():
    """Create a simple PyTorch model for testing."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
            
        def forward(self, x):
            return self.fc(x)
    
    return SimpleModel()

@pytest.fixture
def sample_model_tensorflow():
    """Create a simple TensorFlow model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(10,))
    ])
    return model

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y 