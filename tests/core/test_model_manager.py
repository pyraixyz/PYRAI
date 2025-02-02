"""
Tests for the Model Manager implementation.
"""

import pytest
import torch
import tensorflow as tf
from pathlib import Path
import tempfile
import json

from src.core.model_manager import ModelManager
from src.core.storage_manager import StorageManager
from src.core.exceptions import ModelError, ModelNotFoundError


class SimpleModel(torch.nn.Module):
    """Simple PyTorch model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)


def create_tf_model():
    """Create a simple TensorFlow model for testing"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(10,))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


@pytest.fixture
async def model_manager():
    """Create a temporary model manager for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = StorageManager(storage_path=temp_dir)
        manager = ModelManager(storage_manager=storage)
        yield manager


@pytest.mark.asyncio
async def test_save_and_load_pytorch(model_manager):
    """Test saving and loading PyTorch models"""
    model_id = "test_pytorch"
    model = SimpleModel()
    
    # Save model
    assert await model_manager.save(model_id, model)
    
    # Load model
    loaded_model = await model_manager.load(model_id)
    assert isinstance(loaded_model, torch.nn.Module)
    assert type(loaded_model) == type(model)


@pytest.mark.asyncio
async def test_save_and_load_tensorflow(model_manager):
    """Test saving and loading TensorFlow models"""
    model_id = "test_tensorflow"
    model = create_tf_model()
    
    # Save model
    assert await model_manager.save(model_id, model)
    
    # Load model
    loaded_model = await model_manager.load(model_id)
    assert isinstance(loaded_model, tf.keras.Model)


@pytest.mark.asyncio
async def test_versioning(model_manager):
    """Test model versioning"""
    model_id = "test_versions"
    model = SimpleModel()
    
    # Save multiple versions
    for _ in range(3):
        assert await model_manager.save(model_id, model)
    
    # Check metadata
    info = await model_manager.get_model_info(model_id)
    assert info is not None
    assert len(info["versions"]) == 3
    
    # Load specific version
    loaded_model = await model_manager.load(model_id, version=2)
    assert isinstance(loaded_model, torch.nn.Module)


@pytest.mark.asyncio
async def test_delete_model(model_manager):
    """Test model deletion"""
    model_id = "test_delete"
    model = SimpleModel()
    
    # Save model
    await model_manager.save(model_id, model)
    
    # Delete model
    assert await model_manager.delete(model_id)
    
    # Verify deletion
    with pytest.raises(ModelNotFoundError):
        await model_manager.load(model_id)


@pytest.mark.asyncio
async def test_list_models(model_manager):
    """Test listing models"""
    models = {
        "model1": SimpleModel(),
        "model2": create_tf_model()
    }
    
    # Save multiple models
    for model_id, model in models.items():
        await model_manager.save(model_id, model)
    
    # List models
    model_list = await model_manager.list_models()
    assert len(model_list) == 2
    assert all(m["id"] in models for m in model_list)


@pytest.mark.asyncio
async def test_model_info(model_manager):
    """Test getting model information"""
    model_id = "test_info"
    model = SimpleModel()
    
    # Save model
    await model_manager.save(model_id, model)
    
    # Get info
    info = await model_manager.get_model_info(model_id)
    assert info is not None
    assert "versions" in info
    assert len(info["versions"]) == 1
    assert info["versions"][0]["type"] == "pytorch"


@pytest.mark.asyncio
async def test_invalid_model_type(model_manager):
    """Test handling invalid model types"""
    model_id = "test_invalid"
    invalid_model = "not a real model"
    
    with pytest.raises(ModelError):
        await model_manager.save(model_id, invalid_model) 