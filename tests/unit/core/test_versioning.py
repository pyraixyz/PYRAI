import pytest
from datetime import datetime
import numpy as np
from engine.core.versioning import ModelVersion, VersionManager

def test_model_version():
    """Test ModelVersion class."""
    version = ModelVersion(
        version_id="test_v1",
        model_id="model_1",
        description="Test model",
        created_at=datetime.utcnow(),
        metrics={"accuracy": 0.95},
        parameters={"lr": 0.001},
        tags=["test"]
    )
    
    # Test to_dict
    data = version.to_dict()
    assert data['version_id'] == "test_v1"
    assert data['model_id'] == "model_1"
    assert data['metrics']['accuracy'] == 0.95
    assert "test" in data['tags']
    
    # Test from_dict
    new_version = ModelVersion.from_dict(data)
    assert new_version.version_id == version.version_id
    assert new_version.metrics == version.metrics

def test_version_manager(temp_dir, sample_model_pytorch):
    """Test VersionManager class."""
    manager = VersionManager(str(temp_dir))
    
    # Test save version
    version = manager.save_version(
        model=sample_model_pytorch,
        model_id="test_model",
        description="Test model",
        metrics={"accuracy": 0.95},
        parameters={"lr": 0.001},
        tags=["test"]
    )
    
    assert version.model_id == "test_model"
    
    # Test load version
    loaded_model, loaded_version = manager.load_version(
        "test_model",
        version.version_id
    )
    assert isinstance(loaded_model, type(sample_model_pytorch))
    assert loaded_version.metrics == version.metrics
    
    # Test list versions
    versions = manager.list_versions("test_model")
    assert len(versions) == 1
    assert versions[0].version_id == version.version_id
    
    # Test delete version
    manager.delete_version("test_model", version.version_id)
    versions = manager.list_versions("test_model")
    assert len(versions) == 0 