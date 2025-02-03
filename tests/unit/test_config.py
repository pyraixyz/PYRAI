import pytest
from pyrai.config import Config

def test_config_initialization():
    """Test configuration initialization."""
    config_dict = {
        "training": {
            "batch_size": 32,
            "epochs": 10
        },
        "model": {
            "type": "pytorch",
            "device": "cuda"
        }
    }
    
    config = Config(config_dict)
    assert config.get("training.batch_size") == 32
    assert config.get("model.device") == "cuda"

def test_config_default_values():
    """Test configuration default values."""
    config = Config({})
    
    assert config.get("training.batch_size", 64) == 64
    assert config.get("nonexistent.key", "default") == "default"

def test_config_update():
    """Test configuration updates."""
    config = Config({
        "training": {
            "batch_size": 32
        }
    })
    
    config.update({
        "training": {
            "batch_size": 64,
            "epochs": 20
        }
    })
    
    assert config.get("training.batch_size") == 64
    assert config.get("training.epochs") == 20

def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        Config({
            "training": {
                "batch_size": "invalid"  # Should be int
            }
        })
    
    with pytest.raises(ValueError):
        Config({
            "model": {
                "device": "invalid_device"  # Should be cuda or cpu
            }
        })

def test_config_export():
    """Test configuration export."""
    config_dict = {
        "training": {
            "batch_size": 32,
            "epochs": 10
        }
    }
    
    config = Config(config_dict)
    exported = config.to_dict()
    
    assert exported == config_dict 