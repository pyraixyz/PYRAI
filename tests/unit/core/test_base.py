import pytest
import numpy as np
from engine.core.base import Engine, ModelBackend, Trainer

def test_engine_abstract():
    """Test Engine abstract class cannot be instantiated."""
    with pytest.raises(TypeError):
        Engine({})

def test_model_backend_abstract():
    """Test ModelBackend abstract class cannot be instantiated."""
    with pytest.raises(TypeError):
        ModelBackend()

def test_trainer_abstract():
    """Test Trainer abstract class cannot be instantiated."""
    with pytest.raises(TypeError):
        Trainer()

class TestEngine(Engine):
    """Concrete Engine implementation for testing."""
    def initialize(self):
        self._initialized = True
    
    def shutdown(self):
        self._initialized = False

def test_engine_initialization():
    """Test Engine initialization."""
    config = {"test": "config"}
    engine = TestEngine(config)
    
    assert not engine._initialized
    assert engine.config == config
    
    engine.initialize()
    assert engine._initialized
    
    engine.shutdown()
    assert not engine._initialized 