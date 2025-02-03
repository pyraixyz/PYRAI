import pytest
import torch
import tensorflow as tf
from engine.core.optimizer import PyTorchOptimizer, TensorFlowOptimizer

def test_pytorch_optimizer(sample_model_pytorch):
    """Test PyTorch optimizer wrapper."""
    optimizer = PyTorchOptimizer(
        optimizer_class=torch.optim.Adam,
        params=sample_model_pytorch.parameters(),
        lr=0.001
    )
    
    # Test configuration
    config = optimizer.get_config()
    assert config['optimizer_class'] == 'Adam'
    assert config['lr'] == 0.001
    
    # Test state dict
    state = optimizer.state_dict()
    assert 'state' in state
    assert 'param_groups' in state
    
    # Test loading state
    optimizer.load_state_dict(state)

def test_tensorflow_optimizer():
    """Test TensorFlow optimizer wrapper."""
    optimizer = TensorFlowOptimizer(
        optimizer_class=tf.keras.optimizers.Adam,
        learning_rate=0.001
    )
    
    # Test configuration
    config = optimizer.get_config()
    assert config['optimizer_class'] == 'Adam'
    assert config['learning_rate'] == 0.001
    
    # Test weights
    weights = optimizer.state_dict()
    
    # Test loading weights
    optimizer.load_state_dict(weights) 