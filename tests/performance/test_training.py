import pytest
import time
import torch
import numpy as np
from pyrai import Engine
from pyrai.backends import PyTorchBackend

@pytest.mark.performance
def test_training_performance():
    """Test training performance."""
    # Initialize engine
    engine = Engine(backend=PyTorchBackend())
    
    # Create large dataset
    batch_size = 128
    num_batches = 100
    input_size = 1000
    
    train_data = torch.randn(batch_size * num_batches, input_size)
    labels = torch.randint(0, 2, (batch_size * num_batches,))
    
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 2)
    )
    
    # Measure training time
    start_time = time.time()
    history = engine.train(
        model=model,
        train_data=train_data,
        labels=labels,
        batch_size=batch_size,
        epochs=5
    )
    end_time = time.time()
    
    training_time = end_time - start_time
    
    # Assert performance metrics
    assert training_time < 60  # Training should complete within 60 seconds
    assert history["loss"][-1] < history["loss"][0]  # Loss should decrease

@pytest.mark.performance
def test_inference_performance():
    """Test inference performance."""
    engine = Engine(backend=PyTorchBackend())
    
    # Create model and data
    input_size = 1000
    num_samples = 1000
    
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 2)
    )
    
    inputs = torch.randn(num_samples, input_size)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        predictions = model(inputs)
    end_time = time.time()
    
    inference_time = end_time - start_time
    
    # Assert performance metrics
    assert inference_time < 1  # Inference should complete within 1 second
    assert predictions.shape == (num_samples, 2) 