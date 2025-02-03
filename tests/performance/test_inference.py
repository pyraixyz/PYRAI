import pytest
import time
import torch
import tensorflow as tf
import numpy as np
from engine.backends.pytorch import PyTorchBackend
from engine.backends.tensorflow import TensorFlowBackend
from engine.monitoring.profiler import ModelProfiler

@pytest.mark.performance
def test_pytorch_inference_performance(temp_dir):
    """Test PyTorch inference performance."""
    profiler = ModelProfiler(str(temp_dir))
    backend = PyTorchBackend()
    
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    # Create sample input
    sample_input = torch.randn(1, 100)
    
    # Profile inference
    results = profiler.profile_inference(
        model=model,
        backend=backend,
        sample_input=sample_input,
        batch_sizes=[1, 8, 32, 128],
        num_iterations=100
    )
    
    # Verify performance metrics
    assert 'latency' in results
    assert 'throughput' in results
    assert 'memory' in results
    
    # Check performance thresholds
    assert results['latency'][1]['mean'] < 0.1  # 100ms max latency
    assert results['throughput'][32] > 100  # Min 100 samples/sec at batch 32

@pytest.mark.performance
def test_memory_profiling(temp_dir):
    """Test memory profiling."""
    profiler = ModelProfiler(str(temp_dir))
    
    # Profile PyTorch model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 10)
    )
    
    sample_input = torch.randn(1, 100)
    memory_profile = profiler.profile_memory(
        model=model,
        backend=PyTorchBackend(),
        sample_input=sample_input
    )
    
    assert 'pytorch' in memory_profile
    assert 'memory_by_operator' in memory_profile['pytorch'] 