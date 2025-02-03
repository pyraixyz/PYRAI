import pytest
import torch
import numpy as np
from pyrai.metrics import Accuracy, Precision, MetricCollection

@pytest.fixture
def sample_data():
    """Create sample predictions and targets."""
    preds = torch.tensor([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1]
    ])
    targets = torch.tensor([1, 0, 1, 0])
    return preds, targets

def test_accuracy(sample_data):
    """Test accuracy metric."""
    preds, targets = sample_data
    metric = Accuracy()
    
    metric.update(preds, targets)
    accuracy = metric.compute()
    
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
    assert accuracy == 1.0  # All predictions are correct

def test_precision(sample_data):
    """Test precision metric."""
    preds, targets = sample_data
    metric = Precision(average="macro")
    
    metric.update(preds, targets)
    precision = metric.compute()
    
    assert isinstance(precision, float)
    assert 0 <= precision <= 1

def test_metric_collection(sample_data):
    """Test metric collection."""
    preds, targets = sample_data
    metrics = MetricCollection({
        "accuracy": Accuracy(),
        "precision": Precision(average="macro")
    })
    
    metrics.update(preds, targets)
    results = metrics.compute()
    
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "precision" in results
    assert all(0 <= v <= 1 for v in results.values())

def test_metric_reset(sample_data):
    """Test metric reset."""
    preds, targets = sample_data
    metric = Accuracy()
    
    metric.update(preds, targets)
    before_reset = metric.compute()
    
    metric.reset()
    metric.update(preds, targets)
    after_reset = metric.compute()
    
    assert before_reset == after_reset 