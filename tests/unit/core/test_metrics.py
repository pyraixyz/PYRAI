import pytest
import numpy as np
from engine.core.metrics import ClassificationMetrics, RegressionMetrics, MetricCollection

def test_classification_metrics():
    """Test classification metrics."""
    metrics = ClassificationMetrics(metric_type='accuracy')
    
    # Test with perfect predictions
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    metrics.update(y_true, y_pred)
    assert metrics.compute() == 1.0
    
    # Test with imperfect predictions
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    
    metrics.reset()
    metrics.update(y_true, y_pred)
    assert metrics.compute() == 0.75

def test_regression_metrics():
    """Test regression metrics."""
    metrics = RegressionMetrics(metric_type='mse')
    
    # Test with perfect predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    
    metrics.update(y_true, y_pred)
    assert metrics.compute() == 0.0
    
    # Test with imperfect predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1])
    
    metrics.reset()
    metrics.update(y_true, y_pred)
    assert np.isclose(metrics.compute(), 0.01)

def test_metric_collection():
    """Test metric collection."""
    metrics = {
        'accuracy': ClassificationMetrics(metric_type='accuracy'),
        'precision': ClassificationMetrics(metric_type='precision')
    }
    collection = MetricCollection(metrics)
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    
    collection.update(y_true, y_pred)
    results = collection.compute()
    
    assert 'accuracy' in results
    assert 'precision' in results
    assert results['accuracy'] == 0.75 