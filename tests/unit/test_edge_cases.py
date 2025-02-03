import pytest
import numpy as np
import torch
import tensorflow as tf
from engine.core.preprocessing import NumericPreprocessor
from engine.core.metrics import ClassificationMetrics
from engine.backends.pytorch import PyTorchBackend
from engine.backends.tensorflow import TensorFlowBackend

def test_preprocessing_empty_data():
    """Test preprocessing with empty data."""
    preprocessor = NumericPreprocessor()
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        preprocessor.fit(empty_df)

def test_preprocessing_all_missing():
    """Test preprocessing with all missing values."""
    data = pd.DataFrame({
        'a': [np.nan, np.nan, np.nan],
        'b': [np.nan, np.nan, np.nan]
    })
    preprocessor = NumericPreprocessor(handle_missing='mean')
    
    with pytest.raises(ValueError):
        preprocessor.fit(data)

def test_metrics_empty_predictions():
    """Test metrics with empty predictions."""
    metrics = ClassificationMetrics()
    
    with pytest.raises(ValueError):
        metrics.update(np.array([]), np.array([]))

def test_model_invalid_input():
    """Test model prediction with invalid input."""
    backend = PyTorchBackend()
    model = torch.nn.Linear(10, 2)
    
    with pytest.raises(ValueError):
        backend.predict(model, torch.randn(1, 5))  # Wrong input size

def test_large_batch_size():
    """Test training with very large batch size."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # PyTorch
    train_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y)
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1000  # Larger than dataset
    )
    
    model = torch.nn.Linear(10, 2)
    trainer = PyTorchTrainer()
    
    # Should handle this gracefully
    history = trainer.train(model, train_loader, epochs=1)
    assert 'train_loss' in history 