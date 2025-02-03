import pytest
import torch
import tensorflow as tf
import numpy as np
from engine.core.versioning import VersionManager
from engine.core.preprocessing import PreprocessingPipeline
from engine.backends.pytorch import PyTorchBackend, PyTorchTrainer
from engine.monitoring.logger import ModelLogger

@pytest.mark.integration
def test_end_to_end_training(temp_dir):
    """Test end-to-end training pipeline."""
    # Setup components
    version_manager = VersionManager(str(temp_dir / 'models'))
    backend = PyTorchBackend()
    trainer = PyTorchTrainer()
    logger = ModelLogger(str(temp_dir / 'logs'), 'test-service')
    
    # Create sample data
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Create and preprocess dataset
    train_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[:800]),
        torch.LongTensor(y[:800])
    )
    valid_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[800:]),
        torch.LongTensor(y[800:])
    )
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    )
    
    # Train model
    history = trainer.train(
        model=model,
        train_data=train_loader,
        valid_data=valid_loader,
        epochs=5
    )
    
    # Log training results
    logger.log_training(
        model_id="test_model",
        metrics={"loss": history['train_loss'][-1]},
        parameters={"lr": 0.001},
        duration=100.0
    )
    
    # Save model version
    version = version_manager.save_version(
        model=model,
        model_id="test_model",
        description="Integration test model",
        metrics={"loss": history['train_loss'][-1]},
        parameters={"lr": 0.001}
    )
    
    # Load and verify model
    loaded_model, _ = version_manager.load_version(
        "test_model",
        version.version_id
    )
    
    # Make predictions
    X_test = torch.randn(10, 10)
    with torch.no_grad():
        original_pred = model(X_test)
        loaded_pred = loaded_model(X_test)
    
    assert torch.allclose(original_pred, loaded_pred) 