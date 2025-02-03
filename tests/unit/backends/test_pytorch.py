import pytest
import torch
import numpy as np
from engine.backends.pytorch import PyTorchBackend, PyTorchTrainer

def test_pytorch_backend(sample_model_pytorch, temp_dir):
    """Test PyTorch backend."""
    backend = PyTorchBackend()
    
    # Test save/load model
    model_path = temp_dir / "model.pt"
    backend.save_model(sample_model_pytorch, str(model_path))
    loaded_model = backend.load_model(str(model_path))
    
    # Verify model structure
    assert isinstance(loaded_model, torch.nn.Module)
    assert len(list(loaded_model.parameters())) == len(list(sample_model_pytorch.parameters()))
    
    # Test predict
    input_tensor = torch.randn(1, 10)
    with torch.no_grad():
        output = backend.predict(loaded_model, input_tensor)
    assert output.shape == (1, 2)

def test_pytorch_trainer(sample_model_pytorch, sample_data):
    """Test PyTorch trainer."""
    trainer = PyTorchTrainer()
    X, y = sample_data
    
    # Create data loaders
    train_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y)
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Test training
    history = trainer.train(
        model=sample_model_pytorch,
        train_data=train_loader,
        epochs=2
    )
    
    assert 'train_loss' in history
    assert len(history['train_loss']) == 2
    
    # Test evaluation
    metrics = trainer.evaluate(
        model=sample_model_pytorch,
        test_data=train_loader
    )
    
    assert 'test_loss' in metrics
    assert 'accuracy' in metrics 