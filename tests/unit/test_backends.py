import pytest
import torch
import tensorflow as tf
from pyrai.backends import PyTorchBackend, TensorFlowBackend

class TestPyTorchBackend:
    @pytest.fixture
    def backend(self):
        return PyTorchBackend()
    
    @pytest.fixture
    def model(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        return SimpleModel()
    
    def test_save_load_model(self, backend, model, tmp_path):
        """Test model saving and loading."""
        save_path = tmp_path / "model.pt"
        
        # Save model
        backend.save_model(model, str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = backend.load_model(str(save_path))
        assert isinstance(loaded_model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 10)
        with torch.no_grad():
            original_output = model(x)
            loaded_output = loaded_model(x)
        
        assert torch.allclose(original_output, loaded_output)
    
    def test_predict(self, backend, model):
        """Test model prediction."""
        inputs = torch.randn(1, 10)
        predictions = backend.predict(model, inputs)
        
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (1, 2)

class TestTensorFlowBackend:
    @pytest.fixture
    def backend(self):
        return TensorFlowBackend()
    
    @pytest.fixture
    def model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(10,))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def test_save_load_model(self, backend, model, tmp_path):
        """Test model saving and loading."""
        save_path = tmp_path / "model"
        
        # Save model
        backend.save_model(model, str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = backend.load_model(str(save_path))
        assert isinstance(loaded_model, tf.keras.Model)
        
        # Test prediction
        x = tf.random.normal((1, 10))
        original_output = model.predict(x)
        loaded_output = loaded_model.predict(x)
        
        assert tf.reduce_all(tf.equal(original_output, loaded_output))
    
    def test_predict(self, backend, model):
        """Test model prediction."""
        inputs = tf.random.normal((1, 10))
        predictions = backend.predict(model, inputs)
        
        assert isinstance(predictions, tf.Tensor)
        assert predictions.shape == (1, 2) 