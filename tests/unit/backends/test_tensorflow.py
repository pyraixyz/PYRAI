import pytest
import tensorflow as tf
import numpy as np
from engine.backends.tensorflow import TensorFlowBackend, TensorFlowTrainer

def test_tensorflow_backend(sample_model_tensorflow, temp_dir):
    """Test TensorFlow backend."""
    backend = TensorFlowBackend()
    
    # Test save/load model
    model_path = temp_dir / "model"
    backend.save_model(sample_model_tensorflow, str(model_path))
    loaded_model = backend.load_model(str(model_path))
    
    # Verify model structure
    assert isinstance(loaded_model, tf.keras.Model)
    assert len(loaded_model.weights) == len(sample_model_tensorflow.weights)
    
    # Test predict
    input_tensor = tf.random.normal((1, 10))
    output = backend.predict(loaded_model, input_tensor)
    assert output.shape == (1, 2)

def test_tensorflow_trainer(sample_model_tensorflow, sample_data):
    """Test TensorFlow trainer."""
    trainer = TensorFlowTrainer()
    X, y = sample_data
    
    # Create tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((X, y))\
        .batch(32)
    
    # Test training
    history = trainer.train(
        model=sample_model_tensorflow,
        train_data=train_data,
        epochs=2
    )
    
    assert 'loss' in history.history
    assert len(history.history['loss']) == 2
    
    # Test evaluation
    metrics = trainer.evaluate(
        model=sample_model_tensorflow,
        test_data=train_data
    )
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics 