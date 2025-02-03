import tensorflow as tf
from typing import Any, Dict, Optional
from ..core.base import ModelBackend, Trainer

class TensorFlowBackend(ModelBackend):
    """TensorFlow implementation of ModelBackend."""
    
    def load_model(self, path: str) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def save_model(self, model: tf.keras.Model, path: str) -> None:
        model.save(path)
    
    def predict(self, model: tf.keras.Model, inputs: tf.Tensor) -> tf.Tensor:
        return model.predict(inputs)

class TensorFlowTrainer(Trainer):
    """TensorFlow implementation of Trainer."""
    
    def __init__(self,
                 optimizer_cls: Any = tf.keras.optimizers.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 strategy: Optional[tf.distribute.Strategy] = None):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.strategy = strategy or tf.distribute.get_strategy()
    
    def train(self,
              model: tf.keras.Model,
              train_data: tf.data.Dataset,
              valid_data: Optional[tf.data.Dataset] = None,
              epochs: int = 10,
              loss_fn: Any = tf.keras.losses.SparseCategoricalCrossentropy(),
              **kwargs) -> Dict[str, Any]:
        
        with self.strategy.scope():
            optimizer = self.optimizer_cls(**self.optimizer_kwargs)
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
            
        history = model.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            **kwargs
        )
        
        return history.history
    
    def evaluate(self,
                model: tf.keras.Model,
                test_data: tf.data.Dataset) -> Dict[str, Any]:
        results = model.evaluate(test_data, return_dict=True)
        return results 