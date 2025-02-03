from typing import Any, Dict, Optional, Iterator
import tensorflow as tf
from .base import ModelBackend

class TensorFlowBackend(ModelBackend):
    """TensorFlow model backend."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize TensorFlow backend."""
        self.device = device
        if device:
            tf.config.set_visible_devices(
                tf.config.list_physical_devices(device),
                device
            )
    
    def to_device(
        self,
        model: tf.keras.Model,
        device: str
    ) -> tf.keras.Model:
        """Move model to device."""
        with tf.device(device):
            return model
    
    def create_optimizer(
        self,
        model: tf.keras.Model,
        config: Dict[str, Any]
    ) -> tf.keras.optimizers.Optimizer:
        """Create optimizer."""
        name = config.get("name", "adam").lower()
        
        if name == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=config.get("learning_rate", 0.001)
            )
        elif name == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=config.get("learning_rate", 0.01),
                momentum=config.get("momentum", 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
    
    def create_scheduler(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        config: Dict[str, Any]
    ) -> Optional[tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Create learning rate scheduler."""
        if not config:
            return None
            
        name = config.get("name", "").lower()
        
        if name == "cosine":
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=config.get("initial_learning_rate", 0.001),
                decay_steps=config["decay_steps"]
            )
        elif name == "exponential":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=config.get("initial_learning_rate", 0.001),
                decay_steps=config["decay_steps"],
                decay_rate=config.get("decay_rate", 0.9)
            )
        else:
            return None
    
    def create_dataloader(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """Create data loader."""
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    @tf.function
    def train_step(
        self,
        model: tf.keras.Model,
        batch: Any,
        optimizer: tf.keras.optimizers.Optimizer
    ) -> float:
        """Perform training step."""
        inputs = self.get_inputs(batch)
        targets = self.get_targets(batch)
        
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = self._compute_loss(outputs, targets)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    @tf.function
    def validation_step(
        self,
        model: tf.keras.Model,
        batch: Any
    ) -> float:
        """Perform validation step."""
        inputs = self.get_inputs(batch)
        targets = self.get_targets(batch)
        
        outputs = model(inputs, training=False)
        loss = self._compute_loss(outputs, targets)
        
        return loss
    
    def get_predictions(
        self,
        model: tf.keras.Model,
        batch: Any
    ) -> tf.Tensor:
        """Get model predictions."""
        inputs = self.get_inputs(batch)
        return model(inputs, training=False)
    
    def get_targets(self, batch: Any) -> tf.Tensor:
        """Get target values."""
        if isinstance(batch, (tuple, list)):
            return batch[1]
        return batch.targets
    
    def get_inputs(self, batch: Any) -> tf.Tensor:
        """Get input values."""
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch.inputs
    
    def save_model(
        self,
        model: tf.keras.Model,
        path: str
    ) -> None:
        """Save model."""
        model.save(path)
    
    def load_model(
        self,
        path: str,
        device: Optional[str] = None
    ) -> tf.keras.Model:
        """Load model."""
        with tf.device(device or self.device or "/cpu:0"):
            return tf.keras.models.load_model(path)
    
    def _compute_loss(
        self,
        outputs: tf.Tensor,
        targets: tf.Tensor
    ) -> tf.Tensor:
        """Compute loss value."""
        if outputs.shape[-1] == 1:  # Binary classification
            return tf.keras.losses.binary_crossentropy(
                targets,
                outputs,
                from_logits=True
            )
        else:  # Multi-class classification
            return tf.keras.losses.sparse_categorical_crossentropy(
                targets,
                outputs,
                from_logits=True
            )
    
    def predict(
        self,
        model: tf.keras.Model,
        inputs: Any
    ) -> tf.Tensor:
        """Make predictions with TensorFlow model.
        
        Args:
            model: TensorFlow model
            inputs: Input data
            
        Returns:
            Model predictions
        """
        inputs = self._prepare_inputs(inputs)
        with tf.device(self.device):
            return model.predict(inputs)
    
    def prepare_training(
        self,
        model: tf.keras.Model,
        config: Dict[str, Any]
    ) -> None:
        """Prepare TensorFlow model for training.
        
        Args:
            model: TensorFlow model
            config: Training configuration
        """
        # Setup optimizer
        optimizer_config = config.get("optimizer", {})
        optimizer_type = optimizer_config.pop("type", "adam")
        
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(**optimizer_config)
        elif optimizer_type == "sgd":
            optimizer = tf.keras.optimizers.SGD(**optimizer_config)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup loss function
        loss_type = config.get("loss", "categorical_crossentropy")
        if loss_type == "categorical_crossentropy":
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
        elif loss_type == "mse":
            loss_fn = tf.keras.losses.MeanSquaredError()
        else:
            raise ValueError(f"Unsupported loss: {loss_type}")
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=config.get("metrics", ["accuracy"])
        )
    
    def train_epoch(
        self,
        model: tf.keras.Model,
        train_data: Any,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            model: TensorFlow model
            train_data: Training data
            config: Training configuration
            
        Returns:
            Training metrics
        """
        dataset = self._prepare_dataset(train_data, config)
        history = model.fit(
            dataset,
            epochs=1,
            verbose=0
        )
        return history.history
    
    def validate(
        self,
        model: tf.keras.Model,
        val_data: Any,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate TensorFlow model.
        
        Args:
            model: TensorFlow model
            val_data: Validation data
            config: Validation configuration
            
        Returns:
            Validation metrics
        """
        dataset = self._prepare_dataset(val_data, config)
        results = model.evaluate(dataset, verbose=0)
        return dict(zip(model.metrics_names, results))
    
    def _prepare_inputs(self, inputs: Any) -> tf.Tensor:
        """Prepare inputs for model."""
        if isinstance(inputs, tf.Tensor):
            return inputs
        elif isinstance(inputs, (list, tuple)):
            return tf.convert_to_tensor(inputs)
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
    
    def _prepare_dataset(
        self,
        data: Any,
        config: Dict[str, Any]
    ) -> tf.data.Dataset:
        """Prepare TensorFlow dataset."""
        if isinstance(data, tf.data.Dataset):
            return data
        
        batch_size = config.get("batch_size", 32)
        
        if isinstance(data, tuple) and len(data) == 2:
            dataset = tf.data.Dataset.from_tensor_slices(data)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((data,))
            
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE) 