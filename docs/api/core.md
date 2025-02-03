# Core API Reference

## Engine

The main class that coordinates all PyRAI functionality.

### Engine Class

```python
class Engine:
    def __init__(
        self, 
        backend: ModelBackend,
        config: Optional[Dict] = None
    ):
        """Initialize PyRAI engine.
        
        Args:
            backend: Model backend (PyTorch/TensorFlow)
            config: Engine configuration
        """
        pass
        
    def train(
        self,
        model: Any,
        train_data: Any,
        validation_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """Train a model.
        
        Args:
            model: Model to train
            train_data: Training data
            validation_data: Validation data
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        pass
        
    def deploy(
        self,
        model_id: str,
        version: str,
        config: Optional[Dict] = None
    ) -> Deployment:
        """Deploy a model.
        
        Args:
            model_id: Model identifier
            version: Model version
            config: Deployment configuration
            
        Returns:
            Deployment object
        """
        pass
```

### ModelBackend Class

```python
class ModelBackend(ABC):
    @abstractmethod
    def save_model(
        self,
        model: Any,
        path: str
    ) -> None:
        """Save model to disk."""
        pass
        
    @abstractmethod
    def load_model(
        self,
        path: str
    ) -> Any:
        """Load model from disk."""
        pass
        
    @abstractmethod
    def predict(
        self,
        model: Any,
        inputs: Any
    ) -> Any:
        """Make predictions."""
        pass
```

## Trainer

Handles model training functionality.

### Trainer Class

```python
class Trainer:
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Any
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer instance
            loss_fn: Loss function
        """
        pass
        
    def train_step(
        self,
        batch: Any
    ) -> float:
        """Execute single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Loss value
        """
        pass
        
    def validate(
        self,
        val_data: Any
    ) -> Dict[str, float]:
        """Validate model.
        
        Args:
            val_data: Validation data
            
        Returns:
            Validation metrics
        """
        pass
```

## Configuration

### Config Class

```python
class Config:
    def __init__(
        self,
        config_dict: Dict
    ):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        pass
        
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        pass
        
    def update(
        self,
        updates: Dict
    ) -> None:
        """Update configuration.
        
        Args:
            updates: Configuration updates
        """
        pass
``` 