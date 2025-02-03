# Backend API Reference

## PyTorch Backend

### PyTorchBackend Class

```python
from typing import Any, Dict, Optional
import torch
from pyrai.core import ModelBackend

class PyTorchBackend(ModelBackend):
    def __init__(
        self,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize PyTorch backend.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            **kwargs: Additional backend options
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def save_model(
        self,
        model: torch.nn.Module,
        path: str
    ) -> None:
        """Save PyTorch model.
        
        Args:
            model: PyTorch model to save
            path: Save path
        """
        torch.save(model.state_dict(), path)
        
    def load_model(
        self,
        path: str,
        model_cls: Optional[type] = None
    ) -> torch.nn.Module:
        """Load PyTorch model.
        
        Args:
            path: Path to saved model
            model_cls: Optional model class for initialization
            
        Returns:
            Loaded PyTorch model
        """
        pass
        
    def predict(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Make predictions with PyTorch model.
        
        Args:
            model: PyTorch model
            inputs: Input tensor
            
        Returns:
            Model predictions
        """
        pass
```

### PyTorchTrainer Class

```python
class PyTorchTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Any,
        device: str = 'cuda'
    ):
        """Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            device: Device to use
        """
        pass
        
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics
        """
        pass
```

## TensorFlow Backend

### TensorFlowBackend Class

```python
import tensorflow as tf
from pyrai.core import ModelBackend

class TensorFlowBackend(ModelBackend):
    def __init__(
        self,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize TensorFlow backend.
        
        Args:
            device: Device to use ('/GPU:0' or '/CPU:0')
            **kwargs: Additional backend options
        """
        pass
        
    def save_model(
        self,
        model: tf.keras.Model,
        path: str
    ) -> None:
        """Save TensorFlow model.
        
        Args:
            model: TensorFlow model to save
            path: Save path
        """
        model.save(path)
        
    def load_model(
        self,
        path: str
    ) -> tf.keras.Model:
        """Load TensorFlow model.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded TensorFlow model
        """
        pass
```

## Custom Backend Development

To implement a custom backend:

```python
from pyrai.core import ModelBackend
from typing import Any

class CustomBackend(ModelBackend):
    def __init__(self, **kwargs):
        """Initialize custom backend."""
        super().__init__()
        
    def save_model(
        self,
        model: Any,
        path: str
    ) -> None:
        """Implement model saving."""
        pass
        
    def load_model(
        self,
        path: str
    ) -> Any:
        """Implement model loading."""
        pass
        
    def predict(
        self,
        model: Any,
        inputs: Any
    ) -> Any:
        """Implement prediction."""
        pass
``` 