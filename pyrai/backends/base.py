from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def to_device(self, model: Any, device: str) -> Any:
        """Move model to device.
        
        Args:
            model: Model to move
            device: Target device
            
        Returns:
            Model on device
        """
        pass
    
    @abstractmethod
    def create_optimizer(
        self,
        model: Any,
        config: Dict[str, Any]
    ) -> Any:
        """Create optimizer.
        
        Args:
            model: Model to optimize
            config: Optimizer configuration
            
        Returns:
            Optimizer instance
        """
        pass
    
    @abstractmethod
    def create_scheduler(
        self,
        optimizer: Any,
        config: Dict[str, Any]
    ) -> Optional[Any]:
        """Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            config: Scheduler configuration
            
        Returns:
            Scheduler instance or None
        """
        pass
    
    @abstractmethod
    def create_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True
    ) -> Iterator:
        """Create data loader.
        
        Args:
            dataset: Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            Data loader iterator
        """
        pass
    
    @abstractmethod
    def train_step(
        self,
        model: Any,
        batch: Any,
        optimizer: Any
    ) -> float:
        """Perform single training step.
        
        Args:
            model: Model to train
            batch: Input batch
            optimizer: Optimizer instance
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def validation_step(
        self,
        model: Any,
        batch: Any
    ) -> float:
        """Perform single validation step.
        
        Args:
            model: Model to validate
            batch: Input batch
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def get_predictions(
        self,
        model: Any,
        batch: Any
    ) -> Any:
        """Get model predictions.
        
        Args:
            model: Model to use
            batch: Input batch
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def get_targets(self, batch: Any) -> Any:
        """Get target values from batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Target values
        """
        pass
    
    @abstractmethod
    def save_model(
        self,
        model: Any,
        path: str
    ) -> None:
        """Save model to disk.
        
        Args:
            model: Model to save
            path: Save path
        """
        pass
    
    @abstractmethod
    def load_model(
        self,
        path: str,
        device: Optional[str] = None
    ) -> Any:
        """Load model from disk.
        
        Args:
            path: Model path
            device: Target device
            
        Returns:
            Loaded model
        """
        pass 