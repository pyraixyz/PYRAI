from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class Engine(ABC):
    """Base class for all PYRAI engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the engine."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the engine."""
        pass

class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def load_model(self, path: str) -> Any:
        """Load a model from path."""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, path: str) -> None:
        """Save a model to path."""
        pass
    
    @abstractmethod
    def predict(self, model: Any, inputs: Any) -> Any:
        """Make predictions using the model."""
        pass

class Trainer(ABC):
    """Abstract base class for model training."""
    
    @abstractmethod
    def train(self, 
              model: Any,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, Any]:
        """Train a model."""
        pass
    
    @abstractmethod
    def evaluate(self,
                model: Any,
                test_data: Any) -> Dict[str, Any]:
        """Evaluate a model."""
        pass 