from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import tensorflow as tf

class BaseOptimizer(ABC):
    """Base class for all optimizers."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        pass

class PyTorchOptimizer(BaseOptimizer):
    """PyTorch optimizer wrapper."""
    
    def __init__(self, 
                 optimizer_class: Any,
                 params: Any,
                 **kwargs):
        self.optimizer = optimizer_class(params, **kwargs)
        self.config = kwargs
        
    def get_config(self) -> Dict[str, Any]:
        return {
            'optimizer_class': self.optimizer.__class__.__name__,
            **self.config
        }
        
    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.state_dict()
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

class TensorFlowOptimizer(BaseOptimizer):
    """TensorFlow optimizer wrapper."""
    
    def __init__(self,
                 optimizer_class: Any,
                 **kwargs):
        self.optimizer = optimizer_class(**kwargs)
        self.config = kwargs
        
    def get_config(self) -> Dict[str, Any]:
        return {
            'optimizer_class': self.optimizer.__class__.__name__,
            **self.config
        }
        
    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.get_weights()
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.set_weights(state_dict) 