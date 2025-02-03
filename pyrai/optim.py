from typing import Any, Dict, Iterator, Optional, Union
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.nn.parameter import Parameter

class OptimizerWrapper:
    """Base optimizer wrapper."""
    
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-3,
        **kwargs
    ):
        """Initialize optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            **kwargs: Additional optimizer parameters
        """
        self.params = list(params)
        self.lr = lr
        self.kwargs = kwargs
        self._init_optimizer()
        
    def _init_optimizer(self) -> None:
        """Initialize underlying optimizer."""
        raise NotImplementedError
        
    def zero_grad(self) -> None:
        """Clear gradients."""
        self.optimizer.zero_grad()
        
    def step(self) -> None:
        """Update parameters."""
        self.optimizer.step()
        
    def state_dict(self) -> Dict:
        """Get optimizer state."""
        return self.optimizer.state_dict()
        
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)

class Adam(OptimizerWrapper):
    """Adam optimizer wrapper."""
    
    def _init_optimizer(self) -> None:
        """Initialize Adam optimizer."""
        self.optimizer = optim.Adam(
            self.params,
            lr=self.lr,
            **self.kwargs
        )

class SGD(OptimizerWrapper):
    """SGD optimizer wrapper."""
    
    def _init_optimizer(self) -> None:
        """Initialize SGD optimizer."""
        self.optimizer = optim.SGD(
            self.params,
            lr=self.lr,
            **self.kwargs
        )

class AdamW(OptimizerWrapper):
    """AdamW optimizer wrapper."""
    
    def _init_optimizer(self) -> None:
        """Initialize AdamW optimizer."""
        self.optimizer = optim.AdamW(
            self.params,
            lr=self.lr,
            **self.kwargs
        )

def create_optimizer(
    name: str,
    params: Iterator[Parameter],
    **kwargs
) -> OptimizerWrapper:
    """Create optimizer instance.
    
    Args:
        name: Optimizer name
        params: Model parameters
        **kwargs: Optimizer parameters
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': Adam,
        'sgd': SGD,
        'adamw': AdamW
    }
    
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}")
        
    return optimizers[name](params, **kwargs) 