from typing import Any, Dict, Optional, Union
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class SchedulerWrapper:
    """Base scheduler wrapper."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        **kwargs
    ):
        """Initialize scheduler.
        
        Args:
            optimizer: Optimizer instance
            **kwargs: Additional scheduler parameters
        """
        self.optimizer = optimizer
        self.kwargs = kwargs
        self._init_scheduler()
        
    def _init_scheduler(self) -> None:
        """Initialize underlying scheduler."""
        raise NotImplementedError
        
    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate."""
        self.scheduler.step(metrics)
        
    def state_dict(self) -> Dict:
        """Get scheduler state."""
        return self.scheduler.state_dict()
        
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)

class CosineAnnealingLR(SchedulerWrapper):
    """Cosine annealing scheduler."""
    
    def _init_scheduler(self) -> None:
        """Initialize cosine scheduler."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            **self.kwargs
        )

class ReduceLROnPlateau(SchedulerWrapper):
    """Reduce LR on plateau scheduler."""
    
    def _init_scheduler(self) -> None:
        """Initialize plateau scheduler."""
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **self.kwargs
        )

class OneCycleLR(SchedulerWrapper):
    """One cycle learning rate scheduler."""
    
    def _init_scheduler(self) -> None:
        """Initialize one cycle scheduler."""
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            **self.kwargs
        )

class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total number of steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Get learning rates."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            alpha = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * alpha for base_lr in self.base_lrs]
            
        # Cosine decay
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * alpha
            for base_lr in self.base_lrs
        ]

def create_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs
) -> SchedulerWrapper:
    """Create scheduler instance.
    
    Args:
        name: Scheduler name
        optimizer: Optimizer instance
        **kwargs: Scheduler parameters
        
    Returns:
        Scheduler instance
    """
    schedulers = {
        'cosine': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'one_cycle': OneCycleLR,
        'warmup': WarmupScheduler
    }
    
    if name not in schedulers:
        raise ValueError(f"Unsupported scheduler: {name}")
        
    return schedulers[name](optimizer, **kwargs) 