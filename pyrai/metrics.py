from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metric:
    """Base metric class."""
    
    def __init__(self):
        """Initialize metric."""
        self.reset()
        
    def update(self, preds: Any, targets: Any) -> None:
        """Update metric state.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        raise NotImplementedError
        
    def compute(self) -> float:
        """Compute metric value.
        
        Returns:
            Metric value
        """
        raise NotImplementedError
        
    def reset(self) -> None:
        """Reset metric state."""
        raise NotImplementedError

class Accuracy(Metric):
    """Accuracy metric."""
    
    def __init__(self):
        """Initialize accuracy metric."""
        super().__init__()
        
    def update(self, preds: Any, targets: Any) -> None:
        """Update accuracy state."""
        preds = self._convert_tensor(preds)
        targets = self._convert_tensor(targets)
        
        if len(preds.shape) > 1:
            preds = preds.argmax(axis=1)
            
        self.correct += (preds == targets).sum()
        self.total += len(targets)
        
    def compute(self) -> float:
        """Compute accuracy value."""
        return float(self.correct) / self.total if self.total > 0 else 0.0
        
    def reset(self) -> None:
        """Reset accuracy state."""
        self.correct = 0
        self.total = 0
        
    def _convert_tensor(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, (torch.Tensor, tf.Tensor)):
            return tensor.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor type: {type(tensor)}")

class Precision(Metric):
    """Precision metric."""
    
    def __init__(self, average: str = "macro"):
        """Initialize precision metric.
        
        Args:
            average: Averaging method
        """
        super().__init__()
        self.average = average
        
    def update(self, preds: Any, targets: Any) -> None:
        """Update precision state."""
        preds = self._convert_tensor(preds)
        targets = self._convert_tensor(targets)
        
        if len(preds.shape) > 1:
            preds = preds.argmax(axis=1)
            
        self.preds.extend(preds)
        self.targets.extend(targets)
        
    def compute(self) -> float:
        """Compute precision value."""
        return precision_score(
            self.targets,
            self.preds,
            average=self.average,
            zero_division=0
        )
        
    def reset(self) -> None:
        """Reset precision state."""
        self.preds = []
        self.targets = []
        
    def _convert_tensor(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, (torch.Tensor, tf.Tensor)):
            return tensor.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor type: {type(tensor)}")

class MetricCollection:
    """Collection of metrics."""
    
    def __init__(self, metrics: Dict[str, Metric]):
        """Initialize metric collection.
        
        Args:
            metrics: Dictionary of metrics
        """
        self.metrics = metrics
        
    def update(self, preds: Any, targets: Any) -> None:
        """Update all metrics."""
        for metric in self.metrics.values():
            metric.update(preds, targets)
            
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            name: metric.compute()
            for name, metric in self.metrics.items()
        }
        
    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset() 