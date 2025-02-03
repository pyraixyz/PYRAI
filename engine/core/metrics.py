from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

class BaseMetric(ABC):
    """Base class for all metrics."""
    
    @abstractmethod
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update metric with new predictions."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass

class ClassificationMetrics(BaseMetric):
    """Classification metrics implementation."""
    
    def __init__(self, metric_type: str = 'accuracy', average: str = 'macro'):
        self.metric_type = metric_type
        self.average = average
        self.y_true_list: List[np.ndarray] = []
        self.y_pred_list: List[np.ndarray] = []
        
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)
        
    def compute(self) -> float:
        y_true = np.concatenate(self.y_true_list)
        y_pred = np.concatenate(self.y_pred_list)
        
        if self.metric_type == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric_type == 'precision':
            return precision_score(y_true, y_pred, average=self.average)
        elif self.metric_type == 'recall':
            return recall_score(y_true, y_pred, average=self.average)
        elif self.metric_type == 'f1':
            return f1_score(y_true, y_pred, average=self.average)
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")
            
    def reset(self) -> None:
        self.y_true_list = []
        self.y_pred_list = []

class RegressionMetrics(BaseMetric):
    """Regression metrics implementation."""
    
    def __init__(self, metric_type: str = 'mse'):
        self.metric_type = metric_type
        self.y_true_list: List[np.ndarray] = []
        self.y_pred_list: List[np.ndarray] = []
        
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)
        
    def compute(self) -> float:
        y_true = np.concatenate(self.y_true_list)
        y_pred = np.concatenate(self.y_pred_list)
        
        if self.metric_type == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif self.metric_type == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric_type == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.metric_type == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")
            
    def reset(self) -> None:
        self.y_true_list = []
        self.y_pred_list = []

class MetricCollection:
    """Collection of metrics."""
    
    def __init__(self, metrics: Dict[str, BaseMetric]):
        self.metrics = metrics
        
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        for metric in self.metrics.values():
            metric.update(y_true, y_pred)
            
    def compute(self) -> Dict[str, float]:
        return {
            name: metric.compute()
            for name, metric in self.metrics.items()
        }
        
    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset() 