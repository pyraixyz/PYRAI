from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from .parallel import ParallelStrategy
from .strategy import ModelProfile
from .monitor import RuntimeStats
from .profiler import ProfileResult

@dataclass
class PerformancePrediction:
    """Performance prediction results."""
    throughput: float  # Samples/second
    latency: float    # Seconds
    memory_usage: float  # Bytes
    communication_overhead: float  # Seconds
    device_utilization: List[float]  # Per-device utilization

@dataclass
class PredictionResult:
    """Performance prediction result."""
    metric: str
    value: float
    confidence: float
    trend: str
    factors: Dict[str, float]

class PerformancePredictor:
    """Predicts future performance trends."""
    
    def __init__(
        self,
        history_window: int = 100,
        prediction_horizon: int = 10,
        min_samples: int = 20
    ):
        """Initialize predictor.
        
        Args:
            history_window: Window size for training
            prediction_horizon: Steps to predict ahead
            min_samples: Minimum samples for prediction
        """
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        self.min_samples = min_samples
        self.history: List[RuntimeStats] = []
        self.models: Dict[str, GradientBoostingRegressor] = {}
        
    def update(self, stats: RuntimeStats):
        """Update prediction models with new data.
        
        Args:
            stats: Current runtime statistics
        """
        self.history.append(stats)
        if len(self.history) > self.history_window:
            self.history.pop(0)
            
        if len(self.history) >= self.min_samples:
            self._train_models()
            
    def predict(
        self,
        metric: str,
        steps: Optional[int] = None
    ) -> Optional[PredictionResult]:
        """Predict future performance.
        
        Args:
            metric: Metric to predict
            steps: Optional number of steps ahead
            
        Returns:
            Prediction result if available
        """
        if len(self.history) < self.min_samples:
            return None
            
        if metric not in self.models:
            return None
            
        steps = steps or self.prediction_horizon
        model = self.models[metric]
        
        # Get latest features
        features = self._extract_features(self.history[-1])
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get feature importance
        importance = dict(zip(
            self._get_feature_names(),
            model.feature_importances_
        ))
        
        # Calculate confidence
        confidence = 1.0 - model.loss_(
            model.predict([features])[0],
            model.predict_proba([features])[0]
        )
        
        # Determine trend
        current_value = self._get_metric_value(self.history[-1], metric)
        trend = "stable"
        if prediction > current_value * 1.1:
            trend = "increasing"
        elif prediction < current_value * 0.9:
            trend = "decreasing"
            
        return PredictionResult(
            metric=metric,
            value=prediction,
            confidence=confidence,
            trend=trend,
            factors=importance
        )
        
    def _train_models(self):
        """Train prediction models."""
        metrics = [
            "throughput",
            "latency",
            "memory_utilization",
            "compute_utilization",
            "communication_overhead"
        ]
        
        X = []
        y = {metric: [] for metric in metrics}
        
        # Prepare training data
        for i in range(len(self.history) - 1):
            features = self._extract_features(self.history[i])
            X.append(features)
            
            for metric in metrics:
                target = self._get_metric_value(self.history[i+1], metric)
                y[metric].append(target)
                
        # Train models
        for metric in metrics:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
            model.fit(X, y[metric])
            self.models[metric] = model
            
    def _extract_features(self, stats: RuntimeStats) -> List[float]:
        """Extract prediction features."""
        features = [
            np.mean(stats.gpu_memory),
            np.mean(stats.gpu_utilization),
            stats.throughput,
            stats.latency,
            stats.communication_overhead,
            stats.data_loading_time,
            stats.compute_time
        ]
        return features
        
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            "memory_utilization",
            "compute_utilization",
            "throughput",
            "latency",
            "communication_overhead",
            "data_loading_time",
            "compute_time"
        ]
        
    def _get_metric_value(self, stats: RuntimeStats, metric: str) -> float:
        """Get metric value from stats."""
        if metric == "throughput":
            return stats.throughput
        elif metric == "latency":
            return stats.latency
        elif metric == "memory_utilization":
            return np.mean(stats.gpu_memory)
        elif metric == "compute_utilization":
            return np.mean(stats.gpu_utilization)
        elif metric == "communication_overhead":
            return stats.communication_overhead
        else:
            return 0.0

class MLPredictor:
    """Machine learning based performance predictor."""
    
    def __init__(
        self,
        target_metric: str = "throughput",
        history_window: int = 100,
        retrain_interval: int = 1000
    ):
        """Initialize ML predictor.
        
        Args:
            target_metric: Metric to predict
            history_window: Window size for prediction
            retrain_interval: Steps between model retraining
        """
        self.target_metric = target_metric
        self.history_window = history_window
        self.retrain_interval = retrain_interval
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.steps = 0
        self.feature_names: List[str] = []
        
    def update(self, stats: RuntimeStats) -> Optional[PredictionResult]:
        """Update predictor with new statistics."""
        # Extract features
        features = self._extract_features(stats)
        if not self.feature_names:
            self.feature_names = list(features.keys())
            
        # Update history
        X = np.array(list(features.values())).reshape(1, -1)
        y = np.array([getattr(stats, self.target_metric)])
        
        if not hasattr(self, 'X_history'):
            self.X_history = X
            self.y_history = y
        else:
            self.X_history = np.vstack([self.X_history, X])[-self.history_window:]
            self.y_history = np.append(self.y_history, y)[-self.history_window:]
            
        # Retrain model periodically
        self.steps += 1
        if self.steps % self.retrain_interval == 0:
            self._train_model()
            
        # Make prediction if model is trained
        if hasattr(self, 'trained') and self.trained:
            return self.predict(stats)
            
        return None
        
    def predict(self, stats: RuntimeStats) -> PredictionResult:
        """Make performance prediction."""
        features = self._extract_features(stats)
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and confidence
        predictions = []
        for estimator in self.model.estimators_:
            predictions.append(estimator.predict(X_scaled)[0])
            
        predicted_value = np.mean(predictions)
        confidence = 1.0 - (np.std(predictions) / predicted_value)
        
        # Get feature importance
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        return PredictionResult(
            metric=self.target_metric,
            value=predicted_value,
            confidence=confidence,
            trend="stable",
            factors=importance
        )
        
    def _extract_features(self, stats: RuntimeStats) -> Dict[str, float]:
        """Extract features from statistics."""
        return {
            "avg_gpu_util": np.mean(stats.gpu_utilization),
            "max_gpu_util": np.max(stats.gpu_utilization),
            "avg_gpu_mem": np.mean(stats.gpu_memory),
            "max_gpu_mem": np.max(stats.gpu_memory),
            "cpu_util": stats.cpu_utilization,
            "sys_mem": stats.system_memory,
            "latency": stats.latency
        }
        
    def _train_model(self):
        """Train prediction model."""
        if len(self.X_history) < self.history_window // 2:
            return
            
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X_history)
        
        # Train model
        self.model.fit(X_scaled, self.y_history)
        self.trained = True 