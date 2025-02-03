from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import dataclass
from .resources import ResourceType, ResourceAllocation
from .predictor import MLPredictor

@dataclass
class ResourcePrediction:
    """Resource usage prediction."""
    resource_type: ResourceType
    device_id: int
    predicted_usage: float
    confidence: float
    time_horizon: float  # Prediction horizon in seconds

class ResourcePredictor:
    """Predicts future resource usage."""
    
    def __init__(
        self,
        num_devices: int,
        history_window: int = 100,
        prediction_horizons: List[float] = [1.0, 5.0, 15.0]  # seconds
    ):
        """Initialize resource predictor.
        
        Args:
            num_devices: Number of devices
            history_window: History window size
            prediction_horizons: Time horizons for predictions
        """
        self.num_devices = num_devices
        self.history_window = history_window
        self.prediction_horizons = prediction_horizons
        
        # Create predictors for each resource type and device
        self.predictors: Dict[Tuple[ResourceType, int], Dict[float, MLPredictor]] = {}
        for resource_type in ResourceType:
            for device_id in range(num_devices):
                self.predictors[(resource_type, device_id)] = {
                    horizon: MLPredictor(f"{resource_type.value}_usage")
                    for horizon in prediction_horizons
                }
                
        self.history: Dict[Tuple[ResourceType, int], List[Tuple[float, float]]] = {}
        
    def update(
        self,
        resource_type: ResourceType,
        device_id: int,
        usage: float,
        timestamp: float
    ):
        """Update resource usage history."""
        key = (resource_type, device_id)
        if key not in self.history:
            self.history[key] = []
            
        self.history[key].append((timestamp, usage))
        if len(self.history[key]) > self.history_window:
            self.history[key].pop(0)
            
        # Update predictors
        for horizon, predictor in self.predictors[key].items():
            features = self._extract_features(key, timestamp)
            if features:
                predictor.update(features)
                
    def predict(
        self,
        resource_type: ResourceType,
        device_id: int,
        horizon: Optional[float] = None
    ) -> Union[ResourcePrediction, List[ResourcePrediction]]:
        """Predict future resource usage.
        
        Args:
            resource_type: Resource type
            device_id: Device ID
            horizon: Optional specific time horizon
            
        Returns:
            Single prediction or list of predictions for all horizons
        """
        key = (resource_type, device_id)
        if key not in self.history:
            return []
            
        current_time = self.history[key][-1][0]
        features = self._extract_features(key, current_time)
        
        if horizon is not None:
            predictor = self.predictors[key].get(horizon)
            if predictor and features:
                result = predictor.predict(features)
                return ResourcePrediction(
                    resource_type=resource_type,
                    device_id=device_id,
                    predicted_usage=result.predicted_value,
                    confidence=result.confidence,
                    time_horizon=horizon
                )
        else:
            predictions = []
            for horizon, predictor in self.predictors[key].items():
                if features:
                    result = predictor.predict(features)
                    predictions.append(ResourcePrediction(
                        resource_type=resource_type,
                        device_id=device_id,
                        predicted_usage=result.predicted_value,
                        confidence=result.confidence,
                        time_horizon=horizon
                    ))
            return predictions
            
    def _extract_features(
        self,
        key: Tuple[ResourceType, int],
        timestamp: float
    ) -> Optional[Dict[str, float]]:
        """Extract prediction features from history."""
        if len(self.history[key]) < 10:  # Need minimum history
            return None
            
        recent_history = self.history[key][-10:]
        usages = [usage for _, usage in recent_history]
        times = [time for time, _ in recent_history]
        
        return {
            "current_usage": usages[-1],
            "mean_usage": np.mean(usages),
            "std_usage": np.std(usages),
            "usage_trend": np.polyfit(times, usages, 1)[0],
            "max_usage": max(usages),
            "min_usage": min(usages),
            "time_since_peak": timestamp - times[usages.index(max(usages))]
        } 