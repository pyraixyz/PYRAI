from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .resources import ResourceManager, ResourceType, ResourceAllocation
from .resource_predictor import ResourcePredictor, ResourcePrediction

class AdjustmentStrategy(Enum):
    """Resource adjustment strategies."""
    CONSERVATIVE = "conservative"  # Minimize changes
    AGGRESSIVE = "aggressive"      # Maximize performance
    BALANCED = "balanced"          # Balance stability and performance

@dataclass
class AdjustmentAction:
    """Resource adjustment action."""
    resource_type: ResourceType
    device_id: int
    amount_change: float
    priority_change: float
    reason: str

class ResourceAdjuster:
    """Dynamically adjusts resource allocations."""
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        predictor: ResourcePredictor,
        strategy: AdjustmentStrategy = AdjustmentStrategy.BALANCED,
        adjustment_interval: float = 5.0,
        max_adjustment_ratio: float = 0.2  # Maximum 20% change
    ):
        """Initialize resource adjuster.
        
        Args:
            resource_manager: Resource manager
            predictor: Resource predictor
            strategy: Adjustment strategy
            adjustment_interval: Minimum time between adjustments
            max_adjustment_ratio: Maximum adjustment ratio
        """
        self.resource_manager = resource_manager
        self.predictor = predictor
        self.strategy = strategy
        self.adjustment_interval = adjustment_interval
        self.max_adjustment_ratio = max_adjustment_ratio
        self.last_adjustment: Dict[Tuple[ResourceType, int], float] = {}
        
    def adjust(
        self,
        current_time: float,
        min_confidence: float = 0.7
    ) -> List[AdjustmentAction]:
        """Adjust resource allocations.
        
        Args:
            current_time: Current timestamp
            min_confidence: Minimum prediction confidence
            
        Returns:
            List of adjustment actions
        """
        actions = []
        
        # Check each resource and device
        for device_id in range(self.resource_manager.num_devices):
            for resource_type in ResourceType:
                key = (resource_type, device_id)
                
                # Check adjustment interval
                last_time = self.last_adjustment.get(key, 0)
                if current_time - last_time < self.adjustment_interval:
                    continue
                    
                # Get predictions
                predictions = self.predictor.predict(resource_type, device_id)
                if not predictions:
                    continue
                    
                # Filter high-confidence predictions
                confident_predictions = [
                    p for p in predictions
                    if p.confidence >= min_confidence
                ]
                
                if confident_predictions:
                    action = self._determine_adjustment(
                        resource_type,
                        device_id,
                        confident_predictions
                    )
                    if action:
                        actions.append(action)
                        self.last_adjustment[key] = current_time
                        
        return actions
        
    def apply_actions(self, actions: List[AdjustmentAction]) -> bool:
        """Apply adjustment actions."""
        success = True
        
        for action in actions:
            # Release resources if reducing
            if action.amount_change < 0:
                if not self.resource_manager.release(
                    action.resource_type,
                    action.device_id,
                    abs(action.amount_change)
                ):
                    success = False
                    continue
                    
            # Allocate resources if increasing
            if action.amount_change > 0:
                if not self.resource_manager.allocate(
                    action.resource_type,
                    action.device_id,
                    action.amount_change,
                    1.0 + action.priority_change
                ):
                    success = False
                    continue
                    
        return success
        
    def _determine_adjustment(
        self,
        resource_type: ResourceType,
        device_id: int,
        predictions: List[ResourcePrediction]
    ) -> Optional[AdjustmentAction]:
        """Determine resource adjustment based on predictions."""
        current_usage = self.resource_manager.get_utilization(
            resource_type,
            device_id
        )
        
        # Get predicted usage trends
        short_term = predictions[0].predicted_usage
        long_term = predictions[-1].predicted_usage
        
        # Calculate adjustment amount based on strategy
        if self.strategy == AdjustmentStrategy.CONSERVATIVE:
            adjustment = self._conservative_adjustment(
                current_usage,
                short_term,
                long_term
            )
        elif self.strategy == AdjustmentStrategy.AGGRESSIVE:
            adjustment = self._aggressive_adjustment(
                current_usage,
                short_term,
                long_term
            )
        else:  # Balanced
            adjustment = self._balanced_adjustment(
                current_usage,
                short_term,
                long_term
            )
            
        if abs(adjustment) > 0:
            return AdjustmentAction(
                resource_type=resource_type,
                device_id=device_id,
                amount_change=adjustment,
                priority_change=self._calculate_priority_change(
                    current_usage,
                    short_term,
                    long_term
                ),
                reason=self._get_adjustment_reason(
                    current_usage,
                    short_term,
                    long_term
                )
            )
        return None
        
    def _conservative_adjustment(
        self,
        current: float,
        short_term: float,
        long_term: float
    ) -> float:
        """Conservative adjustment strategy."""
        # Only adjust if both predictions agree
        if (short_term > current and long_term > current) or \
           (short_term < current and long_term < current):
            direction = 1 if short_term > current else -1
            return direction * min(
                abs(short_term - current),
                self.max_adjustment_ratio * current
            )
        return 0.0
        
    def _aggressive_adjustment(
        self,
        current: float,
        short_term: float,
        long_term: float
    ) -> float:
        """Aggressive adjustment strategy."""
        # Adjust based on maximum predicted change
        max_prediction = max(short_term, long_term)
        min_prediction = min(short_term, long_term)
        
        if max_prediction > current:
            return min(
                max_prediction - current,
                self.max_adjustment_ratio * current
            )
        elif min_prediction < current:
            return max(
                min_prediction - current,
                -self.max_adjustment_ratio * current
            )
        return 0.0
        
    def _balanced_adjustment(
        self,
        current: float,
        short_term: float,
        long_term: float
    ) -> float:
        """Balanced adjustment strategy."""
        # Weight short-term more heavily
        predicted = 0.7 * short_term + 0.3 * long_term
        if abs(predicted - current) > 0.1 * current:  # 10% threshold
            direction = 1 if predicted > current else -1
            return direction * min(
                abs(predicted - current),
                self.max_adjustment_ratio * current
            )
        return 0.0
        
    def _calculate_priority_change(
        self,
        current: float,
        short_term: float,
        long_term: float
    ) -> float:
        """Calculate priority change based on predictions."""
        # Increase priority if usage is predicted to increase
        avg_prediction = (short_term + long_term) / 2
        if avg_prediction > current:
            return min((avg_prediction - current) / current, 0.5)
        return 0.0
        
    def _get_adjustment_reason(
        self,
        current: float,
        short_term: float,
        long_term: float
    ) -> str:
        """Get human-readable reason for adjustment."""
        if short_term > current and long_term > current:
            return "Predicted sustained increase in resource usage"
        elif short_term < current and long_term < current:
            return "Predicted sustained decrease in resource usage"
        elif short_term > current:
            return "Predicted short-term spike in resource usage"
        else:
            return "Predicted short-term dip in resource usage" 