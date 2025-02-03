from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .resource_adjuster import AdjustmentStrategy
from .coordinator import CoordinationStrategy

@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    adjustment_success_rate: float
    resource_utilization: float
    prediction_accuracy: float
    coordination_overhead: float

class StrategySelector:
    """Dynamically selects optimal strategies."""
    
    def __init__(
        self,
        evaluation_window: int = 100,
        min_switch_interval: float = 30.0  # seconds
    ):
        """Initialize strategy selector.
        
        Args:
            evaluation_window: Window size for strategy evaluation
            min_switch_interval: Minimum time between strategy switches
        """
        self.evaluation_window = evaluation_window
        self.min_switch_interval = min_switch_interval
        self.strategy_history: Dict[str, List[StrategyMetrics]] = {
            strategy.value: []
            for strategy in list(AdjustmentStrategy) + list(CoordinationStrategy)
        }
        self.current_strategies = {
            "adjustment": AdjustmentStrategy.BALANCED,
            "coordination": CoordinationStrategy.HIERARCHICAL
        }
        self.last_switch = 0.0
        
    def update_metrics(
        self,
        strategy: Union[AdjustmentStrategy, CoordinationStrategy],
        metrics: StrategyMetrics
    ):
        """Update strategy performance metrics."""
        self.strategy_history[strategy.value].append(metrics)
        if len(self.strategy_history[strategy.value]) > self.evaluation_window:
            self.strategy_history[strategy.value].pop(0)
            
    def select_strategies(
        self,
        current_time: float,
        system_load: float
    ) -> Dict[str, Union[AdjustmentStrategy, CoordinationStrategy]]:
        """Select optimal strategies based on current conditions.
        
        Args:
            current_time: Current timestamp
            system_load: Current system load
            
        Returns:
            Selected strategies
        """
        if current_time - self.last_switch < self.min_switch_interval:
            return self.current_strategies
            
        new_strategies = {}
        
        # Select adjustment strategy
        if system_load > 0.8:  # High load
            new_strategies["adjustment"] = AdjustmentStrategy.CONSERVATIVE
        elif system_load < 0.3:  # Low load
            new_strategies["adjustment"] = AdjustmentStrategy.AGGRESSIVE
        else:
            new_strategies["adjustment"] = self._select_best_strategy(
                list(AdjustmentStrategy)
            )
            
        # Select coordination strategy
        if system_load > 0.9:  # Very high load
            new_strategies["coordination"] = CoordinationStrategy.DECENTRALIZED
        else:
            new_strategies["coordination"] = self._select_best_strategy(
                list(CoordinationStrategy)
            )
            
        if new_strategies != self.current_strategies:
            self.current_strategies = new_strategies
            self.last_switch = current_time
            
        return self.current_strategies
        
    def _select_best_strategy(
        self,
        strategies: List[Union[AdjustmentStrategy, CoordinationStrategy]]
    ) -> Union[AdjustmentStrategy, CoordinationStrategy]:
        """Select best performing strategy."""
        best_score = float('-inf')
        best_strategy = None
        
        for strategy in strategies:
            history = self.strategy_history[strategy.value]
            if not history:
                continue
                
            # Calculate strategy score
            score = np.mean([
                self._calculate_strategy_score(metrics)
                for metrics in history[-10:]  # Recent history
            ])
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
                
        return best_strategy or strategies[0]  # Default to first strategy
        
    def _calculate_strategy_score(self, metrics: StrategyMetrics) -> float:
        """Calculate overall strategy performance score."""
        return (
            0.4 * metrics.adjustment_success_rate +
            0.3 * metrics.resource_utilization +
            0.2 * metrics.prediction_accuracy -
            0.1 * metrics.coordination_overhead
        ) 