from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .auto_responder import ResponseAction
from .monitor import RuntimeStats
from .predictor import PredictionResult

class ResponseOutcome(Enum):
    """Response evaluation outcomes."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"

@dataclass
class ResponseEvaluation:
    """Response evaluation results."""
    action: ResponseAction
    outcome: ResponseOutcome
    improvement: float
    metrics: Dict[str, float]
    duration: float
    side_effects: List[str]

class ResponseEvaluator:
    """Evaluates automatic response effectiveness."""
    
    def __init__(
        self,
        evaluation_window: int = 50,
        improvement_threshold: float = 0.1,
        regression_threshold: float = -0.05
    ):
        """Initialize response evaluator.
        
        Args:
            evaluation_window: Window size for evaluation
            improvement_threshold: Minimum improvement threshold
            regression_threshold: Maximum regression threshold
        """
        self.evaluation_window = evaluation_window
        self.improvement_threshold = improvement_threshold
        self.regression_threshold = regression_threshold
        self.baseline_stats: Optional[RuntimeStats] = None
        self.history: List[RuntimeStats] = []
        self.evaluations: List[ResponseEvaluation] = []
        
    def start_evaluation(
        self,
        action: ResponseAction,
        stats: RuntimeStats
    ):
        """Start evaluating a response action.
        
        Args:
            action: Response action to evaluate
            stats: Current statistics
        """
        self.baseline_stats = stats
        self.history = [stats]
        
    def update(
        self,
        stats: RuntimeStats,
        predictions: Optional[Dict[str, PredictionResult]] = None
    ) -> Optional[ResponseEvaluation]:
        """Update evaluation with new data.
        
        Args:
            stats: Current statistics
            predictions: Optional performance predictions
            
        Returns:
            Evaluation result if complete
        """
        if not self.baseline_stats:
            return None
            
        self.history.append(stats)
        if len(self.history) > self.evaluation_window:
            return self._complete_evaluation(predictions)
            
        return None
        
    def _complete_evaluation(
        self,
        predictions: Optional[Dict[str, PredictionResult]]
    ) -> ResponseEvaluation:
        """Complete response evaluation."""
        # Calculate metric changes
        metric_changes = self._calculate_metric_changes()
        
        # Determine overall improvement
        improvement = np.mean([
            change for change in metric_changes.values()
            if not np.isnan(change)
        ])
        
        # Check for side effects
        side_effects = self._check_side_effects(metric_changes)
        
        # Determine outcome
        outcome = self._determine_outcome(improvement, side_effects)
        
        # Create evaluation
        evaluation = ResponseEvaluation(
            action=self.current_action,
            outcome=outcome,
            improvement=improvement,
            metrics=metric_changes,
            duration=len(self.history),
            side_effects=side_effects
        )
        
        self.evaluations.append(evaluation)
        return evaluation
        
    def _calculate_metric_changes(self) -> Dict[str, float]:
        """Calculate changes in performance metrics."""
        if not self.baseline_stats or not self.history:
            return {}
            
        baseline = self.baseline_stats
        current = self.history[-1]
        
        return {
            "throughput": (
                current.throughput / baseline.throughput - 1
                if baseline.throughput > 0 else np.nan
            ),
            "latency": (
                baseline.latency / current.latency - 1
                if current.latency > 0 else np.nan
            ),
            "memory_utilization": (
                np.mean(current.gpu_memory) /
                np.mean(baseline.gpu_memory) - 1
                if np.mean(baseline.gpu_memory) > 0 else np.nan
            ),
            "compute_utilization": (
                np.mean(current.gpu_utilization) /
                np.mean(baseline.gpu_utilization) - 1
                if np.mean(baseline.gpu_utilization) > 0 else np.nan
            ),
            "communication_overhead": (
                baseline.communication_overhead /
                current.communication_overhead - 1
                if current.communication_overhead > 0 else np.nan
            )
        }
        
    def _check_side_effects(
        self,
        metric_changes: Dict[str, float]
    ) -> List[str]:
        """Check for negative side effects."""
        side_effects = []
        
        for metric, change in metric_changes.items():
            if np.isnan(change):
                continue
                
            if change < self.regression_threshold:
                side_effects.append(
                    f"Significant regression in {metric}: {change:.1%}"
                )
                
        return side_effects
        
    def _determine_outcome(
        self,
        improvement: float,
        side_effects: List[str]
    ) -> ResponseOutcome:
        """Determine response outcome."""
        if np.isnan(improvement):
            return ResponseOutcome.UNKNOWN
            
        if improvement >= self.improvement_threshold and not side_effects:
            return ResponseOutcome.SUCCESS
        elif improvement >= 0 or len(side_effects) <= 1:
            return ResponseOutcome.PARTIAL
        else:
            return ResponseOutcome.FAILURE 