from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .distillation_monitor import DistillationMetrics, DistillationMonitor

class TuningMode(Enum):
    """Distillation tuning modes."""
    MANUAL = "manual"
    AUTO = "auto"
    ADAPTIVE = "adaptive"

@dataclass
class TuningConfig:
    """Distillation tuning configuration."""
    mode: TuningMode
    temperature_range: Tuple[float, float] = (0.5, 5.0)
    alpha_range: Tuple[float, float] = (0.1, 0.9)
    patience: int = 5
    min_improvement: float = 0.01
    max_trials: int = 20

class DistillationTuner:
    """Tunes knowledge distillation hyperparameters."""
    
    def __init__(
        self,
        config: TuningConfig,
        monitor: DistillationMonitor
    ):
        """Initialize tuner.
        
        Args:
            config: Tuning configuration
            monitor: Distillation monitor
        """
        self.config = config
        self.monitor = monitor
        self.history: List[Dict[str, float]] = []
        self.best_metrics: Optional[DistillationMetrics] = None
        self.best_score: float = float('-inf')
        self.no_improvement: int = 0
        self.trial: int = 0
        
    def suggest_params(
        self,
        current_metrics: DistillationMetrics
    ) -> Optional[Dict[str, float]]:
        """Suggest next hyperparameters."""
        self.trial += 1
        if self.trial > self.config.max_trials:
            return None
            
        # Record metrics
        score = self._calculate_score(current_metrics)
        self.history.append({
            "temperature": current_metrics.temperature,
            "alpha": current_metrics.alpha,
            "score": score,
            "student_loss": current_metrics.student_loss,
            "accuracy_gap": current_metrics.accuracy_gap
        })
        
        # Update best metrics
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = current_metrics
            self.no_improvement = 0
        else:
            self.no_improvement += 1
            
        # Check early stopping
        if self.no_improvement >= self.config.patience:
            return None
            
        # Generate suggestions
        if self.config.mode == TuningMode.MANUAL:
            return None
        elif self.config.mode == TuningMode.AUTO:
            return self._auto_suggest()
        else:  # ADAPTIVE
            return self._adaptive_suggest(current_metrics)
            
    def _calculate_score(self, metrics: DistillationMetrics) -> float:
        """Calculate optimization score."""
        # Combine multiple objectives
        accuracy_weight = 0.6
        loss_weight = 0.4
        
        accuracy_score = 1.0 + metrics.accuracy_gap  # Higher is better
        loss_score = (
            metrics.teacher_loss / metrics.student_loss
            if metrics.student_loss > 0 else 0.0
        )
        
        return (
            accuracy_weight * accuracy_score +
            loss_weight * loss_score
        )
        
    def _auto_suggest(self) -> Dict[str, float]:
        """Generate automatic suggestions."""
        # Grid search over parameter ranges
        temp_min, temp_max = self.config.temperature_range
        alpha_min, alpha_max = self.config.alpha_range
        
        grid_size = 5
        temperatures = np.linspace(temp_min, temp_max, grid_size)
        alphas = np.linspace(alpha_min, alpha_max, grid_size)
        
        # Find best unexplored combination
        tried_params = {
            (h["temperature"], h["alpha"])
            for h in self.history
        }
        
        best_score = float('-inf')
        best_params = None
        
        for temp in temperatures:
            for alpha in alphas:
                if (temp, alpha) in tried_params:
                    continue
                    
                # Estimate score based on nearby points
                nearby_scores = []
                for h in self.history:
                    dist = np.sqrt(
                        (h["temperature"] - temp) ** 2 +
                        (h["alpha"] - alpha) ** 2
                    )
                    if dist < 1.0:
                        nearby_scores.append(h["score"])
                        
                if nearby_scores:
                    estimated_score = np.mean(nearby_scores)
                    if estimated_score > best_score:
                        best_score = estimated_score
                        best_params = {"temperature": temp, "alpha": alpha}
                        
        return best_params or {
            "temperature": np.random.uniform(temp_min, temp_max),
            "alpha": np.random.uniform(alpha_min, alpha_max)
        }
        
    def _adaptive_suggest(
        self,
        current_metrics: DistillationMetrics
    ) -> Dict[str, float]:
        """Generate adaptive suggestions."""
        if not self.history:
            return self._auto_suggest()
            
        # Analyze trends
        recent = self.history[-min(5, len(self.history)):]
        
        temp_trend = np.mean([
            h["score"] - prev["score"]
            for prev, h in zip(recent[:-1], recent[1:])
            if h["temperature"] > prev["temperature"]
        ] or [0])
        
        alpha_trend = np.mean([
            h["score"] - prev["score"]
            for prev, h in zip(recent[:-1], recent[1:])
            if h["alpha"] > prev["alpha"]
        ] or [0])
        
        # Adjust parameters based on trends
        temp_step = 0.2 * np.sign(temp_trend)
        alpha_step = 0.1 * np.sign(alpha_trend)
        
        new_temp = np.clip(
            current_metrics.temperature + temp_step,
            *self.config.temperature_range
        )
        
        new_alpha = np.clip(
            current_metrics.alpha + alpha_step,
            *self.config.alpha_range
        )
        
        return {
            "temperature": new_temp,
            "alpha": new_alpha
        }
        
    def get_best_params(self) -> Optional[Dict[str, float]]:
        """Get best found parameters."""
        if not self.best_metrics:
            return None
            
        return {
            "temperature": self.best_metrics.temperature,
            "alpha": self.best_metrics.alpha
        }
        
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get tuning summary."""
        if not self.history:
            return {}
            
        scores = [h["score"] for h in self.history]
        temps = [h["temperature"] for h in self.history]
        alphas = [h["alpha"] for h in self.history]
        
        return {
            "trials": self.trial,
            "best_score": self.best_score,
            "score_trend": np.polyfit(range(len(scores)), scores, 1)[0],
            "temperature_range": [min(temps), max(temps)],
            "alpha_range": [min(alphas), max(alphas)],
            "no_improvement_steps": self.no_improvement
        } 