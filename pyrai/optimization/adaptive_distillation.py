from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .distillation import DistillationConfig, DistillationMethod, DistillationLoss

class AdaptiveStrategy(Enum):
    """Adaptive distillation strategies."""
    DYNAMIC_TEMP = "dynamic_temp"  # Dynamic temperature
    DYNAMIC_WEIGHT = "dynamic_weight"  # Dynamic loss weights
    CURRICULUM = "curriculum"      # Curriculum learning
    DIFFICULTY = "difficulty"      # Sample difficulty
    UNCERTAINTY = "uncertainty"    # Model uncertainty

@dataclass
class AdaptiveConfig:
    """Adaptive distillation configuration."""
    strategy: AdaptiveStrategy
    min_temp: float = 1.0
    max_temp: float = 10.0
    min_weight: float = 0.1
    max_weight: float = 0.9
    warmup_steps: int = 1000
    update_interval: int = 100
    momentum: float = 0.9
    window_size: int = 100

class AdaptiveDistillation:
    """Adaptive knowledge distillation."""
    
    def __init__(
        self,
        config: AdaptiveConfig,
        base_config: DistillationConfig
    ):
        """Initialize adaptive distillation.
        
        Args:
            config: Adaptive configuration
            base_config: Base distillation configuration
        """
        self.config = config
        self.base_config = base_config
        self.step = 0
        self.history = []
        self.current_temp = base_config.temperature
        self.current_weight = base_config.alpha
        
    def adapt(
        self,
        metrics: Dict[str, float],
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: List[Dict[str, torch.Tensor]],
        labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Adapt distillation parameters.
        
        Args:
            metrics: Training metrics
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            labels: Ground truth labels
            
        Returns:
            Updated temperature and weight
        """
        self.step += 1
        self.history.append(metrics)
        
        if len(self.history) > self.config.window_size:
            self.history.pop(0)
            
        if self.step % self.config.update_interval == 0:
            if self.config.strategy == AdaptiveStrategy.DYNAMIC_TEMP:
                self.current_temp = self._adapt_temperature()
            elif self.config.strategy == AdaptiveStrategy.DYNAMIC_WEIGHT:
                self.current_weight = self._adapt_weight()
            elif self.config.strategy == AdaptiveStrategy.CURRICULUM:
                self.current_temp, self.current_weight = self._adapt_curriculum()
            elif self.config.strategy == AdaptiveStrategy.DIFFICULTY:
                self.current_temp, self.current_weight = self._adapt_difficulty(
                    student_outputs,
                    teacher_outputs,
                    labels
                )
            else:  # UNCERTAINTY
                self.current_temp, self.current_weight = self._adapt_uncertainty(
                    student_outputs,
                    teacher_outputs
                )
                
        return self.current_temp, self.current_weight
        
    def _adapt_temperature(self) -> float:
        """Adapt temperature based on loss trends."""
        if not self.history:
            return self.base_config.temperature
            
        # Calculate loss trend
        losses = [m["distill_loss"] for m in self.history]
        trend = np.mean(np.diff(losses))
        
        # Adjust temperature
        if trend > 0:  # Loss increasing
            temp = max(
                self.current_temp * 0.9,
                self.config.min_temp
            )
        else:  # Loss decreasing
            temp = min(
                self.current_temp * 1.1,
                self.config.max_temp
            )
            
        return temp
        
    def _adapt_weight(self) -> float:
        """Adapt loss weight based on performance."""
        if not self.history:
            return self.base_config.alpha
            
        # Calculate student progress
        ce_losses = [m["ce_loss"] for m in self.history]
        progress = 1 - (ce_losses[-1] / ce_losses[0])
        
        # Adjust weight
        weight = self.config.min_weight + (
            self.config.max_weight - self.config.min_weight
        ) * progress
        
        return weight
        
    def _adapt_curriculum(self) -> Tuple[float, float]:
        """Adapt based on curriculum learning."""
        # Calculate progress
        progress = min(1.0, self.step / self.config.warmup_steps)
        
        # Gradually decrease temperature and increase weight
        temp = self.config.max_temp - (
            self.config.max_temp - self.config.min_temp
        ) * progress
        
        weight = self.config.min_weight + (
            self.config.max_weight - self.config.min_weight
        ) * progress
        
        return temp, weight
        
    def _adapt_difficulty(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: List[Dict[str, torch.Tensor]],
        labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Adapt based on sample difficulty."""
        # Calculate sample difficulties
        student_probs = torch.softmax(student_outputs["logits"], dim=-1)
        teacher_probs = torch.softmax(teacher_outputs[0]["logits"], dim=-1)
        
        # KL divergence as difficulty measure
        difficulties = torch.sum(
            teacher_probs * torch.log(teacher_probs / student_probs),
            dim=-1
        )
        
        # Normalize difficulties
        difficulties = (difficulties - difficulties.min()) / (
            difficulties.max() - difficulties.min() + 1e-6
        )
        
        # Adjust parameters based on average difficulty
        avg_difficulty = difficulties.mean().item()
        temp = self.config.min_temp + (
            self.config.max_temp - self.config.min_temp
        ) * avg_difficulty
        
        weight = self.config.max_weight - (
            self.config.max_weight - self.config.min_weight
        ) * avg_difficulty
        
        return temp, weight
        
    def _adapt_uncertainty(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[float, float]:
        """Adapt based on model uncertainty."""
        # Calculate prediction entropy as uncertainty measure
        student_probs = torch.softmax(student_outputs["logits"], dim=-1)
        student_entropy = -torch.sum(
            student_probs * torch.log(student_probs + 1e-6),
            dim=-1
        ).mean()
        
        teacher_probs = torch.softmax(teacher_outputs[0]["logits"], dim=-1)
        teacher_entropy = -torch.sum(
            teacher_probs * torch.log(teacher_probs + 1e-6),
            dim=-1
        ).mean()
        
        # Normalize entropies
        max_entropy = -np.log(student_probs.size(-1))
        student_uncertainty = (student_entropy / max_entropy).item()
        teacher_uncertainty = (teacher_entropy / max_entropy).item()
        
        # Adjust parameters based on uncertainty difference
        uncertainty_diff = abs(teacher_uncertainty - student_uncertainty)
        temp = self.config.min_temp + (
            self.config.max_temp - self.config.min_temp
        ) * uncertainty_diff
        
        weight = self.config.max_weight - (
            self.config.max_weight - self.config.min_weight
        ) * uncertainty_diff
        
        return temp, weight
        
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get adaptation summary."""
        if not self.history:
            return {}
            
        return {
            "temperature": {
                "current": self.current_temp,
                "min": min(h.get("temperature", float("inf")) for h in self.history),
                "max": max(h.get("temperature", 0) for h in self.history),
                "mean": np.mean([h.get("temperature", 0) for h in self.history])
            },
            "weight": {
                "current": self.current_weight,
                "min": min(h.get("weight", float("inf")) for h in self.history),
                "max": max(h.get("weight", 0) for h in self.history),
                "mean": np.mean([h.get("weight", 0) for h in self.history])
            },
            "metrics": {
                "ce_loss": [h["ce_loss"] for h in self.history],
                "distill_loss": [h["distill_loss"] for h in self.history],
                "total_loss": [h["total_loss"] for h in self.history]
            }
        } 