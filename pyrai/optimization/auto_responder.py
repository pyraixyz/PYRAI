from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .alerter import Alert, AlertType, AlertSeverity
from .monitor import RuntimeStats
from .predictor import PredictionResult

class ResponseAction(Enum):
    """Automatic response actions."""
    ADJUST_BATCH_SIZE = "adjust_batch_size"
    ADJUST_LEARNING_RATE = "adjust_learning_rate"
    ENABLE_CHECKPOINTING = "enable_checkpointing"
    ENABLE_COMPRESSION = "enable_compression"
    ADJUST_PIPELINE = "adjust_pipeline"
    REBALANCE_LOAD = "rebalance_load"

@dataclass
class ResponseConfig:
    """Response configuration."""
    action: ResponseAction
    params: Dict[str, Any]
    priority: int
    conditions: List[AlertType]

class AutoResponder:
    """Automatic performance issue responder."""
    
    def __init__(
        self,
        trainer: Any,  # Training framework interface
        configs: List[ResponseConfig],
        min_confidence: float = 0.7,
        cooldown: int = 100  # Steps between actions
    ):
        """Initialize auto responder.
        
        Args:
            trainer: Training framework interface
            configs: Response configurations
            min_confidence: Minimum prediction confidence
            cooldown: Steps between actions
        """
        self.trainer = trainer
        self.configs = {
            tuple(config.conditions): config
            for config in configs
        }
        self.min_confidence = min_confidence
        self.cooldown = cooldown
        self.last_action = 0
        self.action_history: List[Tuple[ResponseAction, Dict[str, Any]]] = []
        
    def respond(
        self,
        alerts: List[Alert],
        predictions: Dict[str, PredictionResult],
        stats: RuntimeStats
    ) -> Optional[Tuple[ResponseAction, Dict[str, Any]]]:
        """Generate response to performance issues.
        
        Args:
            alerts: Active alerts
            predictions: Performance predictions
            stats: Current statistics
            
        Returns:
            Response action and parameters if needed
        """
        if not alerts or len(self.action_history) > self.last_action + self.cooldown:
            return None
            
        # Get alert types
        alert_types = tuple(sorted(alert.type for alert in alerts))
        
        # Find matching config
        config = self.configs.get(alert_types)
        if not config:
            return None
            
        # Check predictions
        if not self._validate_predictions(predictions):
            return None
            
        # Generate response parameters
        params = self._generate_params(config, alerts, predictions, stats)
        
        # Apply response
        success = self._apply_response(config.action, params)
        if success:
            self.action_history.append((config.action, params))
            self.last_action = len(self.action_history)
            return config.action, params
            
        return None
        
    def _validate_predictions(
        self,
        predictions: Dict[str, PredictionResult]
    ) -> bool:
        """Validate prediction confidence."""
        return all(
            pred.confidence >= self.min_confidence
            for pred in predictions.values()
        )
        
    def _generate_params(
        self,
        config: ResponseConfig,
        alerts: List[Alert],
        predictions: Dict[str, PredictionResult],
        stats: RuntimeStats
    ) -> Dict[str, Any]:
        """Generate response parameters."""
        if config.action == ResponseAction.ADJUST_BATCH_SIZE:
            return self._generate_batch_size_params(alerts, predictions, stats)
        elif config.action == ResponseAction.ADJUST_LEARNING_RATE:
            return self._generate_learning_rate_params(alerts, predictions)
        elif config.action == ResponseAction.ENABLE_CHECKPOINTING:
            return self._generate_checkpointing_params(alerts)
        elif config.action == ResponseAction.ENABLE_COMPRESSION:
            return self._generate_compression_params(alerts, stats)
        elif config.action == ResponseAction.ADJUST_PIPELINE:
            return self._generate_pipeline_params(alerts, predictions)
        else:  # REBALANCE_LOAD
            return self._generate_load_balance_params(alerts, stats)
            
    def _generate_batch_size_params(
        self,
        alerts: List[Alert],
        predictions: Dict[str, PredictionResult],
        stats: RuntimeStats
    ) -> Dict[str, Any]:
        """Generate batch size adjustment parameters."""
        current_batch_size = self.trainer.get_batch_size()
        memory_alert = next(
            (a for a in alerts if a.type == AlertType.MEMORY_OVERFLOW),
            None
        )
        
        if memory_alert:
            # Reduce batch size
            return {
                "batch_size": int(current_batch_size * 0.8),
                "reason": "memory_overflow"
            }
            
        utilization_alert = next(
            (a for a in alerts if a.type == AlertType.LOW_UTILIZATION),
            None
        )
        
        if utilization_alert:
            # Increase batch size
            return {
                "batch_size": int(current_batch_size * 1.2),
                "reason": "low_utilization"
            }
            
        return {"batch_size": current_batch_size}
        
    def _generate_learning_rate_params(
        self,
        alerts: List[Alert],
        predictions: Dict[str, PredictionResult]
    ) -> Dict[str, Any]:
        """Generate learning rate adjustment parameters."""
        current_lr = self.trainer.get_learning_rate()
        convergence_alert = next(
            (a for a in alerts if a.type == AlertType.CONVERGENCE_ISSUE),
            None
        )
        
        if convergence_alert:
            # Adjust learning rate based on convergence
            if predictions.get("convergence_rate", None):
                pred = predictions["convergence_rate"]
                if pred.trend == "decreasing":
                    return {
                        "learning_rate": current_lr * 0.5,
                        "reason": "slow_convergence"
                    }
                    
        return {"learning_rate": current_lr}
        
    def _generate_checkpointing_params(
        self,
        alerts: List[Alert]
    ) -> Dict[str, Any]:
        """Generate checkpointing parameters."""
        memory_alert = next(
            (a for a in alerts if a.type == AlertType.MEMORY_OVERFLOW),
            None
        )
        
        if memory_alert and memory_alert.severity == AlertSeverity.CRITICAL:
            return {
                "enable": True,
                "frequency": 100,  # Steps between checkpoints
                "reason": "critical_memory"
            }
            
        return {"enable": False}
        
    def _generate_compression_params(
        self,
        alerts: List[Alert],
        stats: RuntimeStats
    ) -> Dict[str, Any]:
        """Generate compression parameters."""
        overhead_alert = next(
            (a for a in alerts if a.type == AlertType.HIGH_OVERHEAD),
            None
        )
        
        if overhead_alert:
            return {
                "enable": True,
                "algorithm": "fp16",  # or other compression algorithm
                "reason": "high_communication_overhead"
            }
            
        return {"enable": False}
        
    def _generate_pipeline_params(
        self,
        alerts: List[Alert],
        predictions: Dict[str, PredictionResult]
    ) -> Dict[str, Any]:
        """Generate pipeline adjustment parameters."""
        pipeline_alert = next(
            (a for a in alerts if a.type == AlertType.PIPELINE_STALL),
            None
        )
        
        if pipeline_alert:
            current_chunks = self.trainer.get_pipeline_chunks()
            return {
                "chunks": current_chunks * 2,
                "reason": "pipeline_inefficiency"
            }
            
        return {"chunks": self.trainer.get_pipeline_chunks()}
        
    def _generate_load_balance_params(
        self,
        alerts: List[Alert],
        stats: RuntimeStats
    ) -> Dict[str, Any]:
        """Generate load balancing parameters."""
        scaling_alert = next(
            (a for a in alerts if a.type == AlertType.SCALING_ISSUE),
            None
        )
        
        if scaling_alert:
            return {
                "rebalance": True,
                "strategy": "dynamic",
                "reason": "poor_scaling"
            }
            
        return {"rebalance": False}
        
    def _apply_response(
        self,
        action: ResponseAction,
        params: Dict[str, Any]
    ) -> bool:
        """Apply response action."""
        try:
            if action == ResponseAction.ADJUST_BATCH_SIZE:
                self.trainer.set_batch_size(params["batch_size"])
            elif action == ResponseAction.ADJUST_LEARNING_RATE:
                self.trainer.set_learning_rate(params["learning_rate"])
            elif action == ResponseAction.ENABLE_CHECKPOINTING:
                self.trainer.set_checkpointing(params["enable"], params.get("frequency"))
            elif action == ResponseAction.ENABLE_COMPRESSION:
                self.trainer.set_compression(params["enable"], params.get("algorithm"))
            elif action == ResponseAction.ADJUST_PIPELINE:
                self.trainer.set_pipeline_chunks(params["chunks"])
            else:  # REBALANCE_LOAD
                self.trainer.rebalance_load(params["strategy"])
            return True
        except:
            return False 