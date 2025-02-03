from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .metrics_collector import MetricsCollector, MetricType
from .alert_rules import AlertRule, AlertLevel

@dataclass
class DistillationMetrics:
    """Knowledge distillation metrics."""
    student_loss: float
    teacher_loss: float
    distillation_loss: float
    accuracy_gap: float
    temperature: float
    alpha: float

class DistillationMonitor:
    """Monitors knowledge distillation process."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        alert_manager: Any,  # AlertManager
        min_accuracy_gap: float = -0.1,
        max_loss_ratio: float = 2.0
    ):
        """Initialize monitor.
        
        Args:
            collector: Metrics collector
            alert_manager: Alert manager
            min_accuracy_gap: Minimum acceptable accuracy gap
            max_loss_ratio: Maximum acceptable loss ratio
        """
        self.collector = collector
        self.alert_manager = alert_manager
        self.min_accuracy_gap = min_accuracy_gap
        self.max_loss_ratio = max_loss_ratio
        self._setup_rules()
        
    def record_metrics(
        self,
        metrics: DistillationMetrics,
        step: int
    ):
        """Record distillation metrics."""
        labels = {"step": str(step)}
        
        # Record individual metrics
        self.collector.record_metric(
            "distillation_student_loss",
            metrics.student_loss,
            MetricType.GAUGE,
            labels
        )
        
        self.collector.record_metric(
            "distillation_teacher_loss",
            metrics.teacher_loss,
            MetricType.GAUGE,
            labels
        )
        
        self.collector.record_metric(
            "distillation_loss",
            metrics.distillation_loss,
            MetricType.GAUGE,
            labels
        )
        
        self.collector.record_metric(
            "distillation_accuracy_gap",
            metrics.accuracy_gap,
            MetricType.GAUGE,
            labels
        )
        
        self.collector.record_metric(
            "distillation_temperature",
            metrics.temperature,
            MetricType.GAUGE,
            labels
        )
        
        self.collector.record_metric(
            "distillation_alpha",
            metrics.alpha,
            MetricType.GAUGE,
            labels
        )
        
        # Record derived metrics
        if metrics.teacher_loss > 0:
            loss_ratio = metrics.student_loss / metrics.teacher_loss
            self.collector.record_metric(
                "distillation_loss_ratio",
                loss_ratio,
                MetricType.GAUGE,
                labels
            )
            
    def _setup_rules(self):
        """Setup monitoring rules."""
        # Accuracy gap rule
        self.alert_manager.add_rule(AlertRule(
            name="low_accuracy",
            metric="distillation_accuracy_gap",
            condition=lambda x: x < self.min_accuracy_gap,
            level=AlertLevel.WARNING,
            labels={"type": "distillation"},
            description="Student accuracy too low compared to teacher",
            for_duration=300.0  # 5 minutes
        ))
        
        # Loss ratio rule
        self.alert_manager.add_rule(AlertRule(
            name="high_loss_ratio",
            metric="distillation_loss_ratio",
            condition=lambda x: x > self.max_loss_ratio,
            level=AlertLevel.WARNING,
            labels={"type": "distillation"},
            description="Student loss too high compared to teacher",
            for_duration=300.0  # 5 minutes
        ))
        
        # Temperature rule
        self.alert_manager.add_rule(AlertRule(
            name="extreme_temperature",
            metric="distillation_temperature",
            condition=lambda x: x < 0.1 or x > 10.0,
            level=AlertLevel.WARNING,
            labels={"type": "distillation"},
            description="Distillation temperature out of reasonable range",
            for_duration=300.0  # 5 minutes
        ))
        
    def get_summary(self) -> Dict[str, Any]:
        """Get distillation summary."""
        metrics = [
            "distillation_student_loss",
            "distillation_teacher_loss",
            "distillation_loss",
            "distillation_accuracy_gap",
            "distillation_loss_ratio"
        ]
        
        summary = {}
        for metric in metrics:
            stats = self.collector.get_summary(metric)
            if stats:
                summary[metric] = stats
                
        return summary 