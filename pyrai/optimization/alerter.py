from typing import Any, Dict, List, Optional, Union, Tuple
import time
from dataclasses import dataclass
from enum import Enum
from .monitor import RuntimeStats
from .profiler import ProfileResult, ProfilerMetric
from .advanced_profiler import AdvancedProfileStats

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of performance alerts."""
    MEMORY_OVERFLOW = "memory_overflow"
    LOW_UTILIZATION = "low_utilization"
    HIGH_OVERHEAD = "high_overhead"
    CONVERGENCE_ISSUE = "convergence_issue"
    SCALING_ISSUE = "scaling_issue"
    PIPELINE_STALL = "pipeline_stall"

@dataclass
class Alert:
    """Performance alert details."""
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float
    metrics: Dict[str, float]
    suggestions: List[str]

class PerformanceAlerter:
    """Monitors and alerts on performance issues."""
    
    def __init__(
        self,
        memory_threshold: float = 0.9,
        utilization_threshold: float = 0.5,
        overhead_threshold: float = 0.3,
        convergence_threshold: float = 0.1,
        alert_cooldown: float = 300.0  # 5 minutes
    ):
        """Initialize alerter.
        
        Args:
            memory_threshold: Memory usage threshold
            utilization_threshold: Utilization threshold
            overhead_threshold: Overhead threshold
            convergence_threshold: Convergence threshold
            alert_cooldown: Minimum time between similar alerts
        """
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        self.overhead_threshold = overhead_threshold
        self.convergence_threshold = convergence_threshold
        self.alert_cooldown = alert_cooldown
        self.last_alerts: Dict[AlertType, float] = {}
        self.alert_history: List[Alert] = []
        
    def check(
        self,
        stats: RuntimeStats,
        profile_results: List[ProfileResult],
        advanced_stats: Optional[AdvancedProfileStats] = None
    ) -> List[Alert]:
        """Check for performance issues.
        
        Args:
            stats: Runtime statistics
            profile_results: Profiling results
            advanced_stats: Optional advanced statistics
            
        Returns:
            List of generated alerts
        """
        current_time = time.time()
        alerts = []
        
        # Check memory usage
        memory_alert = self._check_memory(stats, current_time)
        if memory_alert:
            alerts.append(memory_alert)
            
        # Check utilization
        utilization_alert = self._check_utilization(stats, current_time)
        if utilization_alert:
            alerts.append(utilization_alert)
            
        # Check overhead
        overhead_alert = self._check_overhead(stats, current_time)
        if overhead_alert:
            alerts.append(overhead_alert)
            
        # Check advanced metrics if available
        if advanced_stats:
            convergence_alert = self._check_convergence(
                advanced_stats,
                current_time
            )
            if convergence_alert:
                alerts.append(convergence_alert)
                
            scaling_alert = self._check_scaling(
                advanced_stats,
                current_time
            )
            if scaling_alert:
                alerts.append(scaling_alert)
                
            pipeline_alert = self._check_pipeline(
                advanced_stats,
                current_time
            )
            if pipeline_alert:
                alerts.append(pipeline_alert)
                
        # Update history
        self.alert_history.extend(alerts)
        return alerts
        
    def _check_memory(
        self,
        stats: RuntimeStats,
        current_time: float
    ) -> Optional[Alert]:
        """Check memory usage issues."""
        if self._is_cooldown(AlertType.MEMORY_OVERFLOW, current_time):
            return None
            
        memory_util = np.mean(stats.gpu_memory)
        if memory_util > self.memory_threshold:
            severity = AlertSeverity.CRITICAL if memory_util > 0.95 else AlertSeverity.WARNING
            
            return Alert(
                type=AlertType.MEMORY_OVERFLOW,
                severity=severity,
                message=f"High memory usage detected: {memory_util:.1%}",
                timestamp=current_time,
                metrics={"memory_utilization": memory_util},
                suggestions=[
                    "Consider reducing batch size",
                    "Enable gradient checkpointing",
                    "Use model parallelism"
                ]
            )
            
        return None
        
    def _check_utilization(
        self,
        stats: RuntimeStats,
        current_time: float
    ) -> Optional[Alert]:
        """Check utilization issues."""
        if self._is_cooldown(AlertType.LOW_UTILIZATION, current_time):
            return None
            
        gpu_util = np.mean(stats.gpu_utilization)
        if gpu_util < self.utilization_threshold:
            return Alert(
                type=AlertType.LOW_UTILIZATION,
                severity=AlertSeverity.WARNING,
                message=f"Low GPU utilization: {gpu_util:.1%}",
                timestamp=current_time,
                metrics={"gpu_utilization": gpu_util},
                suggestions=[
                    "Increase batch size",
                    "Check data loading pipeline",
                    "Optimize model architecture"
                ]
            )
            
        return None
        
    def _check_overhead(
        self,
        stats: RuntimeStats,
        current_time: float
    ) -> Optional[Alert]:
        """Check overhead issues."""
        if self._is_cooldown(AlertType.HIGH_OVERHEAD, current_time):
            return None
            
        total_time = stats.compute_time + stats.communication_overhead
        overhead_ratio = stats.communication_overhead / total_time
        
        if overhead_ratio > self.overhead_threshold:
            return Alert(
                type=AlertType.HIGH_OVERHEAD,
                severity=AlertSeverity.WARNING,
                message=f"High communication overhead: {overhead_ratio:.1%}",
                timestamp=current_time,
                metrics={"overhead_ratio": overhead_ratio},
                suggestions=[
                    "Enable gradient compression",
                    "Optimize communication pattern",
                    "Check network configuration"
                ]
            )
            
        return None
        
    def _check_convergence(
        self,
        stats: AdvancedProfileStats,
        current_time: float
    ) -> Optional[Alert]:
        """Check convergence issues."""
        if self._is_cooldown(AlertType.CONVERGENCE_ISSUE, current_time):
            return None
            
        if stats.convergence_rate < self.convergence_threshold:
            return Alert(
                type=AlertType.CONVERGENCE_ISSUE,
                severity=AlertSeverity.ERROR,
                message=f"Slow convergence detected: {stats.convergence_rate:.2f}",
                timestamp=current_time,
                metrics={"convergence_rate": stats.convergence_rate},
                suggestions=[
                    "Adjust learning rate",
                    "Check for gradient issues",
                    "Validate data quality"
                ]
            )
            
        return None
        
    def _check_scaling(
        self,
        stats: AdvancedProfileStats,
        current_time: float
    ) -> Optional[Alert]:
        """Check scaling issues."""
        if self._is_cooldown(AlertType.SCALING_ISSUE, current_time):
            return None
            
        if stats.scaling_efficiency < 0.7:  # Less than 70% scaling efficiency
            return Alert(
                type=AlertType.SCALING_ISSUE,
                severity=AlertSeverity.WARNING,
                message=f"Poor scaling efficiency: {stats.scaling_efficiency:.1%}",
                timestamp=current_time,
                metrics={"scaling_efficiency": stats.scaling_efficiency},
                suggestions=[
                    "Optimize data parallelism",
                    "Check load balancing",
                    "Adjust batch size distribution"
                ]
            )
            
        return None
        
    def _check_pipeline(
        self,
        stats: AdvancedProfileStats,
        current_time: float
    ) -> Optional[Alert]:
        """Check pipeline issues."""
        if self._is_cooldown(AlertType.PIPELINE_STALL, current_time):
            return None
            
        if stats.pipeline_efficiency < 0.8:  # Less than 80% pipeline efficiency
            return Alert(
                type=AlertType.PIPELINE_STALL,
                severity=AlertSeverity.WARNING,
                message=f"Pipeline inefficiency detected: {stats.pipeline_efficiency:.1%}",
                timestamp=current_time,
                metrics={"pipeline_efficiency": stats.pipeline_efficiency},
                suggestions=[
                    "Adjust pipeline chunks",
                    "Balance stage workloads",
                    "Check for bottlenecks"
                ]
            )
            
        return None
        
    def _is_cooldown(self, alert_type: AlertType, current_time: float) -> bool:
        """Check if alert is in cooldown period."""
        last_time = self.last_alerts.get(alert_type, 0)
        if current_time - last_time < self.alert_cooldown:
            return True
            
        self.last_alerts[alert_type] = current_time
        return False 