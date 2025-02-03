from typing import Any, Dict, List, Optional, Union, Callable
import time
from dataclasses import dataclass
from enum import Enum
from .monitor import RuntimeStats

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Performance alert."""
    timestamp: float
    severity: AlertSeverity
    message: str
    metrics: Dict[str, float]
    source: str

class AlertRule:
    """Performance alert rule."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown: float = 300.0  # 5 minutes
    ):
        """Initialize alert rule.
        
        Args:
            name: Rule name
            condition: Alert condition function
            message: Alert message
            severity: Alert severity
            cooldown: Minimum time between alerts
        """
        self.name = name
        self.condition = condition
        self.message = message
        self.severity = severity
        self.cooldown = cooldown
        self.last_triggered = 0.0
        
    def check(self, stats: Dict[str, Any]) -> Optional[Alert]:
        """Check if alert should be triggered."""
        current_time = time.time()
        if current_time - self.last_triggered < self.cooldown:
            return None
            
        if self.condition(stats):
            self.last_triggered = current_time
            return Alert(
                timestamp=current_time,
                severity=self.severity,
                message=self.message,
                metrics={k: v for k, v in stats.items() if isinstance(v, (int, float))},
                source=self.name
            )
        return None

class AlertManager:
    """Performance alert manager."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self.handlers: List[Callable[[Alert], None]] = []
        
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules.append(rule)
        
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler."""
        self.handlers.append(handler)
        
    def check_alerts(self, stats: Dict[str, Any]):
        """Check all alert rules."""
        for rule in self.rules:
            alert = rule.check(stats)
            if alert:
                self.alerts.append(alert)
                for handler in self.handlers:
                    handler(alert)
                    
    @classmethod
    def create_default(cls) -> 'AlertManager':
        """Create alert manager with default rules."""
        manager = cls()
        
        # Add default rules
        manager.add_rule(AlertRule(
            name="high_gpu_memory",
            condition=lambda stats: any(mem > 90 for mem in stats["gpu_memory"]),
            message="GPU memory usage exceeds 90%",
            severity=AlertSeverity.WARNING
        ))
        
        manager.add_rule(AlertRule(
            name="low_gpu_utilization",
            condition=lambda stats: stats["gpu_utilization"] < 50,
            message="GPU utilization below 50%",
            severity=AlertSeverity.INFO
        ))
        
        manager.add_rule(AlertRule(
            name="high_latency",
            condition=lambda stats: stats["latency"] > 1.0,
            message="Training latency exceeds 1 second",
            severity=AlertSeverity.WARNING
        ))
        
        manager.add_rule(AlertRule(
            name="throughput_drop",
            condition=lambda stats: (
                stats["throughput"] < 0.7 * getattr(stats, "peak_throughput", float("inf"))
            ),
            message="Training throughput dropped by more than 30%",
            severity=AlertSeverity.ERROR
        ))
        
        # Add default handlers
        manager.add_handler(lambda alert: print(
            f"[{alert.severity.value.upper()}] {alert.message}"
        ))
        
        return manager 