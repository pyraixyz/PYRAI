from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import time
from dataclasses import dataclass
from enum import Enum
from .metrics_collector import MetricsCollector, MetricType
from .monitor import RuntimeStats

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertState(Enum):
    """Alert states."""
    INACTIVE = "inactive"
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"

@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    metric: str
    condition: Callable[[float], bool]
    level: AlertLevel
    labels: Dict[str, str]
    description: str
    for_duration: float = 60.0  # Duration before firing
    resolve_duration: float = 60.0  # Duration before resolving

@dataclass
class Alert:
    """Active alert details."""
    rule: AlertRule
    state: AlertState
    value: float
    start_time: float
    last_update: float
    annotations: Dict[str, str]

class AlertManager:
    """Manages metric-based alerts."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        check_interval: float = 10.0
    ):
        """Initialize alert manager.
        
        Args:
            collector: Metrics collector
            check_interval: Alert check interval
        """
        self.collector = collector
        self.check_interval = check_interval
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_check = 0.0
        
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules.append(rule)
        
    def check_alerts(self) -> List[Alert]:
        """Check alert rules."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return []
            
        self.last_check = current_time
        new_alerts = []
        
        for rule in self.rules:
            # Get latest metric value
            metrics = self.collector.get_metric(
                rule.metric,
                start_time=current_time - max(
                    rule.for_duration,
                    rule.resolve_duration
                )
            )
            
            if not metrics:
                continue
                
            latest = metrics[-1].value
            if isinstance(latest, (list, tuple)):
                latest = float(latest[0])
                
            # Check condition
            condition_met = rule.condition(latest)
            
            # Get or create alert
            alert = self.active_alerts.get(rule.name)
            
            if condition_met:
                if not alert:
                    # New alert
                    alert = Alert(
                        rule=rule,
                        state=AlertState.PENDING,
                        value=latest,
                        start_time=current_time,
                        last_update=current_time,
                        annotations={}
                    )
                    self.active_alerts[rule.name] = alert
                    
                else:
                    # Update existing alert
                    alert.value = latest
                    alert.last_update = current_time
                    
                    # Check if should fire
                    if (
                        alert.state == AlertState.PENDING and
                        current_time - alert.start_time >= rule.for_duration
                    ):
                        alert.state = AlertState.FIRING
                        new_alerts.append(alert)
                        
            else:
                if alert:
                    # Check if should resolve
                    if current_time - alert.last_update >= rule.resolve_duration:
                        alert.state = AlertState.RESOLVED
                        new_alerts.append(alert)
                        self.alert_history.append(alert)
                        del self.active_alerts[rule.name]
                        
        return new_alerts
        
    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """Get active alerts."""
        alerts = list(self.active_alerts.values())
        if level:
            alerts = [a for a in alerts if a.rule.level == level]
        return alerts
        
    def get_alert_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Alert]:
        """Get alert history."""
        alerts = self.alert_history
        
        if start_time:
            alerts = [a for a in alerts if a.start_time >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.start_time <= end_time]
            
        return alerts
        
    def create_basic_rules(self):
        """Create basic alert rules."""
        # Memory usage rule
        self.add_rule(AlertRule(
            name="high_memory_usage",
            metric="test_gpu_memory",
            condition=lambda x: x > 0.9,
            level=AlertLevel.WARNING,
            labels={"type": "resource"},
            description="GPU memory usage above 90%",
            for_duration=300.0  # 5 minutes
        ))
        
        # Low GPU utilization rule
        self.add_rule(AlertRule(
            name="low_gpu_utilization",
            metric="test_gpu_utilization",
            condition=lambda x: x < 0.3,
            level=AlertLevel.WARNING,
            labels={"type": "resource"},
            description="GPU utilization below 30%",
            for_duration=600.0  # 10 minutes
        ))
        
        # High latency rule
        self.add_rule(AlertRule(
            name="high_latency",
            metric="test_latency",
            condition=lambda x: x > 1.0,
            level=AlertLevel.CRITICAL,
            labels={"type": "performance"},
            description="Training latency above 1 second",
            for_duration=300.0  # 5 minutes
        ))
        
        # Low throughput rule
        self.add_rule(AlertRule(
            name="low_throughput",
            metric="test_throughput",
            condition=lambda x: x < 50,
            level=AlertLevel.WARNING,
            labels={"type": "performance"},
            description="Training throughput below 50 samples/sec",
            for_duration=600.0  # 10 minutes
        )) 