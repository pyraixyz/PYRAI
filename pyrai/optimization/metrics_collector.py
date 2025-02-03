from typing import Any, Dict, List, Optional, Union, Tuple
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .monitor import RuntimeStats
from .ab_tester import TestResult
from .parallel_tester import TestStatus

class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

@dataclass
class MetricValue:
    """Metric value details."""
    type: MetricType
    value: Union[int, float, List[float]]
    timestamp: float
    labels: Dict[str, str]

class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(
        self,
        history_window: int = 1000,
        aggregation_interval: float = 60.0  # 1 minute
    ):
        """Initialize collector.
        
        Args:
            history_window: Maximum metrics to keep
            aggregation_interval: Interval for aggregation
        """
        self.history_window = history_window
        self.aggregation_interval = aggregation_interval
        self.metrics: Dict[str, List[MetricValue]] = {}
        self.last_aggregation = time.time()
        
    def record_metric(
        self,
        name: str,
        value: Union[int, float, List[float]],
        type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
            
        metric = MetricValue(
            type=type,
            value=value,
            timestamp=time.time(),
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
        
        # Trim history if needed
        if len(self.metrics[name]) > self.history_window:
            self.metrics[name] = self.metrics[name][-self.history_window:]
            
        # Check if should aggregate
        if time.time() - self.last_aggregation >= self.aggregation_interval:
            self._aggregate_metrics()
            
    def record_test_metrics(
        self,
        test_id: str,
        stats: RuntimeStats,
        result: Optional[TestResult] = None
    ):
        """Record test-related metrics."""
        labels = {"test_id": test_id}
        
        # Record performance metrics
        self.record_metric(
            "test_throughput",
            stats.throughput,
            MetricType.GAUGE,
            labels
        )
        
        self.record_metric(
            "test_latency",
            stats.latency,
            MetricType.GAUGE,
            labels
        )
        
        self.record_metric(
            "test_gpu_utilization",
            np.mean(stats.gpu_utilization),
            MetricType.GAUGE,
            labels
        )
        
        self.record_metric(
            "test_gpu_memory",
            np.mean(stats.gpu_memory),
            MetricType.GAUGE,
            labels
        )
        
        # Record test results if available
        if result:
            self.record_metric(
                "test_improvement",
                result.improvement,
                MetricType.GAUGE,
                labels
            )
            
            self.record_metric(
                "test_confidence",
                result.confidence,
                MetricType.GAUGE,
                labels
            )
            
    def get_metric(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """Get metric values."""
        if name not in self.metrics:
            return []
            
        values = self.metrics[name]
        
        # Filter by time range
        if start_time:
            values = [v for v in values if v.timestamp >= start_time]
        if end_time:
            values = [v for v in values if v.timestamp <= end_time]
            
        # Filter by labels
        if labels:
            values = [
                v for v in values
                if all(v.labels.get(k) == v for k, v in labels.items())
            ]
            
        return values
        
    def get_summary(
        self,
        name: str,
        window: Optional[float] = None
    ) -> Dict[str, float]:
        """Get metric summary statistics."""
        end_time = time.time()
        start_time = end_time - window if window else None
        
        values = self.get_metric(name, start_time, end_time)
        if not values:
            return {}
            
        raw_values = [v.value for v in values]
        
        return {
            "count": len(raw_values),
            "min": np.min(raw_values),
            "max": np.max(raw_values),
            "mean": np.mean(raw_values),
            "std": np.std(raw_values),
            "p50": np.percentile(raw_values, 50),
            "p90": np.percentile(raw_values, 90),
            "p99": np.percentile(raw_values, 99)
        }
        
    def _aggregate_metrics(self):
        """Aggregate old metrics."""
        current_time = time.time()
        self.last_aggregation = current_time
        
        for name, values in self.metrics.items():
            # Find values to aggregate
            cutoff_time = current_time - self.aggregation_interval
            old_values = [v for v in values if v.timestamp < cutoff_time]
            
            if not old_values:
                continue
                
            # Group by labels
            by_labels: Dict[str, List[MetricValue]] = {}
            for value in old_values:
                key = tuple(sorted(value.labels.items()))
                if key not in by_labels:
                    by_labels[key] = []
                by_labels[key].append(value)
                
            # Aggregate each group
            for labels, group in by_labels.items():
                if group[0].type == MetricType.COUNTER:
                    # Sum counters
                    total = sum(v.value for v in group)
                    self.record_metric(
                        f"{name}_total",
                        total,
                        MetricType.COUNTER,
                        dict(labels)
                    )
                    
                elif group[0].type == MetricType.GAUGE:
                    # Average gauges
                    avg = np.mean([v.value for v in group])
                    self.record_metric(
                        f"{name}_avg",
                        avg,
                        MetricType.GAUGE,
                        dict(labels)
                    )
                    
                else:  # HISTOGRAM
                    # Compute percentiles
                    values = [v.value for v in group]
                    for p in [50, 90, 99]:
                        percentile = np.percentile(values, p)
                        self.record_metric(
                            f"{name}_p{p}",
                            percentile,
                            MetricType.GAUGE,
                            dict(labels)
                        )
                        
            # Remove aggregated values
            self.metrics[name] = [
                v for v in values
                if v.timestamp >= cutoff_time
            ] 