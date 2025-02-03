from typing import Any, Dict, List, Optional, Union, Tuple
import time
import json
from pathlib import Path
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from .metrics_collector import MetricsCollector, MetricType, MetricValue

class MetricsExporter:
    """Exports metrics to various formats/systems."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        export_dir: str = "metrics",
        prometheus_port: int = 8000
    ):
        """Initialize exporter.
        
        Args:
            collector: Metrics collector
            export_dir: Directory for exported files
            prometheus_port: Prometheus exporter port
        """
        self.collector = collector
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.prometheus_port = prometheus_port
        self.prometheus_metrics: Dict[str, Any] = {}
        
    def start_prometheus_exporter(self):
        """Start Prometheus metrics exporter."""
        # Create Prometheus metrics
        for name, values in self.collector.metrics.items():
            if not values:
                continue
                
            metric_type = values[0].type
            if metric_type == MetricType.COUNTER:
                self.prometheus_metrics[name] = Counter(
                    name,
                    "Counter metric"
                )
            elif metric_type == MetricType.GAUGE:
                self.prometheus_metrics[name] = Gauge(
                    name,
                    "Gauge metric"
                )
            else:  # HISTOGRAM
                self.prometheus_metrics[name] = Histogram(
                    name,
                    "Histogram metric"
                )
                
        # Start server
        start_http_server(self.prometheus_port)
        
    def update_prometheus_metrics(self):
        """Update Prometheus metrics."""
        for name, values in self.collector.metrics.items():
            if not values or name not in self.prometheus_metrics:
                continue
                
            metric = self.prometheus_metrics[name]
            latest = values[-1]
            
            if latest.type == MetricType.COUNTER:
                metric.inc(latest.value)
            elif latest.type == MetricType.GAUGE:
                metric.set(latest.value)
            else:  # HISTOGRAM
                metric.observe(latest.value)
                
    def export_json(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Export metrics to JSON file."""
        timestamp = int(time.time())
        filename = self.export_dir / f"metrics_{timestamp}.json"
        
        data = {}
        for name, values in self.collector.metrics.items():
            metrics = self.collector.get_metric(name, start_time, end_time)
            data[name] = [
                {
                    "timestamp": m.timestamp,
                    "value": m.value,
                    "type": m.type.value,
                    "labels": m.labels
                }
                for m in metrics
            ]
            
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
            
        return filename
        
    def export_csv(
        self,
        metric_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Export single metric to CSV file."""
        timestamp = int(time.time())
        filename = self.export_dir / f"{metric_name}_{timestamp}.csv"
        
        metrics = self.collector.get_metric(metric_name, start_time, end_time)
        if not metrics:
            return None
            
        # Get all unique label keys
        label_keys = set()
        for m in metrics:
            label_keys.update(m.labels.keys())
            
        # Write CSV
        with open(filename, "w") as f:
            # Write header
            header = ["timestamp", "value", "type"] + sorted(label_keys)
            f.write(",".join(header) + "\n")
            
            # Write data
            for m in metrics:
                row = [
                    str(m.timestamp),
                    str(m.value),
                    m.type.value
                ]
                row.extend(
                    str(m.labels.get(k, ""))
                    for k in sorted(label_keys)
                )
                f.write(",".join(row) + "\n")
                
        return filename 