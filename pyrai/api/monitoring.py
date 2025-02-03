"""API monitoring and metrics collection."""

from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Any
import time
import psutil
import structlog

logger = structlog.get_logger()

class APIMetrics:
    """API metrics collector."""
    
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            "api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status"]
        )
        
        self.request_latency = Histogram(
            "api_request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"]
        )
        
        # Task metrics
        self.active_tasks = Gauge(
            "api_active_tasks",
            "Number of active tasks"
        )
        
        self.task_duration = Histogram(
            "api_task_duration_seconds",
            "Task processing duration in seconds",
            ["task_type"]
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            "api_cpu_usage_percent",
            "CPU usage percentage"
        )
        
        self.memory_usage = Gauge(
            "api_memory_usage_bytes",
            "Memory usage in bytes"
        )
        
    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record API request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.request_latency.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    def update_task_metrics(self, active_tasks: int):
        """Update task metrics."""
        self.active_tasks.set(active_tasks)
        
    def record_task_duration(self, task_type: str, duration: float):
        """Record task processing duration."""
        self.task_duration.labels(task_type=task_type).observe(duration)
        
    def update_system_metrics(self):
        """Update system resource metrics."""
        self.cpu_usage.set(psutil.cpu_percent())
        self.memory_usage.set(psutil.Process().memory_info().rss)

class MetricsCollector:
    """Metrics collection and monitoring."""
    
    def __init__(self):
        self.metrics = APIMetrics()
        
    async def collect_request_metrics(self, request, call_next):
        """Collect request metrics middleware."""
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        self.metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        return response
        
    def start_monitoring(self):
        """Start system metrics collection."""
        async def collect_system_metrics():
            while True:
                self.metrics.update_system_metrics()
                await asyncio.sleep(60)  # Update every minute
                
        asyncio.create_task(collect_system_metrics())

metrics_collector = MetricsCollector() 