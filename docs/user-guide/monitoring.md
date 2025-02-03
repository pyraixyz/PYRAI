# Monitoring Guide

## Basic Monitoring

### Metrics Collection

```python
from pyrai import Engine
from pyrai.monitoring import MetricsCollector

# Initialize metrics collector
metrics = MetricsCollector()

# Get basic metrics
basic_metrics = metrics.get_metrics(
    model_id="my_model",
    version="v1"
)

# Get specific metrics
latency = metrics.get_metric(
    model_id="my_model",
    metric_name="latency",
    aggregation="p95"
)
```

### Performance Monitoring

```python
# Monitor performance metrics
performance_metrics = metrics.monitor_performance(
    model_id="my_model",
    metrics=[
        "latency",
        "throughput",
        "error_rate",
        "cpu_usage",
        "memory_usage"
    ],
    interval="1m"
)
```

## Advanced Monitoring

### Custom Metrics

```python
from pyrai.monitoring import CustomMetric

# Define custom metric
@CustomMetric(name="custom_accuracy")
def calculate_accuracy(predictions, targets):
    return sum(p == t for p, t in zip(predictions, targets)) / len(predictions)

# Register custom metric
metrics.register_metric(calculate_accuracy)

# Collect custom metrics
custom_metrics = metrics.get_custom_metrics(
    model_id="my_model",
    metric_names=["custom_accuracy"]
)
```

### Alerts

```python
from pyrai.monitoring import Alert

# Configure alert
alert = Alert(
    name="high_latency",
    condition="latency > 100ms",
    duration="5m",
    severity="warning"
)

# Add alert actions
alert.add_action(
    "slack",
    webhook_url="https://hooks.slack.com/..."
)

# Register alert
metrics.register_alert(alert)
```

## Logging

### Log Collection

```python
from pyrai.monitoring import LogCollector

# Initialize log collector
logs = LogCollector()

# Get model logs
model_logs = logs.get_logs(
    model_id="my_model",
    start_time="2023-01-01T00:00:00Z",
    end_time="2023-01-02T00:00:00Z",
    level="ERROR"
)

# Stream logs
for log in logs.stream_logs(model_id="my_model"):
    print(log)
```

### Log Analysis

```python
# Analyze logs
analysis = logs.analyze(
    model_id="my_model",
    analysis_type="error_patterns",
    time_range="24h"
)

# Get error statistics
error_stats = logs.get_error_stats(
    model_id="my_model",
    group_by="error_type"
)
```

## Visualization

### Dashboards

```python
from pyrai.monitoring import Dashboard

# Create dashboard
dashboard = Dashboard(name="Model Performance")

# Add panels
dashboard.add_panel(
    title="Latency",
    metric="latency",
    visualization="line"
)

dashboard.add_panel(
    title="Error Rate",
    metric="error_rate",
    visualization="gauge"
)

# Save dashboard
dashboard.save()
``` 