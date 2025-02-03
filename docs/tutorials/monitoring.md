# Monitoring and Metrics Guide

## Training Metrics

### Performance Monitoring

```python
from pyrai.monitoring import MetricsCollector

# Initialize collector
collector = MetricsCollector()

# Configure metrics
collector.configure(
    metrics=[
        "loss",
        "accuracy",
        "gpu_utilization",
        "memory_usage",
        "throughput"
    ],
    interval=10  # seconds
)

# Start collection
collector.start()

# Training loop
for epoch in range(num_epochs):
    collector.log_metrics({
        "loss": current_loss,
        "accuracy": current_accuracy
    })

# Get metrics summary
summary = collector.get_summary()
```

### Resource Monitoring

```python
from pyrai.monitoring import ResourceMonitor

# Initialize monitor
monitor = ResourceMonitor()

# Configure alerts
monitor.set_alerts({
    "gpu_memory": {
        "threshold": 0.9,  # 90% usage
        "action": "notify"
    },
    "cpu_utilization": {
        "threshold": 0.8,
        "action": "scale"
    }
})

# Start monitoring
monitor.start()
```

## Visualization

### Real-time Dashboard

```python
from pyrai.monitoring import Dashboard

# Create dashboard
dashboard = Dashboard()

# Add metrics panels
dashboard.add_panel(
    name="Training Progress",
    metrics=["loss", "accuracy"],
    type="line"
)

dashboard.add_panel(
    name="Resource Usage",
    metrics=["gpu_util", "memory_util"],
    type="gauge"
)

# Start dashboard
dashboard.serve(port=8080)
```

### Export Reports

```python
# Generate training report
report = collector.generate_report(
    format="pdf",
    sections=[
        "training_metrics",
        "resource_usage",
        "model_performance"
    ]
)

# Export metrics
collector.export_metrics(
    format="csv",
    path="./metrics.csv"
)
``` 