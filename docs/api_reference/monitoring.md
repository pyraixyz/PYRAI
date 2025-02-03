# Monitoring API Reference

## Metrics Collection

### MetricsCollector

Collects and manages training metrics.

#### Methods

##### `__init__(storage_path: Optional[str] = None)`
Initialize metrics collector.

**Parameters:**
- `storage_path`: Path to store metrics data

##### `configure(metrics: List[str], interval: int)`
Configure metrics collection.

**Parameters:**
- `metrics`: List of metrics to collect
- `interval`: Collection interval in seconds

### ResourceMonitor

Monitors system resource usage.

#### Methods

##### `__init__()`
Initialize resource monitor.

##### `set_alerts(config: Dict[str, Dict[str, Any]])`
Configure monitoring alerts.

**Parameters:**
- `config`: Alert configuration

## Visualization

### Dashboard

Real-time metrics dashboard.

#### Methods

##### `__init__(title: str = "PyRAI Dashboard")`
Initialize dashboard.

**Parameters:**
- `title`: Dashboard title

##### `add_panel(name: str, metrics: List[str], type: str)`
Add metrics panel to dashboard.

**Parameters:**
- `name`: Panel name
- `metrics`: Metrics to display
- `type`: Panel type ("line", "gauge", "bar") 