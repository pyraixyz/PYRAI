# Distributed Training Example

This example demonstrates how to use PyRAI for distributed training across multiple nodes.

## Setup

```python
from pyrai.core import DistributedCoordinator
from pyrai.core.throughput import ThroughputConfig

# Configure distributed training
config = ThroughputConfig(
    batch_size=128,
    gradient_accumulation_steps=4,
    mixed_precision=True
)

# Initialize coordinator
coordinator = DistributedCoordinator()

# Register nodes
coordinator.register_node("node1", NodeConfig(
    ip="192.168.1.100",
    port=29500,
    gpu_count=4,
    memory_gb=32
))
```

## Training

```python
# Start training
results = coordinator.train_distributed(
    model_config=model_config,
    data_config=data_config,
    throughput_config=config
)

# Monitor progress
for metrics in coordinator.stream_metrics():
    print(f"Loss: {metrics['loss']:.4f}")
``` 