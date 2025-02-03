# Deployment Guide

Learn how to deploy PyRAI models in production environments.

## Kubernetes Deployment

1. Create deployment configuration:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pyrai-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pyrai-model
  template:
    metadata:
      labels:
        app: pyrai-model
    spec:
      containers:
      - name: model-server
        image: pyraixyz/model-server:latest
        resources:
          requests:
            memory: "4Gi"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            nvidia.com/gpu: 1
```

2. Deploy using PyRAI client:
```python
deployment = client.deploy(
    model="model1",
    platform="kubernetes",
    config="k8s-config.yaml"
)
```

## Auto-Scaling

Configure auto-scaling based on metrics:
```python
deployment.configure_scaling(
    min_replicas=1,
    max_replicas=10,
    target_cpu_utilization=80,
    target_gpu_utilization=70
)
```

## Monitoring

Monitor deployment metrics:
```python
metrics = deployment.get_metrics()
print(f"Request latency: {metrics['latency_ms']}ms")
print(f"Throughput: {metrics['requests_per_second']} RPS")
``` 