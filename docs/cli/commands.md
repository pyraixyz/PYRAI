# Command Line Interface

PyRAI provides a command-line interface for common operations.

## Basic Usage

```bash
# Train a model
pyrai train model.py data.csv --config config.yml

# Deploy a model
pyrai deploy my_model v1 --config deploy.yml

# Check model status
pyrai status my_model --version v1

# Monitor metrics
pyrai monitor my_model --metric latency --metric throughput
```

## Commands

### Train

Train a model using specified data.

```bash
pyrai train MODEL_PATH DATA_PATH [OPTIONS]

Options:
  -c, --config PATH    Configuration file
  -b, --backend TEXT   Model backend (pytorch/tensorflow)
  --help              Show this message and exit
```

Example configuration (config.yml):
```yaml
training:
  batch_size: 32
  epochs: 10
  optimizer:
    type: adam
    lr: 0.001
  scheduler:
    type: cosine
    T_max: 100
```

### Deploy

Deploy a trained model.

```bash
pyrai deploy MODEL_ID VERSION [OPTIONS]

Options:
  -c, --config PATH   Deployment configuration
  --help             Show this message and exit
```

Example configuration (deploy.yml):
```yaml
resources:
  cpu: "1"
  memory: "2Gi"
  gpu: "1"
scaling:
  min_replicas: 1
  max_replicas: 3
  target_cpu_utilization: 80
networking:
  port: 8000
  protocol: http
```

### Status

Check model status.

```bash
pyrai status MODEL_ID [OPTIONS]

Options:
  -v, --version TEXT   Model version
  --help              Show this message and exit
```

### Monitor

Monitor model metrics.

```bash
pyrai monitor MODEL_ID [OPTIONS]

Options:
  -v, --version TEXT   Model version
  -m, --metric TEXT   Metrics to display (multiple allowed)
  --help             Show this message and exit
``` 