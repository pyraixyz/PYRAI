# Quick Start Guide

This guide will help you get started with PyRAI for AI model training and deployment.

## Basic Usage

```python
from pyrai import PyRAIClient

# Initialize client
client = PyRAIClient(
    api_key="your-api-key",
    chain="solana"
)

# Load pre-trained model
model = client.load_model("resnet50")

# Configure training
config = {
    "batch_size": 128,
    "epochs": 10,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "distributed": True
}

# Start training
task = client.train(
    model=model,
    dataset="imagenet",
    config=config
)

# Monitor progress
client.monitor(task.id)
```

## Blockchain Integration

```python
# Configure blockchain
client.configure_chain(
    chain="solana",
    network="devnet",
    wallet="path/to/wallet.json"
)

# Tokenize resources
tokens = client.mint_compute_tokens(
    cpu_hours=100,
    gpu_hours=50
)

# Start training with tokens
task = client.train(
    model=model,
    dataset="imagenet",
    config=config,
    payment=tokens
)
```

## Resource Management

```python
# Configure resource requirements
resources = {
    "cpu_cores": 8,
    "memory_gb": 32,
    "gpu_count": 4
}

# Deploy model
deployment = client.deploy(
    model=model,
    resources=resources,
    auto_scale=True
)

# Make predictions
predictions = deployment.predict(data)
``` 