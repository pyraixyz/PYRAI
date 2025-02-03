# PyRAI API Reference

## Engine

The main interface for model training and deployment.

```python
class Engine:
    def __init__(self, backend: ModelBackend, config: Config):
        """Initialize engine.
        
        Args:
            backend: Model backend
            config: Configuration
        """
        pass
        
    def train(self, model, train_data, val_data=None, metrics=None):
        """Train model.
        
        Args:
            model: Model to train
            train_data: Training data
            val_data: Validation data
            metrics: Training metrics
            
        Returns:
            Training history
        """
        pass
        
    def deploy(self, model, model_id, version=None):
        """Deploy model.
        
        Args:
            model: Model to deploy
            model_id: Model identifier
            version: Model version
            
        Returns:
            Deployment instance
        """
        pass
```

## Configuration

Configuration management system.

```python
class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        pass
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        pass
```

## Metrics

Metrics computation system.

```python
class MetricCollection:
    def __init__(self, metrics: Dict[str, Metric]):
        """Initialize metric collection.
        
        Args:
            metrics: Dictionary of metrics
        """
        pass
        
    def update(self, preds: Any, targets: Any):
        """Update metrics.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        pass
        
    def compute(self) -> Dict[str, float]:
        """Compute metrics.
        
        Returns:
            Dictionary of metric values
        """
        pass
```

## Command Line Interface

PyRAI provides several command line tools:

### Model Analysis
```bash
pyrai analyze MODEL_PATH --input-size INPUT_SIZE
```

### Model Export
```bash
pyrai export MODEL_PATH OUTPUT_PATH --format {onnx,torchscript} --input-size INPUT_SIZE
```

### Model Benchmark
```bash
pyrai benchmark MODEL_PATH [--output-dir DIR] [--batch-sizes SIZES] [--iterations N]
``` 