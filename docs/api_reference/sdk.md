# SDK API Reference

## PyRAI SDK

### Client

Main SDK client for interacting with PyRAI services.

#### Methods

##### `__init__(api_key: str, base_url: str = "https://api.pyrai.xyz")`
Initialize SDK client.

**Parameters:**
- `api_key`: API authentication key
- `base_url`: API base URL

##### `create_distillation(teacher_model: ModelConfig, student_model: ModelConfig, config: DistillationConfig) -> str`
Create new distillation task.

**Parameters:**
- `teacher_model`: Teacher model configuration
- `student_model`: Student model configuration
- `config`: Distillation configuration

**Returns:**
- Task ID

### Model Configuration

#### ModelConfig

Configuration for AI models.

```python
class ModelConfig:
    """Model configuration."""
    name: str
    architecture: str
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]]
```

### Training Configuration

#### TrainingConfig

Configuration for model training.

```python
class TrainingConfig:
    """Training configuration."""
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float
    distributed: bool = False
    mixed_precision: bool = False
```

### Resource Configuration

#### ResourceConfig

Configuration for compute resources.

```python
class ResourceConfig:
    """Resource configuration."""
    cpu_cores: int
    memory_gb: float
    gpu_count: Optional[int]
    gpu_type: Optional[str]
```
