from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import tensorflow as tf
from enum import Enum

class ParallelStrategy(Enum):
    """Model parallelism strategy."""
    PIPELINE = "pipeline"
    TENSOR = "tensor"
    HYBRID = "hybrid"

class ModelParallelWrapper:
    """Wrapper for model parallel training."""
    
    def __init__(
        self,
        strategy: ParallelStrategy,
        num_gpus: int,
        chunk_size: Optional[int] = None
    ):
        """Initialize model parallel wrapper.
        
        Args:
            strategy: Parallelism strategy
            num_gpus: Number of GPUs to use
            chunk_size: Size of pipeline chunks
        """
        self.strategy = strategy
        self.num_gpus = num_gpus
        self.chunk_size = chunk_size
        
    def parallelize(
        self,
        model: Union[torch.nn.Module, tf.keras.Model]
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Parallelize model across devices."""
        if isinstance(model, torch.nn.Module):
            return self._parallelize_torch(model)
        else:
            return self._parallelize_tf(model)
            
    def _parallelize_torch(
        self,
        model: torch.nn.Module
    ) -> torch.nn.Module:
        """Parallelize PyTorch model."""
        if self.strategy == ParallelStrategy.PIPELINE:
            return self._pipeline_parallel_torch(model)
        elif self.strategy == ParallelStrategy.TENSOR:
            return self._tensor_parallel_torch(model)
        else:  # Hybrid
            return self._hybrid_parallel_torch(model)
            
    def _pipeline_parallel_torch(
        self,
        model: torch.nn.Module
    ) -> torch.nn.Module:
        """Parallelize PyTorch model."""
        if self.strategy == ParallelStrategy.PIPELINE:
            return self._pipeline_parallel_torch(model)
        elif self.strategy == ParallelStrategy.TENSOR:
            return self._tensor_parallel_torch(model)
        else:  # Hybrid
            return self._hybrid_parallel_torch(model)
            
    def _tensor_parallel_torch(
        self,
        model: torch.nn.Module
    ) -> torch.nn.Module:
        """Apply tensor parallelism to PyTorch model."""
        # Split large linear and conv layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.weight.size(0) >= 512:  # Only split large layers
                    split_size = module.weight.size(0) // self.num_gpus
                    splits = []
                    for i in range(self.num_gpus):
                        device = torch.device(f'cuda:{i}')
                        split = type(module)(
                            module.in_features,
                            split_size,
                            bias=module.bias is not None
                        ).to(device)
                        splits.append(split)
                    setattr(model, name, TensorParallel(splits))
                    
        return model 

    def _hybrid_parallel_torch(
        self,
        model: torch.nn.Module
    ) -> torch.nn.Module:
        """Apply hybrid parallelism to PyTorch model."""
        # Split model into pipeline stages
        stages = []
        tensor_parallel_layers = {}
        current_stage = []
        params_per_stage = sum(p.numel() for p in model.parameters()) // self.num_gpus
        current_params = 0
        
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.size(0) >= 512:
                # Apply tensor parallelism to large layers
                split_size = layer.weight.size(0) // (self.num_gpus // 2)
                splits = []
                for i in range(self.num_gpus // 2):
                    device = torch.device(f'cuda:{i}')
                    split = type(layer)(
                        layer.in_features,
                        split_size,
                        bias=layer.bias is not None
                    ).to(device)
                    splits.append(split)
                tensor_parallel_layers[name] = splits
            else:
                layer_params = sum(p.numel() for p in layer.parameters())
                if current_params + layer_params > params_per_stage and current_stage:
                    stages.append(nn.Sequential(*current_stage))
                    current_stage = []
                    current_params = 0
                current_stage.append(layer)
                current_params += layer_params
                
        if current_stage:
            stages.append(nn.Sequential(*current_stage))
            
        # Assign pipeline stages to remaining devices
        for i, stage in enumerate(stages):
            device = torch.device(f'cuda:{i + self.num_gpus//2}')
            stage.to(device)
            
        return HybridParallel(stages, tensor_parallel_layers, self.chunk_size)
        
    def _parallelize_tf(
        self,
        model: tf.keras.Model
    ) -> tf.keras.Model:
        """Parallelize TensorFlow model."""
        if self.strategy == ParallelStrategy.PIPELINE:
            return self._pipeline_parallel_tf(model)
        elif self.strategy == ParallelStrategy.TENSOR:
            return self._tensor_parallel_tf(model)
        else:  # Hybrid
            return self._hybrid_parallel_tf(model)
            
    def _pipeline_parallel_tf(
        self,
        model: tf.keras.Model
    ) -> tf.keras.Model:
        """Apply pipeline parallelism to TensorFlow model."""
        # Split model into stages
        layers = model.layers
        num_layers = len(layers)
        layers_per_stage = num_layers // self.num_gpus
        stages = []
        
        for i in range(self.num_gpus):
            start_idx = i * layers_per_stage
            end_idx = start_idx + layers_per_stage if i < self.num_gpus - 1 else num_layers
            stage_layers = layers[start_idx:end_idx]
            
            with tf.device(f'/gpu:{i}'):
                stage = tf.keras.Sequential(stage_layers)
                stages.append(stage)
                
        return TFPipelineParallel(stages, self.chunk_size)
        
    def _tensor_parallel_tf(
        self,
        model: tf.keras.Model
    ) -> tf.keras.Model:
        """Apply tensor parallelism to TensorFlow model."""
        # Create mirrored strategy for tensor parallelism
        strategy = tf.distribute.MirroredStrategy(
            devices=[f'/gpu:{i}' for i in range(self.num_gpus)]
        )
        
        with strategy.scope():
            parallel_model = tf.keras.models.clone_model(model)
            parallel_model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics
            )
            
        return parallel_model

class PipelineParallel(nn.Module):
    """Pipeline parallel model implementation."""
    
    def __init__(self, stages: List[nn.Module], chunk_size: Optional[int] = None):
        """Initialize pipeline parallel model.
        
        Args:
            stages: List of model stages
            chunk_size: Size of micro-batches
        """
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.chunk_size = chunk_size or 1
        self.num_stages = len(stages)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pipeline parallelism."""
        batch_size = x.size(0)
        chunks = torch.chunk(x, self.chunk_size, dim=0)
        outputs = []
        
        for chunk in chunks:
            # Pipeline forward pass
            stage_output = chunk
            for stage in self.stages:
                stage_output = stage(stage_output)
            outputs.append(stage_output)
            
        return torch.cat(outputs, dim=0)

class TensorParallel(nn.Module):
    """Tensor parallel layer implementation."""
    
    def __init__(self, splits: List[nn.Module]):
        """Initialize tensor parallel layer.
        
        Args:
            splits: List of layer splits
        """
        super().__init__()
        self.splits = nn.ModuleList(splits)
        self.num_splits = len(splits)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallelism."""
        # Split input across devices
        chunks = x.chunk(self.num_splits, dim=-1)
        outputs = []
        
        # Process chunks in parallel
        for split, chunk in zip(self.splits, chunks):
            outputs.append(split(chunk))
            
        # Gather results
        return torch.cat(outputs, dim=-1)

class HybridParallel(nn.Module):
    """Hybrid parallel model implementation."""
    
    def __init__(
        self,
        pipeline_stages: List[nn.Module],
        tensor_parallel_layers: Dict[str, List[nn.Module]],
        chunk_size: Optional[int] = None
    ):
        """Initialize hybrid parallel model.
        
        Args:
            pipeline_stages: List of pipeline stages
            tensor_parallel_layers: Dictionary of tensor parallel layers
            chunk_size: Size of pipeline chunks
        """
        super().__init__()
        self.pipeline = PipelineParallel(pipeline_stages, chunk_size)
        self.tensor_parallel = nn.ModuleDict({
            name: TensorParallel(splits)
            for name, splits in tensor_parallel_layers.items()
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hybrid parallelism."""
        return self.pipeline(x)

class TFPipelineParallel(tf.keras.Model):
    """TensorFlow pipeline parallel model implementation."""
    
    def __init__(self, stages: List[tf.keras.Model], chunk_size: Optional[int] = None):
        """Initialize pipeline parallel model."""
        super().__init__()
        self.stages = stages
        self.chunk_size = chunk_size or 1
        self.num_stages = len(stages)
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with pipeline parallelism."""
        batch_size = tf.shape(inputs)[0]
        chunks = tf.split(inputs, self.chunk_size)
        outputs = []
        
        for chunk in chunks:
            # Pipeline forward pass
            stage_output = chunk
            for stage in self.stages:
                stage_output = stage(stage_output, training=training)
            outputs.append(stage_output)
            
        return tf.concat(outputs, axis=0) 