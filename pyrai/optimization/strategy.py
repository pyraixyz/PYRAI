from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from .parallel import ParallelStrategy

@dataclass
class ModelProfile:
    """Model architecture profile."""
    total_params: int
    layer_sizes: List[int]
    communication_cost: float
    memory_requirement: float
    compute_intensity: float

class StrategySelector:
    """Automatic parallel strategy selector."""
    
    def __init__(
        self,
        num_gpus: int,
        memory_threshold: float = 0.8,
        communication_threshold: float = 0.5
    ):
        """Initialize strategy selector.
        
        Args:
            num_gpus: Number of available GPUs
            memory_threshold: Memory utilization threshold
            communication_threshold: Communication cost threshold
        """
        self.num_gpus = num_gpus
        self.memory_threshold = memory_threshold
        self.communication_threshold = communication_threshold
        
    def select_strategy(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        batch_size: int,
        input_shape: Union[tuple, List[tuple]]
    ) -> Tuple[ParallelStrategy, Dict[str, Any]]:
        """Select optimal parallel strategy.
        
        Args:
            model: Model to parallelize
            batch_size: Training batch size
            input_shape: Input shape(s)
            
        Returns:
            Selected strategy and configuration
        """
        profile = self._profile_model(model, batch_size, input_shape)
        
        # Decision tree for strategy selection
        if profile.memory_requirement > self._get_available_memory() * self.memory_threshold:
            if profile.compute_intensity > 100:  # High compute to memory ratio
                return ParallelStrategy.PIPELINE, {
                    "num_stages": self.num_gpus,
                    "chunk_size": batch_size // self.num_gpus
                }
            else:
                return ParallelStrategy.TENSOR, {
                    "split_size": profile.layer_sizes[0] // self.num_gpus
                }
        elif profile.communication_cost > self.communication_threshold:
            return ParallelStrategy.HYBRID, {
                "pipeline_stages": self.num_gpus // 2,
                "tensor_splits": 2,
                "chunk_size": batch_size // (self.num_gpus // 2)
            }
        else:
            return ParallelStrategy.PIPELINE, {
                "num_stages": self.num_gpus,
                "chunk_size": batch_size // self.num_gpus
            }
            
    def _profile_model(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        batch_size: int,
        input_shape: Union[tuple, List[tuple]]
    ) -> ModelProfile:
        """Profile model characteristics."""
        if isinstance(model, torch.nn.Module):
            return self._profile_torch_model(model, batch_size, input_shape)
        else:
            return self._profile_tf_model(model, batch_size, input_shape)
            
    def _profile_torch_model(
        self,
        model: torch.nn.Module,
        batch_size: int,
        input_shape: Union[tuple, List[tuple]]
    ) -> ModelProfile:
        """Profile PyTorch model."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Get layer sizes
        layer_sizes = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                layer_sizes.append(module.weight.size(0))
                
        # Estimate communication cost
        communication_cost = total_params * 4  # Assuming float32
        
        # Estimate memory requirement
        memory_requirement = (
            total_params * 4 +  # Parameters
            sum(layer_sizes) * batch_size * 4  # Activations
        )
        
        # Estimate compute intensity (FLOPs/byte)
        compute_intensity = self._estimate_flops(model, input_shape) / memory_requirement
        
        return ModelProfile(
            total_params=total_params,
            layer_sizes=layer_sizes,
            communication_cost=communication_cost,
            memory_requirement=memory_requirement,
            compute_intensity=compute_intensity
        )
        
    def _profile_tf_model(
        self,
        model: tf.keras.Model,
        batch_size: int,
        input_shape: Union[tuple, List[tuple]]
    ) -> ModelProfile:
        """Profile TensorFlow model."""
        # Count parameters
        total_params = model.count_params()
        
        # Get layer sizes
        layer_sizes = []
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                layer_sizes.append(layer.units if hasattr(layer, 'units') else layer.filters)
                
        # Estimate communication cost
        communication_cost = total_params * 4  # Assuming float32
        
        # Estimate memory requirement
        memory_requirement = (
            total_params * 4 +  # Parameters
            sum(layer_sizes) * batch_size * 4  # Activations
        )
        
        # Estimate compute intensity
        compute_intensity = self._estimate_tf_flops(model, input_shape) / memory_requirement
        
        return ModelProfile(
            total_params=total_params,
            layer_sizes=layer_sizes,
            communication_cost=communication_cost,
            memory_requirement=memory_requirement,
            compute_intensity=compute_intensity
        )
        
    def _get_available_memory(self) -> float:
        """Get available GPU memory."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        else:
            try:
                return tf.config.experimental.get_memory_info('GPU:0')['peak']
            except:
                return 8 * 1024 * 1024 * 1024  # Default 8GB 

    def _estimate_flops(
        self,
        model: torch.nn.Module,
        input_shape: Union[tuple, List[tuple]]
    ) -> int:
        """Estimate FLOPs for PyTorch model."""
        if isinstance(input_shape, tuple):
            inputs = torch.randn(1, *input_shape)
        else:
            inputs = [torch.randn(1, *shape) for shape in input_shape]
            
        from thop import profile
        flops, _ = profile(model, inputs=(inputs,))
        return flops
        
    def _estimate_tf_flops(
        self,
        model: tf.keras.Model,
        input_shape: Union[tuple, List[tuple]]
    ) -> int:
        """Estimate FLOPs for TensorFlow model."""
        flops = 0
        
        if isinstance(input_shape, tuple):
            input_shapes = [input_shape]
        else:
            input_shapes = input_shape
            
        # Create dummy inputs
        inputs = [tf.keras.Input(shape=shape[1:]) for shape in input_shapes]
        
        # Build model graph
        if len(inputs) == 1:
            outputs = model(inputs[0])
        else:
            outputs = model(inputs)
            
        # Count FLOPs for each layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                flops += 2 * layer.input_shape[-1] * layer.units
            elif isinstance(layer, tf.keras.layers.Conv2D):
                # kernel_h * kernel_w * in_channels * out_channels * out_h * out_w
                kernel_size = layer.kernel_size
                input_shape = layer.input_shape
                output_shape = layer.output_shape
                flops += (
                    2 * kernel_size[0] * kernel_size[1] *
                    input_shape[-1] * layer.filters *
                    output_shape[1] * output_shape[2]
                )
                
        return flops 