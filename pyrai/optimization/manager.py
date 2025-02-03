from typing import Any, Dict, Optional, Union
import torch
import tensorflow as tf
from .optimizer import TorchOptimizer, TensorFlowOptimizer
from .compression import ModelCompressor

class OptimizationManager:
    """Manages model optimization and compression."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize manager.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.torch_optimizer = TorchOptimizer()
        self.tf_optimizer = TensorFlowOptimizer()
        self.compressor = ModelCompressor(config.get("compression", {}))
        
    def optimize_model(
        self,
        model: Union[torch.nn.Module, tf.keras.Model]
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Optimize model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        if isinstance(model, torch.nn.Module):
            return self.torch_optimizer.optimize(model, self.config)
        elif isinstance(model, tf.keras.Model):
            return self.tf_optimizer.optimize(model, self.config)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
            
    def compress_model(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        method: str = "svd"
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Compress model.
        
        Args:
            model: Model to compress
            method: Compression method
            
        Returns:
            Compressed model
        """
        return self.compressor.compress_model(model, method) 