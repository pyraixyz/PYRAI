from typing import Any, Dict, Optional, Union
import torch
import tensorflow as tf
import numpy as np

class ModelCompressor:
    """Model compression utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize compressor.
        
        Args:
            config: Compression configuration
        """
        self.config = config
        
    def compress_weights(
        self,
        weights: Union[torch.Tensor, tf.Tensor, np.ndarray],
        method: str = "svd"
    ) -> Union[torch.Tensor, tf.Tensor, np.ndarray]:
        """Compress model weights.
        
        Args:
            weights: Weight tensor/array to compress
            method: Compression method
            
        Returns:
            Compressed weights
        """
        if method == "svd":
            return self._compress_svd(weights)
        elif method == "pruning":
            return self._compress_pruning(weights)
        else:
            raise ValueError(f"Unknown compression method: {method}")
            
    def _compress_svd(
        self,
        weights: Union[torch.Tensor, tf.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, tf.Tensor, np.ndarray]:
        """Compress using SVD."""
        # Convert to numpy
        if isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().numpy()
        elif isinstance(weights, tf.Tensor):
            weights_np = weights.numpy()
        else:
            weights_np = weights
            
        # Perform SVD
        U, S, V = np.linalg.svd(weights_np, full_matrices=False)
        
        # Keep top-k singular values
        k = self.config.get("svd_rank", min(weights_np.shape) // 2)
        compressed = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
        
        # Convert back to original type
        if isinstance(weights, torch.Tensor):
            return torch.from_numpy(compressed).to(weights.device)
        elif isinstance(weights, tf.Tensor):
            return tf.convert_to_tensor(compressed)
        else:
            return compressed
            
    def _compress_pruning(
        self,
        weights: Union[torch.Tensor, tf.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, tf.Tensor, np.ndarray]:
        """Compress using weight pruning."""
        # Convert to numpy
        if isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().numpy()
        elif isinstance(weights, tf.Tensor):
            weights_np = weights.numpy()
        else:
            weights_np = weights
            
        # Get pruning threshold
        sparsity = self.config.get("pruning_sparsity", 0.5)
        threshold = np.percentile(np.abs(weights_np), sparsity * 100)
        
        # Apply pruning mask
        mask = np.abs(weights_np) > threshold
        pruned = weights_np * mask
        
        # Convert back to original type
        if isinstance(weights, torch.Tensor):
            return torch.from_numpy(pruned).to(weights.device)
        elif isinstance(weights, tf.Tensor):
            return tf.convert_to_tensor(pruned)
        else:
            return pruned
            
    def compress_model(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        method: str = "svd"
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Compress entire model.
        
        Args:
            model: Model to compress
            method: Compression method
            
        Returns:
            Compressed model
        """
        if isinstance(model, torch.nn.Module):
            return self._compress_torch_model(model, method)
        elif isinstance(model, tf.keras.Model):
            return self._compress_tf_model(model, method)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
            
    def _compress_torch_model(
        self,
        model: torch.nn.Module,
        method: str
    ) -> torch.nn.Module:
        """Compress PyTorch model."""
        for name, param in model.named_parameters():
            if "weight" in name:  # Only compress weight matrices
                compressed = self.compress_weights(param.data, method)
                param.data = compressed
        return model
        
    def _compress_tf_model(
        self,
        model: tf.keras.Model,
        method: str
    ) -> tf.keras.Model:
        """Compress TensorFlow model."""
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:  # Layer has weights
                compressed = [
                    self.compress_weights(w, method) 
                    if len(w.shape) > 1  # Only compress weight matrices
                    else w
                    for w in weights
                ]
                layer.set_weights(compressed)
        return model 