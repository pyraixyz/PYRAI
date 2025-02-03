from typing import Dict, Any, Optional, List
import time
import psutil
import numpy as np
from pathlib import Path
import torch
import tensorflow as tf
from ..core.base import ModelBackend

class ModelProfiler:
    """Profiles model performance."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def profile_inference(self,
                         model: Any,
                         backend: ModelBackend,
                         sample_input: Any,
                         batch_sizes: List[int] = [1, 8, 16, 32],
                         num_iterations: int = 100) -> Dict[str, Any]:
        """Profile model inference performance."""
        results = {
            'latency': {},
            'throughput': {},
            'memory': {},
            'gpu_utilization': {}
        }
        
        for batch_size in batch_sizes:
            # Prepare batch input
            batch_input = self._prepare_batch(sample_input, batch_size)
            
            # Warmup
            for _ in range(10):
                backend.predict(model, batch_input)
            
            # Measure latency
            latencies = []
            memory_usage = []
            gpu_util = []
            
            for _ in range(num_iterations):
                start_time = time.time()
                backend.predict(model, batch_input)
                latency = time.time() - start_time
                latencies.append(latency)
                
                # Record memory usage
                memory_usage.append(psutil.Process().memory_info().rss)
                
                # Record GPU utilization if available
                if torch.cuda.is_available():
                    gpu_util.append(torch.cuda.utilization())
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            throughput = batch_size / avg_latency
            avg_memory = np.mean(memory_usage)
            avg_gpu_util = np.mean(gpu_util) if gpu_util else None
            
            results['latency'][batch_size] = {
                'mean': avg_latency,
                'p95': p95_latency
            }
            results['throughput'][batch_size] = throughput
            results['memory'][batch_size] = avg_memory
            if avg_gpu_util is not None:
                results['gpu_utilization'][batch_size] = avg_gpu_util
        
        return results
    
    def profile_memory(self,
                      model: Any,
                      backend: ModelBackend,
                      sample_input: Any) -> Dict[str, Any]:
        """Profile model memory usage."""
        import torch.profiler as torch_profiler
        import tensorflow as tf
        
        results = {}
        
        # PyTorch profiling
        if isinstance(model, torch.nn.Module):
            with torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                backend.predict(model, sample_input)
            
            results['pytorch'] = {
                'memory_by_operator': prof.key_averages().table(
                    sort_by="self_cuda_memory_usage", row_limit=10
                ),
                'trace': prof.key_averages()
            }
        
        # TensorFlow profiling
        elif isinstance(model, tf.keras.Model):
            tf.profiler.experimental.start(str(self.output_dir / 'tf_profile'))
            backend.predict(model, sample_input)
            tf.profiler.experimental.stop()
            
            results['tensorflow'] = {
                'profile_dir': str(self.output_dir / 'tf_profile')
            }
        
        return results
    
    def _prepare_batch(self, sample_input: Any, batch_size: int) -> Any:
        """Prepare input batch of given size."""
        if isinstance(sample_input, np.ndarray):
            return np.repeat(sample_input[np.newaxis], batch_size, axis=0)
        elif isinstance(sample_input, torch.Tensor):
            return sample_input.unsqueeze(0).repeat(batch_size, *[1]*len(sample_input.shape))
        elif isinstance(sample_input, tf.Tensor):
            return tf.repeat(tf.expand_dims(sample_input, 0), batch_size, axis=0)
        else:
            raise ValueError(f"Unsupported input type: {type(sample_input)}") 