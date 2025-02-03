from typing import Any, Dict, List, Optional, Union, Tuple
import time
import torch
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .monitor import RuntimeStats
from .predictor import MLPredictor

@dataclass
class ProfileStats:
    """Model profiling statistics."""
    latency: float  # Average inference latency
    memory: float   # Peak memory usage
    flops: int      # Number of FLOPs
    parameters: int # Number of parameters
    sparsity: float # Model sparsity

class ModelProfiler:
    """Model performance profiler."""
    
    def __init__(self, warmup: int = 10, iterations: int = 100):
        """Initialize profiler.
        
        Args:
            warmup: Number of warmup iterations
            iterations: Number of profiling iterations
        """
        self.warmup = warmup
        self.iterations = iterations
        
    def profile(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        input_shape: Union[tuple, List[tuple]]
    ) -> ProfileStats:
        """Profile model performance.
        
        Args:
            model: Model to profile
            input_shape: Input shape(s)
            
        Returns:
            Profiling statistics
        """
        if isinstance(model, torch.nn.Module):
            return self._profile_torch(model, input_shape)
        else:
            return self._profile_tf(model, input_shape)
            
    def _profile_torch(
        self,
        model: torch.nn.Module,
        input_shape: Union[tuple, List[tuple]]
    ) -> ProfileStats:
        """Profile PyTorch model."""
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        if isinstance(input_shape, tuple):
            inputs = torch.randn(input_shape).to(device)
        else:
            inputs = [torch.randn(shape).to(device) for shape in input_shape]
            
        # Warmup
        for _ in range(self.warmup):
            with torch.no_grad():
                _ = model(inputs)
                
        # Measure latency
        start_time = time.time()
        for _ in range(self.iterations):
            with torch.no_grad():
                _ = model(inputs)
        latency = (time.time() - start_time) / self.iterations
        
        # Count parameters and compute sparsity
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum(
            (p == 0).sum().item() for p in model.parameters()
        )
        sparsity = zero_params / total_params
        
        # Get memory usage
        if torch.cuda.is_available():
            memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory = 0
            
        # Compute FLOPs
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(inputs,))
        
        return ProfileStats(
            latency=latency,
            memory=memory,
            flops=flops,
            parameters=total_params,
            sparsity=sparsity
        ) 

    def _profile_tf(
        self,
        model: tf.keras.Model,
        input_shape: Union[tuple, List[tuple]]
    ) -> ProfileStats:
        """Profile TensorFlow model."""
        # Create dummy input
        if isinstance(input_shape, tuple):
            inputs = tf.random.normal(input_shape)
        else:
            inputs = [tf.random.normal(shape) for shape in input_shape]
            
        # Warmup
        for _ in range(self.warmup):
            _ = model(inputs, training=False)
            
        # Measure latency
        start_time = time.time()
        for _ in range(self.iterations):
            _ = model(inputs, training=False)
        latency = (time.time() - start_time) / self.iterations
        
        # Count parameters and compute sparsity
        total_params = model.count_params()
        zero_params = sum(
            tf.reduce_sum(tf.cast(weights == 0, tf.int32))
            for weights in model.weights
        )
        sparsity = float(zero_params) / total_params
        
        # Get memory info
        memory = 0  # TODO: Implement TF memory profiling
        
        # Compute FLOPs
        flops = self._compute_tf_flops(model, inputs)
        
        return ProfileStats(
            latency=latency,
            memory=memory,
            flops=flops,
            parameters=total_params,
            sparsity=sparsity
        )
        
    def _compute_tf_flops(
        self,
        model: tf.keras.Model,
        inputs: Union[tf.Tensor, List[tf.Tensor]]
    ) -> int:
        """Compute TensorFlow model FLOPs."""
        concrete_func = tf.function(model).get_concrete_function(inputs)
        graph_def = concrete_func.graph.as_graph_def()
        
        # Process graph to count operations
        flops = 0
        for node in graph_def.node:
            if node.op in ['Conv2D', 'MatMul']:
                # Get input shapes
                input_shape = self._get_node_input_shape(node, concrete_func.graph)
                output_shape = self._get_node_output_shape(node, concrete_func.graph)
                
                if node.op == 'Conv2D':
                    # kernel_h * kernel_w * in_channels * out_channels * out_h * out_w
                    kernel_shape = self._get_kernel_shape(node, concrete_func.graph)
                    flops += np.prod(kernel_shape) * np.prod(output_shape[1:3])
                else:  # MatMul
                    # m * k * n for matrix multiplication
                    flops += np.prod(input_shape[1:]) * output_shape[-1]
                    
        return flops 

    def _get_node_input_shape(
        self,
        node: Any,
        graph: Any
    ) -> tuple:
        """Get input shape for graph node."""
        input_tensor = graph.get_tensor_by_name(node.input[0])
        if input_tensor.shape.dims is None:
            return (1,)  # Default shape if unknown
        return tuple(d.value for d in input_tensor.shape.dims)
        
    def _get_node_output_shape(
        self,
        node: Any,
        graph: Any
    ) -> tuple:
        """Get output shape for graph node."""
        output_tensor = graph.get_tensor_by_name(node.name + ":0")
        if output_tensor.shape.dims is None:
            return (1,)  # Default shape if unknown
        return tuple(d.value for d in output_tensor.shape.dims)
        
    def _get_kernel_shape(
        self,
        node: Any,
        graph: Any
    ) -> tuple:
        """Get kernel shape for convolution node."""
        kernel_tensor = graph.get_tensor_by_name(node.input[1])
        if kernel_tensor.shape.dims is None:
            return (1, 1, 1, 1)  # Default shape if unknown
        return tuple(d.value for d in kernel_tensor.shape.dims) 

    def profile_layer(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        layer_name: str,
        input_shape: Union[tuple, List[tuple]]
    ) -> Dict[str, float]:
        """Profile specific layer performance.
        
        Args:
            model: Model containing the layer
            layer_name: Name of layer to profile
            input_shape: Input shape(s)
            
        Returns:
            Layer statistics
        """
        if isinstance(model, torch.nn.Module):
            return self._profile_torch_layer(model, layer_name, input_shape)
        else:
            return self._profile_tf_layer(model, layer_name, input_shape)
            
    def _profile_torch_layer(
        self,
        model: torch.nn.Module,
        layer_name: str,
        input_shape: Union[tuple, List[tuple]]
    ) -> Dict[str, float]:
        """Profile PyTorch layer."""
        model.eval()
        device = next(model.parameters()).device
        
        # Get layer
        layer = dict(model.named_modules())[layer_name]
        
        # Create dummy input
        if isinstance(input_shape, tuple):
            inputs = torch.randn(input_shape).to(device)
        else:
            inputs = [torch.randn(shape).to(device) for shape in input_shape]
            
        # Measure latency
        start_time = time.time()
        for _ in range(self.iterations):
            with torch.no_grad():
                _ = layer(inputs)
        latency = (time.time() - start_time) / self.iterations
        
        # Count parameters
        parameters = sum(p.numel() for p in layer.parameters())
        
        # Compute FLOPs
        from thop import profile as thop_profile
        flops, _ = thop_profile(layer, inputs=(inputs,))
        
        return {
            'latency': latency,
            'parameters': parameters,
            'flops': flops
        }
        
    def _profile_tf_layer(
        self,
        model: tf.keras.Model,
        layer_name: str,
        input_shape: Union[tuple, List[tuple]]
    ) -> Dict[str, float]:
        """Profile TensorFlow layer."""
        # Get layer
        layer = model.get_layer(layer_name)
        
        # Create dummy input
        if isinstance(input_shape, tuple):
            inputs = tf.random.normal(input_shape)
        else:
            inputs = [tf.random.normal(shape) for shape in input_shape]
            
        # Measure latency
        start_time = time.time()
        for _ in range(self.iterations):
            _ = layer(inputs, training=False)
        latency = (time.time() - start_time) / self.iterations
        
        # Count parameters
        parameters = layer.count_params()
        
        # Compute FLOPs
        flops = self._compute_layer_flops(layer, inputs)
        
        return {
            'latency': latency,
            'parameters': parameters,
            'flops': flops
        }
        
    def _compute_layer_flops(
        self,
        layer: tf.keras.layers.Layer,
        inputs: Union[tf.Tensor, List[tf.Tensor]]
    ) -> int:
        """Compute FLOPs for a single layer."""
        if isinstance(layer, tf.keras.layers.Conv2D):
            input_shape = inputs.shape
            kernel = layer.kernel
            # kernel_h * kernel_w * in_channels * out_channels * out_h * out_w
            return int(np.prod(kernel.shape) * np.prod(input_shape[1:3]))
        elif isinstance(layer, tf.keras.layers.Dense):
            input_shape = inputs.shape
            # input_features * output_features
            return int(input_shape[-1] * layer.units)
        else:
            return 0  # Unknown layer type 

class ProfilerMetric(Enum):
    """Performance profiling metrics."""
    MEMORY = "memory"
    COMPUTE = "compute"
    COMMUNICATION = "communication"
    IO = "io"

@dataclass
class ProfileResult:
    """Profiling result."""
    metric: ProfilerMetric
    value: float
    bottleneck_score: float
    optimization_suggestions: List[str]

class PerformanceProfiler:
    """Analyzes training performance bottlenecks."""
    
    def __init__(
        self,
        predictor: MLPredictor,
        profiling_window: int = 100,
        bottleneck_threshold: float = 0.7
    ):
        """Initialize profiler.
        
        Args:
            predictor: Performance predictor
            profiling_window: Window size for analysis
            bottleneck_threshold: Bottleneck detection threshold
        """
        self.predictor = predictor
        self.profiling_window = profiling_window
        self.bottleneck_threshold = bottleneck_threshold
        self.history: List[RuntimeStats] = []
        
    def profile(self, stats: RuntimeStats) -> List[ProfileResult]:
        """Profile performance metrics.
        
        Args:
            stats: Current runtime statistics
            
        Returns:
            List of profiling results
        """
        self.history.append(stats)
        if len(self.history) > self.profiling_window:
            self.history.pop(0)
            
        results = []
        
        # Memory analysis
        memory_result = self._analyze_memory()
        if memory_result:
            results.append(memory_result)
            
        # Compute analysis
        compute_result = self._analyze_compute()
        if compute_result:
            results.append(compute_result)
            
        # Communication analysis
        communication_result = self._analyze_communication()
        if communication_result:
            results.append(communication_result)
            
        # I/O analysis
        io_result = self._analyze_io()
        if io_result:
            results.append(io_result)
            
        return results
        
    def _analyze_memory(self) -> Optional[ProfileResult]:
        """Analyze memory usage patterns."""
        if not self.history:
            return None
            
        # Calculate memory metrics
        memory_utils = [
            np.mean(stats.gpu_memory)
            for stats in self.history
        ]
        
        avg_util = np.mean(memory_utils)
        peak_util = np.max(memory_utils)
        variability = np.std(memory_utils)
        
        # Detect bottlenecks
        bottleneck_score = 0.0
        suggestions = []
        
        if peak_util > 0.9:  # 90% peak usage
            bottleneck_score += 0.5
            suggestions.append("Consider reducing batch size or model size")
            
        if variability > 0.2:  # High variability
            bottleneck_score += 0.3
            suggestions.append("Memory usage is unstable, check for memory leaks")
            
        if avg_util > 0.8:  # Sustained high usage
            bottleneck_score += 0.4
            suggestions.append("System is memory-bound, consider model optimization")
            
        if bottleneck_score >= self.bottleneck_threshold:
            return ProfileResult(
                metric=ProfilerMetric.MEMORY,
                value=avg_util,
                bottleneck_score=bottleneck_score,
                optimization_suggestions=suggestions
            )
            
        return None
        
    def _analyze_compute(self) -> Optional[ProfileResult]:
        """Analyze compute utilization patterns."""
        if not self.history:
            return None
            
        # Calculate compute metrics
        gpu_utils = [
            np.mean(stats.gpu_utilization)
            for stats in self.history
        ]
        
        avg_util = np.mean(gpu_utils)
        variability = np.std(gpu_utils)
        
        # Detect bottlenecks
        bottleneck_score = 0.0
        suggestions = []
        
        if avg_util < 0.5:  # Low GPU utilization
            bottleneck_score += 0.4
            suggestions.append("GPU underutilized, check data loading pipeline")
            
        if variability > 0.3:  # High variability
            bottleneck_score += 0.3
            suggestions.append("Unstable GPU utilization, check workload balance")
            
        if avg_util > 0.95:  # Very high utilization
            bottleneck_score += 0.5
            suggestions.append("System is compute-bound, consider model parallelism")
            
        if bottleneck_score >= self.bottleneck_threshold:
            return ProfileResult(
                metric=ProfilerMetric.COMPUTE,
                value=avg_util,
                bottleneck_score=bottleneck_score,
                optimization_suggestions=suggestions
            )
            
        return None
        
    def _analyze_communication(self) -> Optional[ProfileResult]:
        """Analyze communication patterns."""
        if not self.history:
            return None
            
        # Calculate communication metrics
        comm_overheads = [
            stats.communication_overhead
            for stats in self.history
        ]
        
        avg_overhead = np.mean(comm_overheads)
        peak_overhead = np.max(comm_overheads)
        
        # Detect bottlenecks
        bottleneck_score = 0.0
        suggestions = []
        
        if avg_overhead > 0.2:  # >20% time in communication
            bottleneck_score += 0.4
            suggestions.append("High communication overhead, consider gradient compression")
            
        if peak_overhead > 0.5:  # >50% peak communication time
            bottleneck_score += 0.4
            suggestions.append("Communication spikes detected, check network configuration")
            
        if bottleneck_score >= self.bottleneck_threshold:
            return ProfileResult(
                metric=ProfilerMetric.COMMUNICATION,
                value=avg_overhead,
                bottleneck_score=bottleneck_score,
                optimization_suggestions=suggestions
            )
            
        return None
        
    def _analyze_io(self) -> Optional[ProfileResult]:
        """Analyze I/O patterns."""
        if not self.history:
            return None
            
        # Calculate I/O metrics
        data_loading_times = [
            stats.data_loading_time
            for stats in self.history
        ]
        
        avg_time = np.mean(data_loading_times)
        max_time = np.max(data_loading_times)
        
        # Detect bottlenecks
        bottleneck_score = 0.0
        suggestions = []
        
        if avg_time > 0.1:  # >10% time in data loading
            bottleneck_score += 0.4
            suggestions.append("Slow data loading, consider using data prefetching")
            
        if max_time > 0.3:  # >30% peak loading time
            bottleneck_score += 0.3
            suggestions.append("Data loading spikes detected, check disk I/O")
            
        if bottleneck_score >= self.bottleneck_threshold:
            return ProfileResult(
                metric=ProfilerMetric.IO,
                value=avg_time,
                bottleneck_score=bottleneck_score,
                optimization_suggestions=suggestions
            )
            
        return None 