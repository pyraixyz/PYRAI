from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
from .profiler import ModelProfiler, ProfileStats

@dataclass
class OptimizationAdvice:
    """Optimization advice for model improvement."""
    target: str  # Target metric to improve
    suggestion: str  # Suggested optimization
    expected_improvement: float  # Expected improvement percentage
    difficulty: str  # Implementation difficulty
    priority: int  # Priority level (1-5)

class OptimizationAdvisor:
    """Provides optimization suggestions based on model profiling."""
    
    def __init__(self, profiler: ModelProfiler):
        """Initialize advisor.
        
        Args:
            profiler: Model profiler instance
        """
        self.profiler = profiler
        
    def analyze(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        input_shape: Union[tuple, List[tuple]]
    ) -> List[OptimizationAdvice]:
        """Analyze model and generate optimization advice.
        
        Args:
            model: Model to analyze
            input_shape: Input shape(s)
            
        Returns:
            List of optimization suggestions
        """
        # Profile model
        profile = self.profiler.profile(model, input_shape)
        
        # Generate advice
        advice = []
        
        # Check latency
        if profile.latency > 0.1:  # More than 100ms
            advice.append(OptimizationAdvice(
                target="latency",
                suggestion="Consider quantization to int8",
                expected_improvement=50.0,
                difficulty="medium",
                priority=4
            ))
            
        # Check memory usage
        if profile.memory > 1000:  # More than 1GB
            advice.append(OptimizationAdvice(
                target="memory",
                suggestion="Apply model pruning to reduce model size",
                expected_improvement=30.0,
                difficulty="hard",
                priority=3
            ))
            
        # Check model size
        if profile.parameters > 10e6:  # More than 10M parameters
            advice.append(OptimizationAdvice(
                target="model_size",
                suggestion="Use knowledge distillation to create smaller model",
                expected_improvement=40.0,
                difficulty="hard",
                priority=5
            ))
            
        # Check sparsity
        if profile.sparsity < 0.5:  # Less than 50% sparsity
            advice.append(OptimizationAdvice(
                target="efficiency",
                suggestion="Increase model sparsity through structured pruning",
                expected_improvement=20.0,
                difficulty="medium",
                priority=2
            ))
            
        # Layer-specific analysis
        layer_profiles = self._analyze_layers(model, input_shape)
        bottleneck_layers = self._identify_bottlenecks(layer_profiles)
        
        for layer_name in bottleneck_layers:
            advice.append(OptimizationAdvice(
                target="layer_efficiency",
                suggestion=f"Optimize layer {layer_name} through operator fusion",
                expected_improvement=15.0,
                difficulty="medium",
                priority=3
            ))
            
        return sorted(advice, key=lambda x: x.priority, reverse=True)
        
    def _analyze_layers(
        self,
        model: Any,
        input_shape: Union[tuple, List[tuple]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze individual layer performance."""
        layer_profiles = {}
        
        if isinstance(model, torch.nn.Module):
            for name, _ in model.named_modules():
                if name:  # Skip empty name (root module)
                    profile = self.profiler.profile_layer(model, name, input_shape)
                    layer_profiles[name] = profile
        else:  # TensorFlow
            for layer in model.layers:
                profile = self.profiler.profile_layer(model, layer.name, input_shape)
                layer_profiles[layer.name] = profile
                
        return layer_profiles
        
    def _identify_bottlenecks(
        self,
        layer_profiles: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Identify performance bottleneck layers."""
        bottlenecks = []
        
        # Compute total latency
        total_latency = sum(p['latency'] for p in layer_profiles.values())
        
        # Find layers taking more than 10% of total time
        for name, profile in layer_profiles.items():
            if profile['latency'] / total_latency > 0.1:
                bottlenecks.append(name)
                
        return bottlenecks 