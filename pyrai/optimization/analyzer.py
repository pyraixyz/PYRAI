from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from .predictor import PerformancePrediction
from .parallel import ParallelStrategy

@dataclass
class PerformanceAnalysis:
    """Performance analysis results."""
    bottlenecks: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    visualizations: Dict[str, Any]

class PerformanceAnalyzer:
    """Analyzes parallel training performance."""
    
    def __init__(self, threshold: float = 0.1):
        """Initialize analyzer.
        
        Args:
            threshold: Bottleneck detection threshold
        """
        self.threshold = threshold
        
    def analyze(
        self,
        prediction: PerformancePrediction,
        strategy: ParallelStrategy,
        config: Dict[str, Any]
    ) -> PerformanceAnalysis:
        """Analyze predicted performance.
        
        Args:
            prediction: Performance prediction
            strategy: Parallel strategy
            config: Strategy configuration
            
        Returns:
            Analysis results
        """
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(prediction)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            bottlenecks,
            strategy,
            config
        )
        
        # Compute metrics
        metrics = self._compute_metrics(prediction)
        
        # Create visualizations
        visualizations = self._create_visualizations(prediction)
        
        return PerformanceAnalysis(
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            metrics=metrics,
            visualizations=visualizations
        )
        
    def _identify_bottlenecks(
        self,
        prediction: PerformancePrediction
    ) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check computation vs communication
        compute_time = prediction.latency - prediction.communication_overhead
        if prediction.communication_overhead / prediction.latency > self.threshold:
            bottlenecks.append("communication")
            
        # Check memory usage
        if prediction.memory_usage / (32 * 1024**3) > 0.9:  # >90% of 32GB
            bottlenecks.append("memory")
            
        # Check device utilization
        if any(util < 0.5 for util in prediction.device_utilization):
            bottlenecks.append("device_utilization")
            
        return bottlenecks
        
    def _generate_recommendations(
        self,
        bottlenecks: List[str],
        strategy: ParallelStrategy,
        config: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck == "communication":
                if strategy == ParallelStrategy.PIPELINE:
                    recommendations.append(
                        "Increase pipeline chunk size to reduce communication frequency"
                    )
                elif strategy == ParallelStrategy.TENSOR:
                    recommendations.append(
                        "Reduce tensor parallel degree to decrease all-reduce overhead"
                    )
                    
            elif bottleneck == "memory":
                recommendations.append(
                    "Consider using gradient checkpointing to reduce memory usage"
                )
                recommendations.append(
                    "Try increasing pipeline stages to distribute memory load"
                )
                
            elif bottleneck == "device_utilization":
                recommendations.append(
                    "Adjust load balancing between devices"
                )
                recommendations.append(
                    "Consider hybrid parallelism for better resource utilization"
                )
                
        return recommendations
        
    def _compute_metrics(
        self,
        prediction: PerformancePrediction
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        return {
            "efficiency": np.mean(prediction.device_utilization),
            "communication_ratio": prediction.communication_overhead / prediction.latency,
            "memory_efficiency": prediction.memory_usage / (32 * 1024**3),  # Assuming 32GB
            "throughput_per_gpu": prediction.throughput / len(prediction.device_utilization)
        }
        
    def _create_visualizations(
        self,
        prediction: PerformancePrediction
    ) -> Dict[str, Any]:
        """Create performance visualizations."""
        figs = {}
        
        # Device utilization heatmap
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            np.array(prediction.device_utilization).reshape(1, -1),
            annot=True,
            cmap='YlOrRd',
            ax=ax
        )
        ax.set_title("Device Utilization")
        figs["utilization"] = fig
        
        # Latency breakdown
        fig, ax = plt.subplots(figsize=(8, 6))
        components = {
            "Computation": prediction.latency - prediction.communication_overhead,
            "Communication": prediction.communication_overhead
        }
        plt.pie(
            components.values(),
            labels=components.keys(),
            autopct='%1.1f%%'
        )
        ax.set_title("Latency Breakdown")
        figs["latency"] = fig
        
        return figs 