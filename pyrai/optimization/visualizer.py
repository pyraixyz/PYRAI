from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from .profiler import ProfileResult, ProfilerMetric
from .monitor import RuntimeStats

class PerformanceVisualizer:
    """Visualizes training performance metrics."""
    
    def __init__(
        self,
        history_window: int = 100,
        update_interval: int = 10
    ):
        """Initialize visualizer.
        
        Args:
            history_window: Window size for visualization
            update_interval: Plot update interval
        """
        self.history_window = history_window
        self.update_interval = update_interval
        self.history: List[RuntimeStats] = []
        self.fig = None
        self.axes = None
        
    def update(self, stats: RuntimeStats):
        """Update visualization with new statistics."""
        self.history.append(stats)
        if len(self.history) > self.history_window:
            self.history.pop(0)
            
        if len(self.history) % self.update_interval == 0:
            self._plot_metrics()
            
    def _plot_metrics(self):
        """Plot performance metrics."""
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            plt.ion()
            
        self._plot_memory_usage(self.axes[0, 0])
        self._plot_compute_usage(self.axes[0, 1])
        self._plot_communication(self.axes[1, 0])
        self._plot_throughput(self.axes[1, 1])
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def _plot_memory_usage(self, ax):
        """Plot memory usage over time."""
        ax.clear()
        memory_utils = [
            np.mean(stats.gpu_memory)
            for stats in self.history
        ]
        ax.plot(memory_utils, label='GPU Memory')
        ax.axhline(y=0.9, color='r', linestyle='--', label='Warning Threshold')
        ax.set_title('GPU Memory Usage')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Memory Utilization')
        ax.legend()
        ax.grid(True)
        
    def _plot_compute_usage(self, ax):
        """Plot compute utilization over time."""
        ax.clear()
        gpu_utils = [
            np.mean(stats.gpu_utilization)
            for stats in self.history
        ]
        ax.plot(gpu_utils, label='GPU Utilization')
        ax.set_title('GPU Compute Utilization')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Compute Utilization')
        ax.legend()
        ax.grid(True)
        
    def _plot_communication(self, ax):
        """Plot communication overhead over time."""
        ax.clear()
        comm_overheads = [
            stats.communication_overhead
            for stats in self.history
        ]
        ax.plot(comm_overheads, label='Communication')
        ax.set_title('Communication Overhead')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Overhead (s)')
        ax.legend()
        ax.grid(True)
        
    def _plot_throughput(self, ax):
        """Plot training throughput over time."""
        ax.clear()
        throughputs = [
            stats.throughput
            for stats in self.history
        ]
        ax.plot(throughputs, label='Throughput')
        ax.set_title('Training Throughput')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Samples/second')
        ax.legend()
        ax.grid(True)
        
    def plot_bottlenecks(self, profile_results: List[ProfileResult]):
        """Plot performance bottlenecks."""
        plt.figure(figsize=(10, 6))
        
        metrics = [result.metric.value for result in profile_results]
        scores = [result.bottleneck_score for result in profile_results]
        
        sns.barplot(x=metrics, y=scores)
        plt.title('Performance Bottlenecks')
        plt.xlabel('Metric')
        plt.ylabel('Bottleneck Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def plot_layer_profile(self, layer_stats: Dict[str, Dict[str, float]]):
        """Plot layer-wise performance profile."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        layers = list(layer_stats.keys())
        latencies = [stats['latency'] for stats in layer_stats.values()]
        flops = [stats['flops'] for stats in layer_stats.values()]
        params = [stats['parameters'] for stats in layer_stats.values()]
        
        # Latency plot
        sns.barplot(x=layers, y=latencies, ax=ax1)
        ax1.set_title('Layer Latencies')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # FLOPs plot
        sns.barplot(x=layers, y=flops, ax=ax2)
        ax2.set_title('Layer FLOPs')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Parameters plot
        sns.barplot(x=layers, y=params, ax=ax3)
        ax3.set_title('Layer Parameters')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show() 