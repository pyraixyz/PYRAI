from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .profiler import ProfileStats

class OptimizationVisualizer:
    """Visualization tools for optimization results."""
    
    def plot_profile_comparison(
        self,
        profiles: Dict[str, ProfileStats],
        metrics: Optional[List[str]] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """Plot model profile comparison.
        
        Args:
            profiles: Dictionary of model profiles
            metrics: Metrics to plot
            figsize: Figure size
        """
        if metrics is None:
            metrics = ['latency', 'memory', 'flops', 'parameters', 'sparsity']
            
        # Prepare data
        data = {
            metric: [
                getattr(profile, metric) for profile in profiles.values()
            ]
            for metric in metrics
        }
        
        # Create plot
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=figsize,
            squeeze=False
        )
        
        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            sns.barplot(
                x=list(profiles.keys()),
                y=data[metric],
                ax=ax
            )
            ax.set_title(f"{metric.capitalize()} Comparison")
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def plot_search_history(
        self,
        study: Any,
        figsize: tuple = (12, 4)
    ) -> None:
        """Plot optimization search history.
        
        Args:
            study: Optuna study object
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot optimization history
        ax1.set_title("Optimization History")
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Objective Value")
        
        trials = study.trials_dataframe()
        ax1.plot(trials.number, trials.value, 'o-')
        
        # Plot parameter importances
        importances = optuna.importance.get_param_importances(study)
        ax2.set_title("Parameter Importance")
        
        params = list(importances.keys())
        values = list(importances.values())
        
        sns.barplot(x=values, y=params, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
    def plot_memory_timeline(
        self,
        memory_data: Dict[str, List[float]],
        timestamps: List[float],
        figsize: tuple = (12, 6)
    ) -> None:
        """Plot memory usage timeline.
        
        Args:
            memory_data: Dictionary of memory usage data
            timestamps: List of timestamps
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, usage in memory_data.items():
            plt.plot(timestamps, usage, label=name)
            
        plt.title("Memory Usage Timeline")
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Usage (MB)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_layer_profile(
        self,
        layer_stats: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 8)
    ) -> None:
        """Plot per-layer performance profile.
        
        Args:
            layer_stats: Dictionary of layer statistics
            metrics: Metrics to plot
            figsize: Figure size
        """
        if metrics is None:
            metrics = ['flops', 'parameters', 'latency']
            
        num_layers = len(layer_stats)
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(
            num_metrics,
            1,
            figsize=figsize,
            sharex=True
        )
        
        layer_names = list(layer_stats.keys())
        
        for i, metric in enumerate(metrics):
            values = [stats[metric] for stats in layer_stats.values()]
            
            axes[i].bar(layer_names, values)
            axes[i].set_title(f"Per-Layer {metric.capitalize()}")
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def plot_optimization_progress(
        self,
        metrics_history: Dict[str, List[float]],
        figsize: tuple = (12, 6)
    ) -> None:
        """Plot optimization progress over iterations.
        
        Args:
            metrics_history: Dictionary of metric histories
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for metric, values in metrics_history.items():
            plt.plot(values, label=metric)
            
        plt.title("Optimization Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.show() 