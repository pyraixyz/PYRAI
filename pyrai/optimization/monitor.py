from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import tensorflow as tf
import numpy as np
import time
from dataclasses import dataclass
from threading import Thread
from queue import Queue
import psutil
import GPUtil
from .predictor import PerformancePrediction
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .alert import AlertManager

@dataclass
class RuntimeStats:
    """Runtime performance statistics."""
    timestamp: float
    gpu_utilization: List[float]
    gpu_memory: List[float]
    cpu_utilization: float
    system_memory: float
    throughput: float
    latency: float

class PerformanceMonitor:
    """Online performance monitoring."""
    
    def __init__(
        self,
        sampling_interval: float = 1.0,
        history_size: int = 1000
    ):
        """Initialize monitor.
        
        Args:
            sampling_interval: Sampling interval in seconds
            history_size: Maximum history size
        """
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        self.history: List[RuntimeStats] = []
        self._stop_flag = False
        self._metrics_queue = Queue()
        
    def start(self):
        """Start monitoring."""
        self._stop_flag = False
        self._monitor_thread = Thread(target=self._monitoring_loop)
        self._monitor_thread.start()
        
    def stop(self):
        """Stop monitoring."""
        self._stop_flag = True
        self._monitor_thread.join()
        
    def add_metric(self, name: str, value: float):
        """Add custom metric."""
        self._metrics_queue.put((name, value))
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        if not self.history:
            return {}
            
        latest = self.history[-1]
        return {
            "gpu_utilization": np.mean(latest.gpu_utilization),
            "gpu_memory": np.mean(latest.gpu_memory),
            "cpu_utilization": latest.cpu_utilization,
            "system_memory": latest.system_memory,
            "throughput": latest.throughput,
            "latency": latest.latency
        }
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_flag:
            try:
                # Collect GPU stats
                gpus = GPUtil.getGPUs()
                gpu_utils = [gpu.load * 100 for gpu in gpus]
                gpu_mems = [gpu.memoryUtil * 100 for gpu in gpus]
                
                # Collect CPU and system stats
                cpu_util = psutil.cpu_percent()
                sys_mem = psutil.virtual_memory().percent
                
                # Process custom metrics
                throughput = 0.0
                latency = 0.0
                while not self._metrics_queue.empty():
                    name, value = self._metrics_queue.get()
                    if name == "throughput":
                        throughput = value
                    elif name == "latency":
                        latency = value
                        
                # Create stats
                stats = RuntimeStats(
                    timestamp=time.time(),
                    gpu_utilization=gpu_utils,
                    gpu_memory=gpu_mems,
                    cpu_utilization=cpu_util,
                    system_memory=sys_mem,
                    throughput=throughput,
                    latency=latency
                )
                
                # Update history
                self.history.append(stats)
                if len(self.history) > self.history_size:
                    self.history.pop(0)
                    
            except Exception as e:
                print(f"Monitoring error: {e}")
                
            time.sleep(self.sampling_interval)

class DistributedMonitor(PerformanceMonitor):
    """Distributed performance monitoring."""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        sampling_interval: float = 1.0,
        history_size: int = 1000,
        sync_interval: float = 5.0
    ):
        """Initialize distributed monitor.
        
        Args:
            world_size: Number of processes
            rank: Process rank
            sampling_interval: Sampling interval in seconds
            history_size: Maximum history size
            sync_interval: Statistics sync interval
        """
        super().__init__(sampling_interval, history_size)
        self.world_size = world_size
        self.rank = rank
        self.sync_interval = sync_interval
        self.global_stats: Dict[int, RuntimeStats] = {}
        self._last_sync = 0
        
    def start(self):
        """Start monitoring with synchronization."""
        super().start()
        if self.rank == 0:
            self._sync_thread = Thread(target=self._sync_loop)
            self._sync_thread.start()
            
    def stop(self):
        """Stop monitoring."""
        super().stop()
        if self.rank == 0:
            self._sync_thread.join()
            
    def get_global_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get synchronized statistics from all processes."""
        return {
            rank: self._stats_to_dict(stats)
            for rank, stats in self.global_stats.items()
        }
        
    def _sync_loop(self):
        """Synchronization loop for distributed monitoring."""
        import torch.distributed as dist
        
        while not self._stop_flag:
            current_time = time.time()
            if current_time - self._last_sync >= self.sync_interval:
                # Gather stats from all processes
                local_stats = self.get_stats()
                gathered_stats = [None] * self.world_size
                dist.all_gather_object(gathered_stats, local_stats)
                
                # Update global stats
                for rank, stats in enumerate(gathered_stats):
                    if stats:
                        self.global_stats[rank] = RuntimeStats(
                            timestamp=current_time,
                            **stats
                        )
                        
                self._last_sync = current_time
                
            time.sleep(0.1)
            
    def _stats_to_dict(self, stats: RuntimeStats) -> Dict[str, Any]:
        """Convert RuntimeStats to dictionary."""
        return {
            "timestamp": stats.timestamp,
            "gpu_utilization": stats.gpu_utilization,
            "gpu_memory": stats.gpu_memory,
            "cpu_utilization": stats.cpu_utilization,
            "system_memory": stats.system_memory,
            "throughput": stats.throughput,
            "latency": stats.latency
        }

class EnhancedPerformanceMonitor(PerformanceMonitor):
    """Enhanced performance monitor with alerts and advanced visualization."""
    
    def __init__(
        self,
        sampling_interval: float = 1.0,
        history_size: int = 1000,
        alert_manager: Optional['AlertManager'] = None
    ):
        """Initialize enhanced monitor.
        
        Args:
            sampling_interval: Sampling interval in seconds
            history_size: Maximum history size
            alert_manager: Optional alert manager
        """
        super().__init__(sampling_interval, history_size)
        self.alert_manager = alert_manager or AlertManager.create_default()
        self.peak_throughput = 0.0
        self.metrics_history: Dict[str, List[float]] = {
            "compute_efficiency": [],
            "memory_bandwidth": [],
            "io_wait": [],
            "network_throughput": []
        }
        
    def _monitoring_loop(self):
        """Enhanced monitoring loop."""
        while not self._stop_flag:
            try:
                # Collect basic stats
                stats = super()._monitoring_loop()
                
                # Collect additional metrics
                additional_metrics = self._collect_additional_metrics()
                stats.update(additional_metrics)
                
                # Update peak throughput
                if stats["throughput"] > self.peak_throughput:
                    self.peak_throughput = stats["throughput"]
                stats["peak_throughput"] = self.peak_throughput
                
                # Check alerts
                self.alert_manager.check_alerts(stats)
                
                # Update history
                self._update_history(stats)
                
            except Exception as e:
                print(f"Enhanced monitoring error: {e}")
                
            time.sleep(self.sampling_interval)
            
    def _collect_additional_metrics(self) -> Dict[str, float]:
        """Collect additional performance metrics."""
        metrics = {}
        
        # Compute efficiency (FLOPS utilization)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            metrics["compute_efficiency"] = (
                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            )
        except:
            metrics["compute_efficiency"] = 0.0
            
        # Memory bandwidth
        try:
            metrics["memory_bandwidth"] = (
                pynvml.nvmlDeviceGetMemoryInfo(handle).used /
                pynvml.nvmlDeviceGetMemoryInfo(handle).total * 100
            )
        except:
            metrics["memory_bandwidth"] = 0.0
            
        # IO wait
        try:
            metrics["io_wait"] = psutil.cpu_times_percent().iowait
        except:
            metrics["io_wait"] = 0.0
            
        # Network throughput
        try:
            net_io = psutil.net_io_counters()
            metrics["network_throughput"] = (
                net_io.bytes_sent + net_io.bytes_recv
            ) / 1024 / 1024  # MB/s
        except:
            metrics["network_throughput"] = 0.0
            
        return metrics
        
    def _update_history(self, stats: Dict[str, Any]):
        """Update metrics history."""
        for metric in self.metrics_history:
            if metric in stats:
                self.metrics_history[metric].append(stats[metric])
                if len(self.metrics_history[metric]) > self.history_size:
                    self.metrics_history[metric].pop(0)
                    
    def get_advanced_plots(self) -> Dict[str, plt.Figure]:
        """Generate advanced visualization plots."""
        figs = {}
        
        # Performance correlation matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        metrics_df = pd.DataFrame(self.metrics_history)
        sns.heatmap(
            metrics_df.corr(),
            annot=True,
            cmap='RdYlBu',
            center=0,
            ax=ax
        )
        ax.set_title("Performance Metric Correlations")
        figs["correlations"] = fig
        
        # System bottleneck analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.plot(ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Utilization (%)")
        ax.set_title("System Resource Utilization")
        ax.legend()
        figs["bottlenecks"] = fig
        
        return figs 