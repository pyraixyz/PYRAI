from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .monitor import RuntimeStats
from .predictor import MLPredictor

class SchedulingPolicy(Enum):
    """Scheduling policies."""
    ROUND_ROBIN = "round_robin"
    LOAD_AWARE = "load_aware"
    PERFORMANCE_AWARE = "performance_aware"
    HYBRID = "hybrid"

@dataclass
class SchedulingDecision:
    """Scheduling decision."""
    device_id: int
    batch_size: int
    priority: float

class AdaptiveScheduler:
    """Adaptive workload scheduler."""
    
    def __init__(
        self,
        num_devices: int,
        policy: SchedulingPolicy = SchedulingPolicy.HYBRID,
        history_window: int = 100
    ):
        """Initialize scheduler.
        
        Args:
            num_devices: Number of devices
            policy: Scheduling policy
            history_window: History window size
        """
        self.num_devices = num_devices
        self.policy = policy
        self.history_window = history_window
        self.predictor = MLPredictor("throughput")
        self.device_stats: Dict[int, List[RuntimeStats]] = {
            i: [] for i in range(num_devices)
        }
        
    def schedule(
        self,
        batch: Any,
        stats: Dict[int, RuntimeStats]
    ) -> List[SchedulingDecision]:
        """Make scheduling decisions.
        
        Args:
            batch: Input batch
            stats: Current device statistics
            
        Returns:
            List of scheduling decisions
        """
        # Update device statistics
        for device_id, device_stats in stats.items():
            self.device_stats[device_id].append(device_stats)
            if len(self.device_stats[device_id]) > self.history_window:
                self.device_stats[device_id].pop(0)
                
        if self.policy == SchedulingPolicy.ROUND_ROBIN:
            return self._round_robin_schedule(batch)
        elif self.policy == SchedulingPolicy.LOAD_AWARE:
            return self._load_aware_schedule(batch)
        elif self.policy == SchedulingPolicy.PERFORMANCE_AWARE:
            return self._performance_aware_schedule(batch)
        else:  # Hybrid
            return self._hybrid_schedule(batch)
            
    def _round_robin_schedule(self, batch: Any) -> List[SchedulingDecision]:
        """Simple round-robin scheduling."""
        batch_size = self._get_batch_size(batch)
        sub_batch_size = batch_size // self.num_devices
        
        decisions = []
        for i in range(self.num_devices):
            decisions.append(SchedulingDecision(
                device_id=i,
                batch_size=sub_batch_size,
                priority=1.0
            ))
            
        return decisions
        
    def _load_aware_schedule(self, batch: Any) -> List[SchedulingDecision]:
        """Load-aware scheduling."""
        batch_size = self._get_batch_size(batch)
        device_loads = self._get_device_loads()
        
        # Normalize loads
        total_load = sum(device_loads.values())
        if total_load == 0:
            return self._round_robin_schedule(batch)
            
        decisions = []
        remaining_size = batch_size
        
        for device_id, load in sorted(device_loads.items(), key=lambda x: x[1]):
            device_share = (1 - load/total_load) / (self.num_devices - 1)
            device_batch = int(batch_size * device_share)
            remaining_size -= device_batch
            
            decisions.append(SchedulingDecision(
                device_id=device_id,
                batch_size=device_batch,
                priority=1 - load
            ))
            
        # Assign remaining samples
        if remaining_size > 0:
            decisions[-1].batch_size += remaining_size
            
        return decisions
        
    def _performance_aware_schedule(self, batch: Any) -> List[SchedulingDecision]:
        """Performance-aware scheduling."""
        batch_size = self._get_batch_size(batch)
        device_performance = self._get_device_performance()
        
        # Normalize performance scores
        total_perf = sum(device_performance.values())
        if total_perf == 0:
            return self._round_robin_schedule(batch)
            
        decisions = []
        remaining_size = batch_size
        
        for device_id, perf in sorted(device_performance.items(), key=lambda x: x[1], reverse=True):
            device_share = perf / total_perf
            device_batch = int(batch_size * device_share)
            remaining_size -= device_batch
            
            decisions.append(SchedulingDecision(
                device_id=device_id,
                batch_size=device_batch,
                priority=perf
            ))
            
        # Assign remaining samples
        if remaining_size > 0:
            decisions[0].batch_size += remaining_size
            
        return decisions
        
    def _hybrid_schedule(self, batch: Any) -> List[SchedulingDecision]:
        """Hybrid scheduling combining load and performance."""
        load_decisions = self._load_aware_schedule(batch)
        perf_decisions = self._performance_aware_schedule(batch)
        
        # Combine decisions
        decisions = []
        batch_size = self._get_batch_size(batch)
        remaining_size = batch_size
        
        for ld, pd in zip(load_decisions, perf_decisions):
            # Weighted combination of batch sizes
            combined_batch = int(
                0.4 * ld.batch_size +
                0.6 * pd.batch_size
            )
            remaining_size -= combined_batch
            
            decisions.append(SchedulingDecision(
                device_id=ld.device_id,
                batch_size=combined_batch,
                priority=(ld.priority + pd.priority) / 2
            ))
            
        # Assign remaining samples
        if remaining_size > 0:
            decisions[0].batch_size += remaining_size
            
        return decisions
        
    def _get_device_loads(self) -> Dict[int, float]:
        """Get current device loads."""
        loads = {}
        for device_id, history in self.device_stats.items():
            if history:
                loads[device_id] = np.mean([
                    np.mean(stats.gpu_utilization)
                    for stats in history[-10:]  # Recent history
                ])
            else:
                loads[device_id] = 0.0
        return loads
        
    def _get_device_performance(self) -> Dict[int, float]:
        """Get device performance scores."""
        scores = {}
        for device_id, history in self.device_stats.items():
            if history:
                # Combine multiple metrics
                throughput = np.mean([stats.throughput for stats in history[-10:]])
                latency = np.mean([stats.latency for stats in history[-10:]])
                memory_usage = np.mean([
                    np.mean(stats.gpu_memory)
                    for stats in history[-10:]
                ])
                
                # Higher throughput, lower latency, lower memory usage is better
                scores[device_id] = (
                    throughput /
                    (latency * np.sqrt(memory_usage/100 + 0.1))
                )
            else:
                scores[device_id] = 0.0
        return scores
        
    def _get_batch_size(self, batch: Any) -> int:
        """Get batch size from input batch."""
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        elif isinstance(batch, tf.Tensor):
            return batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            return len(batch)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}") 