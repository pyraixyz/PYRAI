from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from enum import Enum

@dataclass
class DeviceStats:
    """Device performance statistics."""
    device_id: int
    compute_capacity: float  # FLOPS
    memory_capacity: float   # Bytes
    memory_bandwidth: float  # Bytes/s
    current_load: float     # Utilization (0-1)

class BalanceStrategy(Enum):
    """Load balancing strategy."""
    GREEDY = "greedy"
    ROUND_ROBIN = "round_robin"
    MEMORY_AWARE = "memory_aware"
    ADAPTIVE = "adaptive"

class LoadBalancer:
    """Dynamic load balancer for parallel training."""
    
    def __init__(
        self,
        strategy: BalanceStrategy = BalanceStrategy.ADAPTIVE,
        monitoring_interval: float = 1.0
    ):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
            monitoring_interval: Device monitoring interval in seconds
        """
        self.strategy = strategy
        self.monitoring_interval = monitoring_interval
        self.device_stats = {}
        self._initialize_device_stats()
        
    def _initialize_device_stats(self):
        """Initialize device statistics."""
        # PyTorch GPU stats
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.device_stats[f"cuda:{i}"] = DeviceStats(
                    device_id=i,
                    compute_capacity=props.multi_processor_count * props.max_threads_per_block,
                    memory_capacity=props.total_memory,
                    memory_bandwidth=props.memory_clock_rate * props.memory_bus_width / 8,
                    current_load=0.0
                )
                
        # TensorFlow GPU stats
        gpus = tf.config.list_physical_devices('GPU')
        for i, gpu in enumerate(gpus):
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu)
                self.device_stats[f"tf_gpu:{i}"] = DeviceStats(
                    device_id=i,
                    compute_capacity=0.0,  # To be measured dynamically
                    memory_capacity=memory_info['peak'],
                    memory_bandwidth=0.0,  # To be measured dynamically
                    current_load=0.0
                )
            except:
                pass
                
    def get_optimal_device(
        self,
        operation_size: int,
        memory_requirement: int
    ) -> str:
        """Get optimal device for operation."""
        if self.strategy == BalanceStrategy.GREEDY:
            return self._greedy_assignment(operation_size, memory_requirement)
        elif self.strategy == BalanceStrategy.ROUND_ROBIN:
            return self._round_robin_assignment()
        elif self.strategy == BalanceStrategy.MEMORY_AWARE:
            return self._memory_aware_assignment(memory_requirement)
        else:  # Adaptive
            return self._adaptive_assignment(operation_size, memory_requirement)
            
    def update_device_stats(self):
        """Update device statistics."""
        for device_id, stats in self.device_stats.items():
            if device_id.startswith("cuda:"):
                # Update PyTorch GPU stats
                torch.cuda.synchronize(device_id)
                memory_stats = torch.cuda.memory_stats(device_id)
                stats.current_load = memory_stats["allocated_bytes.all.current"] / stats.memory_capacity
            else:
                # Update TensorFlow GPU stats
                try:
                    memory_info = tf.config.experimental.get_memory_info(f"GPU:{stats.device_id}")
                    stats.current_load = memory_info['current'] / memory_info['peak']
                except:
                    pass
                    
    def _greedy_assignment(
        self,
        operation_size: int,
        memory_requirement: int
    ) -> str:
        """Assign to least loaded device."""
        self.update_device_stats()
        devices = sorted(
            self.device_stats.items(),
            key=lambda x: x[1].current_load
        )
        return devices[0][0]
        
    def _round_robin_assignment(self) -> str:
        """Simple round-robin assignment."""
        devices = list(self.device_stats.keys())
        self._current_device = getattr(self, '_current_device', -1) + 1
        return devices[self._current_device % len(devices)]
        
    def _memory_aware_assignment(
        self,
        memory_requirement: int
    ) -> str:
        """Assign based on memory availability."""
        self.update_device_stats()
        for device_id, stats in self.device_stats.items():
            available_memory = stats.memory_capacity * (1 - stats.current_load)
            if available_memory >= memory_requirement:
                return device_id
        raise RuntimeError("No device has sufficient memory")
        
    def _adaptive_assignment(
        self,
        operation_size: int,
        memory_requirement: int
    ) -> str:
        """Adaptive assignment based on multiple factors."""
        self.update_device_stats()
        device_scores = {}
        
        for device_id, stats in self.device_stats.items():
            # Calculate score based on multiple factors
            memory_score = (1 - stats.current_load) * stats.memory_capacity / memory_requirement
            compute_score = stats.compute_capacity / operation_size
            bandwidth_score = stats.memory_bandwidth / (memory_requirement * 2)  # Input + output
            
            # Weighted combination
            device_scores[device_id] = (
                0.4 * memory_score +
                0.4 * compute_score +
                0.2 * bandwidth_score
            )
            
        return max(device_scores.items(), key=lambda x: x[1])[0] 