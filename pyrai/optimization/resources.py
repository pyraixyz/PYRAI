from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ResourceType(Enum):
    """Types of resources."""
    GPU_MEMORY = "gpu_memory"
    CPU_MEMORY = "cpu_memory"
    COMPUTE = "compute"
    NETWORK = "network"

@dataclass
class ResourceAllocation:
    """Resource allocation."""
    resource_type: ResourceType
    device_id: int
    amount: float
    priority: float

class ResourceManager:
    """Manages resource allocation and reservation."""
    
    def __init__(
        self,
        num_devices: int,
        gpu_memory_buffer: float = 0.1,  # 10% buffer
        enable_oversubscription: bool = False
    ):
        """Initialize resource manager.
        
        Args:
            num_devices: Number of devices
            gpu_memory_buffer: GPU memory buffer ratio
            enable_oversubscription: Whether to allow oversubscription
        """
        self.num_devices = num_devices
        self.gpu_memory_buffer = gpu_memory_buffer
        self.enable_oversubscription = enable_oversubscription
        self.allocations: Dict[int, List[ResourceAllocation]] = {
            i: [] for i in range(num_devices)
        }
        self._initialize_resources()
        
    def _initialize_resources(self):
        """Initialize resource tracking."""
        self.total_resources = {}
        self.available_resources = {}
        
        if torch.cuda.is_available():
            for i in range(self.num_devices):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory * (1 - self.gpu_memory_buffer)
                self.total_resources[i] = {
                    ResourceType.GPU_MEMORY: total_memory,
                    ResourceType.COMPUTE: props.multi_processor_count
                }
                self.available_resources[i] = self.total_resources[i].copy()
                
    def allocate(
        self,
        resource_type: ResourceType,
        device_id: int,
        amount: float,
        priority: float = 1.0
    ) -> bool:
        """Allocate resources.
        
        Args:
            resource_type: Type of resource
            device_id: Device ID
            amount: Amount to allocate
            priority: Allocation priority
            
        Returns:
            Whether allocation was successful
        """
        if device_id not in self.available_resources:
            return False
            
        available = self.available_resources[device_id].get(resource_type, 0)
        if amount > available and not self.enable_oversubscription:
            return False
            
        allocation = ResourceAllocation(
            resource_type=resource_type,
            device_id=device_id,
            amount=amount,
            priority=priority
        )
        
        self.allocations[device_id].append(allocation)
        self.available_resources[device_id][resource_type] -= amount
        return True
        
    def release(
        self,
        resource_type: ResourceType,
        device_id: int,
        amount: float
    ) -> bool:
        """Release allocated resources."""
        if device_id not in self.available_resources:
            return False
            
        self.available_resources[device_id][resource_type] += amount
        
        # Remove matching allocations
        self.allocations[device_id] = [
            alloc for alloc in self.allocations[device_id]
            if not (
                alloc.resource_type == resource_type and
                alloc.amount == amount
            )
        ]
        return True
        
    def get_available(
        self,
        resource_type: ResourceType,
        device_id: int
    ) -> float:
        """Get available resource amount."""
        if device_id not in self.available_resources:
            return 0.0
        return self.available_resources[device_id].get(resource_type, 0.0)
        
    def get_utilization(
        self,
        resource_type: ResourceType,
        device_id: int
    ) -> float:
        """Get resource utilization ratio."""
        if device_id not in self.total_resources:
            return 0.0
            
        total = self.total_resources[device_id].get(resource_type, 0.0)
        if total == 0:
            return 0.0
            
        available = self.available_resources[device_id].get(resource_type, 0.0)
        return (total - available) / total 