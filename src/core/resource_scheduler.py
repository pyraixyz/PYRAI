"""
Resource Scheduler implementation for PYRAI.
Handles resource allocation and management for AI workloads.
"""

import os
import psutil
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

import torch
import tensorflow as tf

from .interfaces import ResourceInterface
from .exceptions import ResourceError, ResourceExhaustedError
from .config import config


@dataclass
class ResourceAllocation:
    """Resource allocation details"""
    id: str
    cpu_cores: List[int]
    memory_mb: int
    gpu_devices: List[int]
    timestamp: datetime
    status: str
    owner: Optional[str] = None


class ResourceScheduler(ResourceInterface):
    """
    Resource Scheduler implementation that manages compute resources.
    Handles CPU, GPU, and memory allocation for AI workloads.
    """

    def __init__(self):
        """Initialize the resource scheduler"""
        self.allocations: Dict[str, ResourceAllocation] = {}
        self._used_cpu_cores: Set[int] = set()
        self._used_gpu_devices: Set[int] = set()
        self._used_memory_mb: int = 0
        self._setup_resources()

    def _setup_resources(self) -> None:
        """Setup and validate available resources"""
        try:
            # CPU resources
            self.total_cpu_cores = psutil.cpu_count(logical=True)
            
            # Memory resources
            self.total_memory_mb = psutil.virtual_memory().total // (1024 * 1024)
            
            # GPU resources
            self.total_gpu_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            # Validate minimum requirements
            if self.total_memory_mb < config.min_memory_required:
                raise ResourceError("System does not meet minimum memory requirements")
                
        except Exception as e:
            raise ResourceError(f"Failed to setup resources: {str(e)}")

    def _get_available_cpu_cores(self, count: int) -> List[int]:
        """Get available CPU cores"""
        available_cores = [
            i for i in range(self.total_cpu_cores)
            if i not in self._used_cpu_cores
        ]
        
        if len(available_cores) < count:
            raise ResourceExhaustedError("Not enough CPU cores available")
            
        return available_cores[:count]

    def _get_available_gpu_devices(self, count: int) -> List[int]:
        """Get available GPU devices"""
        if count > 0 and self.total_gpu_devices == 0:
            raise ResourceError("No GPU devices available in the system")
            
        available_devices = [
            i for i in range(self.total_gpu_devices)
            if i not in self._used_gpu_devices
        ]
        
        if len(available_devices) < count:
            raise ResourceExhaustedError("Not enough GPU devices available")
            
        return available_devices[:count]

    async def allocate(self, requirements: Dict[str, Any]) -> str:
        """
        Allocate resources based on requirements.
        
        Args:
            requirements: Dictionary containing resource requirements:
                - cpu_cores: Number of CPU cores needed
                - memory_mb: Amount of memory needed in MB
                - gpu_devices: Number of GPU devices needed
                - owner: Optional owner identifier
                
        Returns:
            str: Allocation ID
            
        Raises:
            ResourceError: If allocation fails
            ResourceExhaustedError: If resources are exhausted
        """
        try:
            # Validate requirements
            cpu_cores = requirements.get("cpu_cores", 1)
            memory_mb = requirements.get("memory_mb", 1024)
            gpu_devices = requirements.get("gpu_devices", 0)
            owner = requirements.get("owner")
            
            # Check memory availability
            if self._used_memory_mb + memory_mb > self.total_memory_mb:
                raise ResourceExhaustedError("Not enough memory available")
            
            # Get available resources
            allocated_cores = self._get_available_cpu_cores(cpu_cores)
            allocated_gpus = self._get_available_gpu_devices(gpu_devices)
            
            # Create allocation
            allocation_id = str(uuid.uuid4())
            allocation = ResourceAllocation(
                id=allocation_id,
                cpu_cores=allocated_cores,
                memory_mb=memory_mb,
                gpu_devices=allocated_gpus,
                timestamp=datetime.utcnow(),
                status="active",
                owner=owner
            )
            
            # Update resource tracking
            self._used_cpu_cores.update(allocated_cores)
            self._used_gpu_devices.update(allocated_gpus)
            self._used_memory_mb += memory_mb
            
            # Store allocation
            self.allocations[allocation_id] = allocation
            
            return allocation_id
            
        except (ResourceError, ResourceExhaustedError):
            raise
        except Exception as e:
            raise ResourceError(f"Failed to allocate resources: {str(e)}")

    async def deallocate(self, resource_id: str) -> bool:
        """
        Deallocate resources.
        
        Args:
            resource_id: The allocation ID to deallocate
            
        Returns:
            bool: True if deallocation was successful
            
        Raises:
            ResourceError: If deallocation fails
        """
        try:
            if resource_id not in self.allocations:
                return False
                
            allocation = self.allocations[resource_id]
            
            # Update resource tracking
            self._used_cpu_cores.difference_update(allocation.cpu_cores)
            self._used_gpu_devices.difference_update(allocation.gpu_devices)
            self._used_memory_mb -= allocation.memory_mb
            
            # Remove allocation
            del self.allocations[resource_id]
            
            return True
            
        except Exception as e:
            raise ResourceError(f"Failed to deallocate resources: {str(e)}")

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current resource status.
        
        Returns:
            Dictionary containing resource status:
                - total_cpu_cores: Total CPU cores
                - used_cpu_cores: Number of used CPU cores
                - total_memory_mb: Total memory in MB
                - used_memory_mb: Used memory in MB
                - total_gpu_devices: Total GPU devices
                - used_gpu_devices: Number of used GPU devices
                - allocations: List of current allocations
                
        Raises:
            ResourceError: If status retrieval fails
        """
        try:
            return {
                "total_cpu_cores": self.total_cpu_cores,
                "used_cpu_cores": len(self._used_cpu_cores),
                "total_memory_mb": self.total_memory_mb,
                "used_memory_mb": self._used_memory_mb,
                "total_gpu_devices": self.total_gpu_devices,
                "used_gpu_devices": len(self._used_gpu_devices),
                "allocations": [
                    {
                        "id": alloc.id,
                        "cpu_cores": len(alloc.cpu_cores),
                        "memory_mb": alloc.memory_mb,
                        "gpu_devices": len(alloc.gpu_devices),
                        "timestamp": alloc.timestamp.isoformat(),
                        "status": alloc.status,
                        "owner": alloc.owner
                    }
                    for alloc in self.allocations.values()
                ]
            }
            
        except Exception as e:
            raise ResourceError(f"Failed to get resource status: {str(e)}")

    async def get_allocation(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific allocation.
        
        Args:
            resource_id: The allocation ID
            
        Returns:
            Allocation details dictionary or None if not found
            
        Raises:
            ResourceError: If retrieval fails
        """
        try:
            allocation = self.allocations.get(resource_id)
            if allocation is None:
                return None
                
            return {
                "id": allocation.id,
                "cpu_cores": allocation.cpu_cores,
                "memory_mb": allocation.memory_mb,
                "gpu_devices": allocation.gpu_devices,
                "timestamp": allocation.timestamp.isoformat(),
                "status": allocation.status,
                "owner": allocation.owner
            }
            
        except Exception as e:
            raise ResourceError(f"Failed to get allocation details: {str(e)}")

    async def update_allocation(self, resource_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing allocation.
        
        Args:
            resource_id: The allocation ID
            updates: Dictionary containing updates
            
        Returns:
            bool: True if update was successful
            
        Raises:
            ResourceError: If update fails
        """
        try:
            if resource_id not in self.allocations:
                return False
                
            allocation = self.allocations[resource_id]
            
            # Update allowed fields
            if "status" in updates:
                allocation.status = updates["status"]
            if "owner" in updates:
                allocation.owner = updates["owner"]
                
            return True
            
        except Exception as e:
            raise ResourceError(f"Failed to update allocation: {str(e)}") 