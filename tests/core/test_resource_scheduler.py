"""
Tests for the Resource Scheduler implementation.
"""

import pytest
import psutil
from datetime import datetime

from src.core.resource_scheduler import ResourceScheduler
from src.core.exceptions import ResourceError, ResourceExhaustedError


@pytest.fixture
async def resource_scheduler():
    """Create a resource scheduler for testing"""
    scheduler = ResourceScheduler()
    yield scheduler


@pytest.mark.asyncio
async def test_resource_initialization(resource_scheduler):
    """Test resource scheduler initialization"""
    status = await resource_scheduler.get_status()
    
    assert status["total_cpu_cores"] > 0
    assert status["total_memory_mb"] > 0
    assert "total_gpu_devices" in status
    assert len(status["allocations"]) == 0


@pytest.mark.asyncio
async def test_basic_allocation(resource_scheduler):
    """Test basic resource allocation"""
    requirements = {
        "cpu_cores": 2,
        "memory_mb": 1024,
        "gpu_devices": 0
    }
    
    # Allocate resources
    allocation_id = await resource_scheduler.allocate(requirements)
    assert allocation_id is not None
    
    # Check allocation
    allocation = await resource_scheduler.get_allocation(allocation_id)
    assert allocation is not None
    assert len(allocation["cpu_cores"]) == 2
    assert allocation["memory_mb"] == 1024
    assert len(allocation["gpu_devices"]) == 0
    assert allocation["status"] == "active"


@pytest.mark.asyncio
async def test_resource_exhaustion(resource_scheduler):
    """Test resource exhaustion handling"""
    # Request more CPU cores than available
    requirements = {
        "cpu_cores": psutil.cpu_count() + 1,
        "memory_mb": 1024
    }
    
    with pytest.raises(ResourceExhaustedError):
        await resource_scheduler.allocate(requirements)


@pytest.mark.asyncio
async def test_memory_allocation(resource_scheduler):
    """Test memory allocation"""
    total_memory = resource_scheduler.total_memory_mb
    
    # Allocate half of available memory
    requirements = {
        "cpu_cores": 1,
        "memory_mb": total_memory // 2
    }
    
    allocation_id = await resource_scheduler.allocate(requirements)
    assert allocation_id is not None
    
    # Try to allocate more than remaining memory
    with pytest.raises(ResourceExhaustedError):
        await resource_scheduler.allocate({
            "cpu_cores": 1,
            "memory_mb": total_memory
        })


@pytest.mark.asyncio
async def test_deallocation(resource_scheduler):
    """Test resource deallocation"""
    requirements = {
        "cpu_cores": 1,
        "memory_mb": 1024
    }
    
    # Allocate and then deallocate
    allocation_id = await resource_scheduler.allocate(requirements)
    assert await resource_scheduler.deallocate(allocation_id)
    
    # Verify deallocation
    status = await resource_scheduler.get_status()
    assert status["used_cpu_cores"] == 0
    assert status["used_memory_mb"] == 0
    
    # Try to deallocate again
    assert not await resource_scheduler.deallocate(allocation_id)


@pytest.mark.asyncio
async def test_allocation_tracking(resource_scheduler):
    """Test allocation tracking"""
    requirements = {
        "cpu_cores": 2,
        "memory_mb": 1024,
        "owner": "test_user"
    }
    
    # Create allocation
    allocation_id = await resource_scheduler.allocate(requirements)
    
    # Check status
    status = await resource_scheduler.get_status()
    assert len(status["allocations"]) == 1
    assert status["allocations"][0]["owner"] == "test_user"
    
    # Update allocation
    assert await resource_scheduler.update_allocation(
        allocation_id,
        {"status": "completed"}
    )
    
    # Verify update
    allocation = await resource_scheduler.get_allocation(allocation_id)
    assert allocation["status"] == "completed"


@pytest.mark.asyncio
async def test_concurrent_allocations(resource_scheduler):
    """Test concurrent resource allocations"""
    async def allocate_and_deallocate():
        allocation_id = await resource_scheduler.allocate({
            "cpu_cores": 1,
            "memory_mb": 512
        })
        await resource_scheduler.deallocate(allocation_id)
        return allocation_id
    
    # Run multiple allocations concurrently
    tasks = [allocate_and_deallocate() for _ in range(5)]
    allocation_ids = await asyncio.gather(*tasks)
    
    # Verify all allocations were successful
    assert len(allocation_ids) == 5
    assert len(set(allocation_ids)) == 5  # All IDs should be unique
    
    # Verify all resources were deallocated
    status = await resource_scheduler.get_status()
    assert status["used_cpu_cores"] == 0
    assert status["used_memory_mb"] == 0 