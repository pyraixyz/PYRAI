import pytest
import asyncio
from engine.distributed.coordinator import DistributedCoordinator

@pytest.mark.asyncio
async def test_coordinator_initialization():
    """Test coordinator initialization."""
    config = {
        "host": "localhost",
        "port": 8000
    }
    coordinator = DistributedCoordinator(config)
    
    assert not coordinator._initialized
    await coordinator.initialize()
    assert coordinator._initialized
    
    await coordinator.shutdown()
    assert not coordinator._initialized

@pytest.mark.asyncio
async def test_worker_registration():
    """Test worker registration and management."""
    coordinator = DistributedCoordinator({"host": "localhost"})
    await coordinator.initialize()
    
    # Test worker registration
    worker_address = "worker1:8000"
    await coordinator.register_worker(worker_address)
    assert worker_address in coordinator.workers
    
    # Test worker unregistration
    await coordinator.unregister_worker(worker_address)
    assert worker_address not in coordinator.workers
    
    await coordinator.shutdown()

@pytest.mark.asyncio
async def test_task_management():
    """Test task submission and results."""
    coordinator = DistributedCoordinator({"host": "localhost"})
    await coordinator.initialize()
    
    # Submit task
    task = {
        "type": "training",
        "model_id": "test_model",
        "data_path": "test_data"
    }
    await coordinator.submit_task(task)
    
    # Check task queue
    assert coordinator.task_queue.qsize() == 1
    
    # Submit result
    result = {
        "status": "completed",
        "metrics": {"accuracy": 0.95}
    }
    await coordinator.result_queue.put(result)
    
    # Get result
    received_result = await coordinator.get_result()
    assert received_result == result
    
    await coordinator.shutdown() 