import pytest
import asyncio
import torch
import numpy as np
from engine.distributed.coordinator import DistributedCoordinator
from engine.monitoring.logger import ModelLogger

@pytest.mark.asyncio
async def test_concurrent_task_processing():
    """Test concurrent task processing."""
    coordinator = DistributedCoordinator({"host": "localhost"})
    await coordinator.initialize()
    
    # Submit multiple tasks concurrently
    num_tasks = 10
    tasks = []
    
    for i in range(num_tasks):
        task = {
            "task_id": f"task_{i}",
            "type": "training",
            "model_id": f"model_{i}"
        }
        tasks.append(coordinator.submit_task(task))
    
    # Wait for all tasks to be submitted
    await asyncio.gather(*tasks)
    
    # Verify task queue size
    assert coordinator.task_queue.qsize() == num_tasks
    
    await coordinator.shutdown()

@pytest.mark.asyncio
async def test_concurrent_model_serving(temp_dir, sample_model_pytorch):
    """Test concurrent model serving."""
    from engine.deployment.server import ModelServer
    from engine.core.versioning import VersionManager
    from engine.backends.pytorch import PyTorchBackend
    import aiohttp
    
    # Setup
    version_manager = VersionManager(str(temp_dir))
    backend = PyTorchBackend()
    server = ModelServer(version_manager, backend)
    
    # Save model
    version = version_manager.save_version(
        model=sample_model_pytorch,
        model_id="test_model",
        description="Concurrency test model",
        metrics={"accuracy": 0.95},
        parameters={"lr": 0.001}
    )
    
    # Start server
    import threading
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
    await asyncio.sleep(2)
    
    # Make concurrent requests
    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/predict",
                json={
                    "inputs": {"data": [[0.0] * 10]},
                    "model_id": "test_model",
                    "version_id": version.version_id
                }
            ) as response:
                return response.status
    
    # Send 100 concurrent requests
    tasks = [make_request() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    
    # Verify all requests succeeded
    assert all(status == 200 for status in results) 