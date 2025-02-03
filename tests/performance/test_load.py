import pytest
import asyncio
import aiohttp
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from engine.deployment.server import ModelServer
from engine.core.versioning import VersionManager
from engine.backends.pytorch import PyTorchBackend

@pytest.mark.performance
async def test_server_load(temp_dir, sample_model_pytorch):
    """Test server under load."""
    # Setup server
    version_manager = VersionManager(str(temp_dir))
    backend = PyTorchBackend()
    
    server = ModelServer(
        version_manager=version_manager,
        backend=backend,
        host="localhost",
        port=8000
    )
    
    # Save test model
    version = version_manager.save_version(
        model=sample_model_pytorch,
        model_id="test_model",
        description="Load test model",
        metrics={"accuracy": 0.95},
        parameters={"lr": 0.001}
    )
    
    # Start server in background
    import threading
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    
    # Setup load test parameters
    num_requests = 1000
    concurrent_users = 50
    
    async def make_request():
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.post(
                "http://localhost:8000/predict",
                json={
                    "inputs": {"data": [[0.0] * 10]},
                    "model_id": "test_model",
                    "version_id": version.version_id
                }
            ) as response:
                latency = time.time() - start_time
                return response.status, latency
    
    # Run load test
    tasks = []
    start_time = time.time()
    
    for _ in range(num_requests):
        if len(tasks) >= concurrent_users:
            done, tasks = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
        tasks.append(asyncio.create_task(make_request()))
    
    # Wait for remaining requests
    if tasks:
        await asyncio.wait(tasks)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    results = [task.result() for task in tasks]
    status_codes = [r[0] for r in results]
    latencies = [r[1] for r in results]
    
    # Assert performance requirements
    assert np.mean(latencies) < 0.1  # Average latency under 100ms
    assert np.percentile(latencies, 95) < 0.2  # P95 latency under 200ms
    assert sum(s == 200 for s in status_codes) / len(status_codes) > 0.99  # 99% success
    assert num_requests / total_time > 100  # Minimum 100 RPS 