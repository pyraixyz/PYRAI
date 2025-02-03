import pytest
import asyncio
import torch
import numpy as np
from engine.distributed.coordinator import DistributedCoordinator
from engine.distributed.worker import Worker
from engine.backends.pytorch import PyTorchTrainer

@pytest.mark.integration
@pytest.mark.asyncio
async def test_distributed_training(temp_dir):
    """Test distributed training with multiple workers."""
    # Setup coordinator
    coordinator = DistributedCoordinator({
        "host": "localhost",
        "port": 8000,
        "storage_path": str(temp_dir)
    })
    await coordinator.initialize()
    
    # Setup workers
    workers = []
    for i in range(3):
        worker = Worker({
            "worker_id": f"worker_{i}",
            "coordinator_url": "http://localhost:8000",
            "device": "cpu"
        })
        await worker.initialize()
        workers.append(worker)
    
    # Create training task
    task = {
        "type": "training",
        "model_id": "test_model",
        "model_config": {
            "type": "pytorch",
            "layers": [
                {"type": "linear", "in_features": 10, "out_features": 64},
                {"type": "relu"},
                {"type": "linear", "out_features": 2}
            ]
        },
        "training_config": {
            "batch_size": 32,
            "epochs": 2,
            "optimizer": {
                "type": "adam",
                "lr": 0.001
            }
        }
    }
    
    # Submit task
    await coordinator.submit_task(task)
    
    # Wait for results
    result = await coordinator.get_result()
    assert result["status"] == "completed"
    assert "metrics" in result
    
    # Cleanup
    for worker in workers:
        await worker.shutdown()
    await coordinator.shutdown() 