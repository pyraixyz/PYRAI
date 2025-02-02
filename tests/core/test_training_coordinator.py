"""
Tests for the Training Coordinator implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.training_coordinator import (
    TrainingCoordinator,
    TrainingStatus,
    TrainingJob,
    TrainingError
)
from src.core.model_manager import ModelManager
from src.core.resource_scheduler import ResourceScheduler


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager"""
    manager = Mock(spec=ModelManager)
    manager.load = AsyncMock(return_value=Mock())
    return manager


@pytest.fixture
def mock_resource_scheduler():
    """Create a mock resource scheduler"""
    scheduler = Mock(spec=ResourceScheduler)
    scheduler.allocate = AsyncMock(return_value="resource-123")
    scheduler.deallocate = AsyncMock(return_value=True)
    return scheduler


@pytest.fixture
async def training_coordinator(mock_model_manager, mock_resource_scheduler):
    """Create a training coordinator for testing"""
    coordinator = TrainingCoordinator(
        model_manager=mock_model_manager,
        resource_scheduler=mock_resource_scheduler
    )
    yield coordinator


class AsyncMock(Mock):
    """Mock for async functions"""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


@pytest.mark.asyncio
async def test_start_training(training_coordinator):
    """Test starting a training job"""
    config = {
        "model_id": "model-123",
        "cpu_cores": 2,
        "memory_mb": 1024,
        "gpu_devices": 1
    }
    
    # Start training
    job_id = await training_coordinator.start_training(config)
    assert job_id is not None
    
    # Check job status
    progress = await training_coordinator.get_progress(job_id)
    assert progress["status"] == TrainingStatus.RUNNING.value
    assert progress["model_id"] == "model-123"
    assert progress["progress"] == 0.0


@pytest.mark.asyncio
async def test_stop_training(training_coordinator):
    """Test stopping a training job"""
    # Start a job
    job_id = await training_coordinator.start_training({
        "model_id": "model-123"
    })
    
    # Stop the job
    success = await training_coordinator.stop_training(job_id)
    assert success
    
    # Verify job was interrupted
    await asyncio.sleep(0.1)  # Allow async tasks to complete
    progress = await training_coordinator.get_progress(job_id)
    assert progress["status"] == TrainingStatus.INTERRUPTED.value


@pytest.mark.asyncio
async def test_list_jobs(training_coordinator):
    """Test listing training jobs"""
    # Start multiple jobs
    job_ids = []
    for i in range(3):
        job_id = await training_coordinator.start_training({
            "model_id": f"model-{i}"
        })
        job_ids.append(job_id)
    
    # List all jobs
    jobs = await training_coordinator.list_jobs()
    assert len(jobs) == 3
    
    # List running jobs
    running_jobs = await training_coordinator.list_jobs(status=TrainingStatus.RUNNING.value)
    assert len(running_jobs) == 3


@pytest.mark.asyncio
async def test_get_progress_nonexistent_job(training_coordinator):
    """Test getting progress for nonexistent job"""
    with pytest.raises(TrainingError):
        await training_coordinator.get_progress("nonexistent-job")


@pytest.mark.asyncio
async def test_cleanup_jobs(training_coordinator):
    """Test cleaning up old jobs"""
    # Create completed job
    job_id = await training_coordinator.start_training({
        "model_id": "model-123"
    })
    
    # Simulate job completion
    job = training_coordinator.jobs[job_id]
    job.status = TrainingStatus.COMPLETED
    job.end_time = datetime.utcnow() - timedelta(days=10)
    
    # Clean up old jobs
    cleaned = await training_coordinator.cleanup_jobs(max_age_days=7)
    assert cleaned == 1
    assert job_id not in training_coordinator.jobs


@pytest.mark.asyncio
async def test_resource_allocation_failure(training_coordinator, mock_resource_scheduler):
    """Test handling resource allocation failure"""
    mock_resource_scheduler.allocate.side_effect = Exception("Resource allocation failed")
    
    with pytest.raises(TrainingError):
        await training_coordinator.start_training({
            "model_id": "model-123"
        })


@pytest.mark.asyncio
async def test_model_loading_failure(training_coordinator, mock_model_manager):
    """Test handling model loading failure"""
    mock_model_manager.load.side_effect = Exception("Model loading failed")
    
    job_id = await training_coordinator.start_training({
        "model_id": "model-123"
    })
    
    # Verify job failed
    await asyncio.sleep(0.1)  # Allow async tasks to complete
    progress = await training_coordinator.get_progress(job_id)
    assert progress["status"] == TrainingStatus.FAILED.value
    assert "Model loading failed" in progress["error"]


@pytest.mark.asyncio
async def test_get_job_logs(training_coordinator):
    """Test getting job logs"""
    # Start a job
    job_id = await training_coordinator.start_training({
        "model_id": "model-123"
    })
    
    # Get logs
    logs = await training_coordinator.get_job_logs(job_id)
    assert isinstance(logs, list)
    assert len(logs) > 0


@pytest.mark.asyncio
async def test_concurrent_jobs(training_coordinator):
    """Test running multiple jobs concurrently"""
    async def start_and_monitor_job(model_id):
        job_id = await training_coordinator.start_training({
            "model_id": model_id
        })
        progress = await training_coordinator.get_progress(job_id)
        return progress["status"]
    
    # Start multiple jobs concurrently
    tasks = [
        start_and_monitor_job(f"model-{i}")
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)
    
    # Verify all jobs started
    assert len(results) == 5
    assert all(status == TrainingStatus.RUNNING.value for status in results) 