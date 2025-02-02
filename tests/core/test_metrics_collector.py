"""
Tests for the Metrics Collector implementation.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.metrics_collector import (
    MetricsCollector,
    SystemMetrics,
    TrainingMetrics,
    MetricsError
)


@pytest.fixture
async def metrics_collector():
    """Create a metrics collector for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = MetricsCollector(storage_path=temp_dir)
        yield collector


@pytest.fixture
def mock_psutil():
    """Create mock psutil functions"""
    with patch("psutil.cpu_percent") as cpu_mock, \
         patch("psutil.virtual_memory") as mem_mock, \
         patch("psutil.disk_usage") as disk_mock, \
         patch("psutil.net_io_counters") as net_mock:
        
        # Setup mock returns
        cpu_mock.return_value = 50.0
        
        mem_mock.return_value = Mock(
            total=16000000000,
            available=8000000000,
            percent=50.0
        )
        
        disk_mock.return_value = Mock(
            total=1000000000000,
            used=500000000000,
            free=500000000000,
            percent=50.0
        )
        
        net_mock.return_value = Mock(
            _asdict=lambda: {
                "bytes_sent": 1000000,
                "bytes_recv": 2000000
            }
        )
        
        yield {
            "cpu": cpu_mock,
            "memory": mem_mock,
            "disk": disk_mock,
            "network": net_mock
        }


@pytest.mark.asyncio
async def test_start_collection(metrics_collector, mock_psutil):
    """Test starting metrics collection"""
    await metrics_collector.start_collection(interval=0.1)
    assert metrics_collector._is_collecting
    assert metrics_collector._collection_task is not None
    
    # Wait for some metrics to be collected
    await asyncio.sleep(0.3)
    
    # Stop collection
    await metrics_collector.stop_collection()
    
    # Verify metrics were collected
    assert len(metrics_collector.system_metrics) > 0
    
    # Verify metric values
    metric = metrics_collector.system_metrics[0]
    assert metric.cpu_percent == 50.0
    assert metric.memory_percent == 50.0
    assert metric.disk_usage == 50.0
    assert metric.network_io["bytes_sent"] == 1000000


@pytest.mark.asyncio
async def test_stop_collection(metrics_collector):
    """Test stopping metrics collection"""
    await metrics_collector.start_collection()
    await metrics_collector.stop_collection()
    
    assert not metrics_collector._is_collecting
    assert metrics_collector._collection_task is None


@pytest.mark.asyncio
async def test_record_training_metrics(metrics_collector):
    """Test recording training metrics"""
    job_id = "job-123"
    metrics = {
        "epoch": 1,
        "loss": 0.5,
        "accuracy": 0.8,
        "learning_rate": 0.001,
        "batch_size": 32,
        "samples_processed": 1000,
        "time_elapsed": 10.0,
        "custom_metrics": {
            "f1_score": 0.75
        }
    }
    
    await metrics_collector.record_training_metrics(job_id, metrics)
    
    # Verify metrics were recorded
    assert job_id in metrics_collector.training_metrics
    assert len(metrics_collector.training_metrics[job_id]) == 1
    
    # Verify metric values
    recorded = metrics_collector.training_metrics[job_id][0]
    assert recorded.epoch == 1
    assert recorded.loss == 0.5
    assert recorded.accuracy == 0.8
    assert recorded.learning_rate == 0.001
    assert recorded.custom_metrics["f1_score"] == 0.75


@pytest.mark.asyncio
async def test_get_system_metrics(metrics_collector, mock_psutil):
    """Test getting system metrics"""
    # Record some metrics
    await metrics_collector.start_collection(interval=0.1)
    await asyncio.sleep(0.3)
    await metrics_collector.stop_collection()
    
    # Get metrics without time filter
    metrics = await metrics_collector.get_system_metrics()
    assert len(metrics) > 0
    
    # Get metrics with time filter
    start_time = datetime.utcnow() - timedelta(minutes=1)
    end_time = datetime.utcnow()
    
    filtered = await metrics_collector.get_system_metrics(
        start_time=start_time,
        end_time=end_time
    )
    assert len(filtered) > 0
    assert all(start_time <= m.timestamp <= end_time for m in filtered)


@pytest.mark.asyncio
async def test_get_training_metrics(metrics_collector):
    """Test getting training metrics"""
    job_id = "job-123"
    
    # Record some metrics
    for i in range(5):
        await metrics_collector.record_training_metrics(job_id, {
            "epoch": i,
            "loss": 1.0 - i * 0.2,
            "accuracy": i * 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "samples_processed": 1000,
            "time_elapsed": 10.0
        })
    
    # Get metrics without time filter
    metrics = await metrics_collector.get_training_metrics(job_id)
    assert len(metrics) == 5
    
    # Get metrics with time filter
    start_time = datetime.utcnow() - timedelta(minutes=1)
    end_time = datetime.utcnow()
    
    filtered = await metrics_collector.get_training_metrics(
        job_id,
        start_time=start_time,
        end_time=end_time
    )
    assert len(filtered) == 5
    assert all(start_time <= m.timestamp <= end_time for m in filtered)


@pytest.mark.asyncio
async def test_get_training_summary(metrics_collector):
    """Test getting training summary"""
    job_id = "job-123"
    
    # Record some metrics
    for i in range(5):
        await metrics_collector.record_training_metrics(job_id, {
            "epoch": i,
            "loss": 1.0 - i * 0.2,
            "accuracy": i * 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "samples_processed": 1000,
            "time_elapsed": 10.0
        })
    
    # Get summary
    summary = await metrics_collector.get_training_summary(job_id)
    
    assert summary["total_epochs"] == 4
    assert 0.0 <= summary["avg_loss"] <= 1.0
    assert 0.0 <= summary["best_accuracy"] <= 1.0
    assert summary["total_time"] == 50.0
    assert summary["samples_per_second"] == 100.0


@pytest.mark.asyncio
async def test_cleanup_metrics(metrics_collector):
    """Test cleaning up old metrics"""
    job_id = "job-123"
    
    # Add old system metrics
    old_time = datetime.utcnow() - timedelta(days=40)
    metrics_collector.system_metrics.append(
        SystemMetrics(
            timestamp=old_time,
            cpu_percent=50.0,
            memory_percent=50.0,
            disk_usage=50.0,
            network_io={}
        )
    )
    
    # Add old training metrics
    metrics_collector.training_metrics[job_id] = [
        TrainingMetrics(
            timestamp=old_time,
            job_id=job_id,
            epoch=0,
            loss=1.0,
            accuracy=0.0,
            learning_rate=0.001,
            batch_size=32,
            samples_processed=0,
            time_elapsed=0.0
        )
    ]
    
    # Clean up metrics
    await metrics_collector.cleanup_metrics(max_age_days=30)
    
    # Verify old metrics were removed
    assert len(metrics_collector.system_metrics) == 0
    assert job_id not in metrics_collector.training_metrics


@pytest.mark.asyncio
async def test_error_handling(metrics_collector):
    """Test error handling"""
    # Test invalid job ID
    metrics = await metrics_collector.get_training_metrics("invalid-job")
    assert len(metrics) == 0
    
    summary = await metrics_collector.get_training_summary("invalid-job")
    assert not summary
    
    # Test invalid metrics data
    with pytest.raises(MetricsError):
        await metrics_collector.record_training_metrics(None, {}) 