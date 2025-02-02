"""
Metrics Collector implementation for PYRAI.
Handles system metrics and training metrics collection.
"""

import os
import time
import psutil
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .interfaces import MetricsInterface
from .exceptions import MetricsError


@dataclass
class SystemMetrics:
    """System metrics data"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[List[float]] = None
    gpu_memory_used: Optional[List[int]] = None
    disk_usage: float = 0.0
    network_io: Dict[str, int] = None


@dataclass
class TrainingMetrics:
    """Training metrics data"""
    timestamp: datetime
    job_id: str
    epoch: int
    loss: float
    accuracy: float
    learning_rate: float
    batch_size: int
    samples_processed: int
    time_elapsed: float
    custom_metrics: Optional[Dict[str, float]] = None


class MetricsCollector(MetricsInterface):
    """
    Metrics Collector implementation that manages system and training metrics.
    Handles metrics collection, storage, and querying.
    """

    def __init__(self, storage_path: str = "metrics"):
        """
        Initialize the metrics collector.
        
        Args:
            storage_path: Path to store metrics data
        """
        self.storage_path = storage_path
        self.system_metrics: List[SystemMetrics] = []
        self.training_metrics: Dict[str, List[TrainingMetrics]] = {}
        self._collection_task = None
        self._is_collecting = False
        
        # Create storage directory if not exists
        os.makedirs(storage_path, exist_ok=True)

    async def start_collection(self, interval: float = 1.0):
        """
        Start metrics collection.
        
        Args:
            interval: Collection interval in seconds
            
        Raises:
            MetricsError: If collection fails to start
        """
        try:
            if self._is_collecting:
                return
                
            self._is_collecting = True
            self._collection_task = asyncio.create_task(
                self._collect_metrics(interval)
            )
            
        except Exception as e:
            raise MetricsError(f"Failed to start metrics collection: {str(e)}")

    async def stop_collection(self):
        """
        Stop metrics collection.
        
        Raises:
            MetricsError: If collection fails to stop
        """
        try:
            if not self._is_collecting:
                return
                
            self._is_collecting = False
            if self._collection_task:
                self._collection_task.cancel()
                try:
                    await self._collection_task
                except asyncio.CancelledError:
                    pass
                self._collection_task = None
                
        except Exception as e:
            raise MetricsError(f"Failed to stop metrics collection: {str(e)}")

    async def _collect_metrics(self, interval: float):
        """Collect system metrics periodically"""
        try:
            while self._is_collecting:
                metrics = await self._get_system_metrics()
                self.system_metrics.append(metrics)
                
                # Save metrics periodically
                if len(self.system_metrics) >= 100:
                    await self._save_system_metrics()
                    self.system_metrics = []
                    
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._is_collecting = False
            raise MetricsError(f"Metrics collection failed: {str(e)}")

    async def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # Get CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get GPU metrics if available
            gpu_util = None
            gpu_memory = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_util = []
                    gpu_memory = []
                    for i in range(torch.cuda.device_count()):
                        gpu_util.append(torch.cuda.utilization(i))
                        gpu_memory.append(torch.cuda.memory_allocated(i))
            except ImportError:
                pass
                
            # Get disk usage
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            
            # Get network IO
            net_io = psutil.net_io_counters()._asdict()
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_utilization=gpu_util,
                gpu_memory_used=gpu_memory,
                disk_usage=disk_percent,
                network_io=net_io
            )
            
        except Exception as e:
            raise MetricsError(f"Failed to get system metrics: {str(e)}")

    async def record_training_metrics(
        self,
        job_id: str,
        metrics: Dict[str, Any]
    ):
        """
        Record training metrics for a job.
        
        Args:
            job_id: Job identifier
            metrics: Training metrics
            
        Raises:
            MetricsError: If recording fails
        """
        try:
            if job_id not in self.training_metrics:
                self.training_metrics[job_id] = []
                
            training_metrics = TrainingMetrics(
                timestamp=datetime.utcnow(),
                job_id=job_id,
                epoch=metrics.get("epoch", 0),
                loss=metrics.get("loss", 0.0),
                accuracy=metrics.get("accuracy", 0.0),
                learning_rate=metrics.get("learning_rate", 0.0),
                batch_size=metrics.get("batch_size", 0),
                samples_processed=metrics.get("samples_processed", 0),
                time_elapsed=metrics.get("time_elapsed", 0.0),
                custom_metrics=metrics.get("custom_metrics")
            )
            
            self.training_metrics[job_id].append(training_metrics)
            
            # Save metrics periodically
            if len(self.training_metrics[job_id]) >= 100:
                await self._save_training_metrics(job_id)
                self.training_metrics[job_id] = []
                
        except Exception as e:
            raise MetricsError(f"Failed to record training metrics: {str(e)}")

    async def get_system_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[SystemMetrics]:
        """
        Get system metrics within time range.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of system metrics
            
        Raises:
            MetricsError: If retrieval fails
        """
        try:
            metrics = self.system_metrics.copy()
            
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
                
            return metrics
            
        except Exception as e:
            raise MetricsError(f"Failed to get system metrics: {str(e)}")

    async def get_training_metrics(
        self,
        job_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TrainingMetrics]:
        """
        Get training metrics for a job within time range.
        
        Args:
            job_id: Job identifier
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of training metrics
            
        Raises:
            MetricsError: If retrieval fails
        """
        try:
            if job_id not in self.training_metrics:
                return []
                
            metrics = self.training_metrics[job_id].copy()
            
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
                
            return metrics
            
        except Exception as e:
            raise MetricsError(f"Failed to get training metrics: {str(e)}")

    async def get_training_summary(self, job_id: str) -> Dict[str, Any]:
        """
        Get training metrics summary for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Summary dictionary containing:
                - total_epochs: Total epochs completed
                - avg_loss: Average loss
                - best_accuracy: Best accuracy achieved
                - total_time: Total training time
                - samples_per_second: Average samples processed per second
                
        Raises:
            MetricsError: If summary generation fails
        """
        try:
            if job_id not in self.training_metrics:
                return {}
                
            metrics = self.training_metrics[job_id]
            if not metrics:
                return {}
                
            # Calculate summary statistics
            total_epochs = max(m.epoch for m in metrics)
            avg_loss = np.mean([m.loss for m in metrics])
            best_accuracy = max(m.accuracy for m in metrics)
            total_time = sum(m.time_elapsed for m in metrics)
            total_samples = sum(m.samples_processed for m in metrics)
            
            return {
                "total_epochs": total_epochs,
                "avg_loss": float(avg_loss),
                "best_accuracy": float(best_accuracy),
                "total_time": total_time,
                "samples_per_second": total_samples / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            raise MetricsError(f"Failed to get training summary: {str(e)}")

    async def _save_system_metrics(self):
        """Save system metrics to storage"""
        try:
            # TODO: Implement actual storage (e.g., to database or file)
            pass
        except Exception as e:
            raise MetricsError(f"Failed to save system metrics: {str(e)}")

    async def _save_training_metrics(self, job_id: str):
        """Save training metrics to storage"""
        try:
            # TODO: Implement actual storage (e.g., to database or file)
            pass
        except Exception as e:
            raise MetricsError(f"Failed to save training metrics: {str(e)}")

    async def cleanup_metrics(self, max_age_days: int = 30):
        """
        Clean up old metrics data.
        
        Args:
            max_age_days: Maximum age of metrics to keep
            
        Raises:
            MetricsError: If cleanup fails
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
            
            # Clean up system metrics
            self.system_metrics = [
                m for m in self.system_metrics
                if m.timestamp >= cutoff_time
            ]
            
            # Clean up training metrics
            for job_id in list(self.training_metrics.keys()):
                self.training_metrics[job_id] = [
                    m for m in self.training_metrics[job_id]
                    if m.timestamp >= cutoff_time
                ]
                if not self.training_metrics[job_id]:
                    del self.training_metrics[job_id]
                    
        except Exception as e:
            raise MetricsError(f"Failed to cleanup metrics: {str(e)}") 