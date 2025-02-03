import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional
from ..core.base import Engine

class Worker(Engine):
    """Worker node for distributed training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.coordinator_url = config['coordinator_url']
        self.worker_id = config['worker_id']
        self.device = config.get('device', 'CPU')
        self.session: Optional[aiohttp.ClientSession] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the worker."""
        self.session = aiohttp.ClientSession()
        await self._register_with_coordinator()
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._initialized = True
        
    async def shutdown(self) -> None:
        """Shutdown the worker."""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        await self._unregister_from_coordinator()
        if self.session:
            await self.session.close()
        self._initialized = False
    
    async def _register_with_coordinator(self) -> None:
        """Register this worker with the coordinator."""
        if not self.session:
            return
            
        async with self.session.post(
            f"{self.coordinator_url}/register",
            json={
                "worker_id": self.worker_id,
                "device": self.device,
                "capabilities": self._get_capabilities()
            }
        ) as response:
            if response.status != 200:
                raise RuntimeError("Failed to register with coordinator")
    
    async def _unregister_from_coordinator(self) -> None:
        """Unregister this worker from the coordinator."""
        if not self.session:
            return
            
        async with self.session.post(
            f"{self.coordinator_url}/unregister",
            json={"worker_id": self.worker_id}
        ) as response:
            if response.status != 200:
                raise RuntimeError("Failed to unregister from coordinator")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to coordinator."""
        while True:
            try:
                if self.session:
                    async with self.session.post(
                        f"{self.coordinator_url}/heartbeat",
                        json={
                            "worker_id": self.worker_id,
                            "status": self._get_status()
                        }
                    ) as response:
                        if response.status != 200:
                            print(f"Heartbeat failed: {response.status}")
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")
                await asyncio.sleep(5)  # Retry after 5 seconds on error
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get worker capabilities."""
        import torch
        import tensorflow as tf
        
        capabilities = {
            "frameworks": [],
            "gpu_available": False,
            "memory_available": self._get_available_memory()
        }
        
        # Check PyTorch
        try:
            if torch.cuda.is_available():
                capabilities["frameworks"].append("pytorch-gpu")
                capabilities["gpu_available"] = True
            else:
                capabilities["frameworks"].append("pytorch-cpu")
        except:
            pass
            
        # Check TensorFlow
        try:
            if tf.config.list_physical_devices('GPU'):
                capabilities["frameworks"].append("tensorflow-gpu")
                capabilities["gpu_available"] = True
            else:
                capabilities["frameworks"].append("tensorflow-cpu")
        except:
            pass
            
        return capabilities
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            "load": self._get_system_load(),
            "memory_available": self._get_available_memory(),
            "gpu_utilization": self._get_gpu_utilization()
        }
    
    def _get_system_load(self) -> float:
        """Get system CPU load."""
        import psutil
        return psutil.cpu_percent() / 100.0
    
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        import psutil
        return psutil.virtual_memory().available
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.utilization() / 100.0
        except:
            pass
        return None 