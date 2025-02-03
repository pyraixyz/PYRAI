import asyncio
from typing import Dict, List, Any, Optional
from ..core.base import Engine
import aiohttp

class DistributedCoordinator(Engine):
    """Coordinates distributed training across multiple nodes."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.workers: List[str] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
    
    async def initialize(self) -> None:
        """Initialize the coordinator."""
        self._initialized = True
        await self._start_worker_monitoring()
    
    async def shutdown(self) -> None:
        """Shutdown the coordinator."""
        await self._stop_worker_monitoring()
        self._initialized = False
    
    async def register_worker(self, worker_address: str) -> None:
        """Register a new worker."""
        if worker_address not in self.workers:
            self.workers.append(worker_address)
    
    async def unregister_worker(self, worker_address: str) -> None:
        """Unregister a worker."""
        if worker_address in self.workers:
            self.workers.remove(worker_address)
    
    async def submit_task(self, task: Dict[str, Any]) -> None:
        """Submit a task for distributed execution."""
        await self.task_queue.put(task)
    
    async def get_result(self) -> Dict[str, Any]:
        """Get a task result."""
        return await self.result_queue.get()
    
    async def _start_worker_monitoring(self) -> None:
        """Start monitoring worker health."""
        # Implementation for worker health monitoring
        pass
    
    async def _stop_worker_monitoring(self) -> None:
        """Stop monitoring worker health."""
        # Implementation for stopping worker monitoring
        pass

    async def _monitor_workers(self) -> None:
        """Monitor worker health and manage task distribution."""
        while True:
            try:
                # Remove unresponsive workers
                current_time = asyncio.get_event_loop().time()
                unresponsive = []
                for worker in self.workers:
                    if current_time - worker.last_heartbeat > 60:  # 60 seconds timeout
                        unresponsive.append(worker)
                
                for worker in unresponsive:
                    await self.unregister_worker(worker.address)
                    
                # Distribute pending tasks
                while not self.task_queue.empty() and self.workers:
                    task = await self.task_queue.get()
                    worker = self._select_worker(task)
                    if worker:
                        asyncio.create_task(self._execute_task(worker, task))
                    else:
                        await self.task_queue.put(task)  # Re-queue if no worker available
                    
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker monitoring error: {e}")
                await asyncio.sleep(5)

    def _select_worker(self, task: Dict[str, Any]) -> Optional[str]:
        """Select the best worker for a given task."""
        if not self.workers:
            return None
        
        # Simple selection strategy - can be enhanced
        available_workers = [w for w in self.workers if w.status['load'] < 0.8]
        if not available_workers:
            return None
        
        # Select worker with lowest load
        return min(available_workers, key=lambda w: w.status['load'])

    async def _execute_task(self, worker: str, task: Dict[str, Any]) -> None:
        """Execute a task on a worker."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{worker}/execute",
                    json=task
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        await self.result_queue.put(result)
                    else:
                        # Handle task execution failure
                        await self.task_queue.put(task)  # Re-queue failed task
        except Exception as e:
            print(f"Task execution error: {e}")
            await self.task_queue.put(task)  # Re-queue failed task 