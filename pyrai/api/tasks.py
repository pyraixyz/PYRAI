"""Asynchronous task queue implementation."""

from typing import Dict, Any, Optional
import asyncio
import uuid
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger()

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    """Async task representation."""
    
    def __init__(self, func, *args, **kwargs):
        self.id = str(uuid.uuid4())
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

class TaskQueue:
    """Asynchronous task queue manager."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.queue = asyncio.Queue()
        
    async def add_task(self, func, *args, **kwargs) -> str:
        """Add new task to queue."""
        task = Task(func, *args, **kwargs)
        self.tasks[task.id] = task
        await self.queue.put(task)
        return task.id
        
    async def process_tasks(self):
        """Process tasks from queue."""
        while True:
            task = await self.queue.get()
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            try:
                task.result = await task.func(*task.args, **task.kwargs)
                task.status = TaskStatus.COMPLETED
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.error(
                    "task_failed",
                    task_id=task.id,
                    error=str(e)
                )
            
            task.completed_at = datetime.utcnow()
            self.queue.task_done()
            
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and details."""
        task = self.tasks.get(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
            
        return {
            "id": task.id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error
        }

task_queue = TaskQueue() 