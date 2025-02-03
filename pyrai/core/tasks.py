from typing import Any, Callable, Dict, Optional, TypeVar, Awaitable
from datetime import datetime
import asyncio
import uuid

from pyrai.core.logging import logger
from pyrai.models.task import Task
from pyrai.db.session import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy.future import select

T = TypeVar('T')

class TaskManager:
    """Background task manager."""

    def __init__(self) -> None:
        """Initialize task manager."""
        self.tasks: Dict[str, asyncio.Task[Any]] = {}
        self.callbacks: Dict[str, Callable[[Any], None]] = {}

    async def create_task(
        self,
        name: str,
        task_type: str,
        params: Dict[str, Any],
        user_id: int,
        model_id: Optional[int] = None,
    ) -> Task:
        """Create a new task."""
        db = SessionLocal()
        try:
            task = Task(
                name=name,
                task_type=task_type,
                params=params,
                user_id=user_id,
                model_id=model_id,
                status="pending"
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            return task
        finally:
            db.close()

    async def start_task(
        self,
        task_id: int,
        coroutine: Callable[[], Awaitable[T]],
        callback: Optional[Callable[[T], None]] = None
    ) -> None:
        """Start a task."""
        task_key = str(uuid.uuid4())
        self.tasks[task_key] = asyncio.create_task(self._run_task(task_id, coroutine))
        if callback:
            self.callbacks[task_key] = callback

    async def _run_task(
        self,
        task_id: int,
        coroutine: Callable[[], Awaitable[T]]
    ) -> None:
        """Run a task and update its status."""
        db = SessionLocal()
        try:
            task = await db.execute(
                select(Task).filter(Task.id == task_id)
            )
            task = task.scalar_one_or_none()
            if not task:
                logger.error(f"Task {task_id} not found")
                return

            task.status = "running"
            task.started_at = datetime.now(datetime.UTC)
            await db.commit()

            try:
                result = await coroutine()
                task.status = "completed"
                task.result = result
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                task.status = "failed"
                task.error = str(e)
            finally:
                task.completed_at = datetime.utcnow()
                await db.commit()

        finally:
            db.close()

    async def cancel_task(self, task_id: int) -> bool:
        """Cancel a running task."""
        for key, task in self.tasks.items():
            if str(task_id) in key:
                task.cancel()
                return True
        return False

    def get_task_progress(self, task_id: int) -> Optional[int]:
        """Get task progress."""
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            return task.progress if task else None
        finally:
            db.close()

    def update_task_progress(self, task_id: int, progress: int) -> None:
        """Update task progress."""
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.progress = progress
                db.commit()
        finally:
            db.close()

task_manager = TaskManager() 