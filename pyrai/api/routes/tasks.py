from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from pyrai.models.task import Task
from pyrai.models.user import User
from pyrai.api.schemas.task import TaskCreate, TaskUpdate, TaskResponse
from pyrai.api.auth import get_current_user
from pyrai.core.tasks import task_manager
from pyrai.db.session import AsyncSession, get_session

router = APIRouter()

@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """List all tasks."""
    tasks = await session.scalars(
        select(Task)
        .where(Task.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
    )
    return tasks.all()

@router.post("", response_model=TaskResponse)
async def create_task(
    task_in: TaskCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Create new task."""
    task = await task_manager.create_task(
        name=task_in.name,
        task_type=task_in.task_type,
        params=task_in.params,
        user_id=current_user.id,
        model_id=task_in.model_id
    )
    return task

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Get task by ID."""
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    if task.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return task

@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Cancel task."""
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    if task.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    success = await task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task cannot be cancelled"
        )
    return {"status": "cancelled"} 