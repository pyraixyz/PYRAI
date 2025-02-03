from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy import select

from pyrai.models.model import Model
from pyrai.models.user import User
from pyrai.api.schemas.model import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
    ModelVersion
)
from pyrai.api.auth import get_current_user
from pyrai.core.state import app_state
from pyrai.db.session import AsyncSession, get_session

router = APIRouter()

@router.get("", response_model=List[ModelResponse])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """List all models."""
    models = await session.scalars(
        select(Model)
        .where(Model.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
    )
    return models.all()

@router.post("", response_model=ModelResponse)
async def create_model(
    model_in: ModelCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Create new model."""
    model = Model(
        **model_in.model_dump(),
        user_id=current_user.id
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model

@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Get model by ID."""
    model = await session.get(Model, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    if model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return model

@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_in: ModelUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Update model."""
    model = await session.get(Model, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    if model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    update_data = model_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)
    
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model

@router.post("/{model_id}/versions", response_model=ModelVersion)
async def create_version(
    model_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Create new model version."""
    model = await session.get(Model, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    if model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Save model file
    file_path = app_state.get_model_path(model_id, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create version
    version = ModelVersion(
        model_id=model_id,
        version=model.next_version,
        file_path=str(file_path)
    )
    session.add(version)
    await session.commit()
    await session.refresh(version)
    return version 