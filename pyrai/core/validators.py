import re
from typing import Optional, Union
from datetime import datetime
import uuid

from pydantic import BaseModel, EmailStr, validator, constr

class UserValidator(BaseModel):
    """User data validator."""
    email: EmailStr
    password: constr(min_length=8, max_length=64)  # type: ignore
    full_name: Optional[str] = None

    @validator("password")
    def password_strength(cls, v: str) -> str:
        """Validate password strength."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character")
        return v

    @validator("full_name")
    def validate_full_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate full name."""
        if v is not None:
            if len(v.strip()) < 2:
                raise ValueError("Full name must be at least 2 characters long")
            if not re.match(r"^[a-zA-Z\s-]+$", v):
                raise ValueError("Full name can only contain letters, spaces, and hyphens")
        return v

class ModelValidator(BaseModel):
    """Model metadata validator."""
    name: constr(min_length=1, max_length=64)  # type: ignore
    version: constr(regex=r"^\d+\.\d+\.\d+$")  # type: ignore
    description: Optional[str] = None
    model_type: str
    framework: str
    config: dict
    metrics: Optional[dict] = None

    @validator("model_type")
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        allowed_types = {"classification", "regression", "generation", "embedding"}
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of: {allowed_types}")
        return v

    @validator("framework")
    def validate_framework(cls, v: str) -> str:
        """Validate framework."""
        allowed_frameworks = {"pytorch", "tensorflow", "scikit-learn", "xgboost"}
        if v not in allowed_frameworks:
            raise ValueError(f"Framework must be one of: {allowed_frameworks}")
        return v

class TaskValidator(BaseModel):
    """Task data validator."""
    name: str
    task_type: str
    params: dict
    user_id: int
    model_id: Optional[int] = None

    @validator("task_type")
    def validate_task_type(cls, v: str) -> str:
        """Validate task type."""
        allowed_types = {"training", "inference", "evaluation", "optimization"}
        if v not in allowed_types:
            raise ValueError(f"Task type must be one of: {allowed_types}")
        return v

def validate_uuid(value: Union[str, uuid.UUID]) -> uuid.UUID:
    """Validate UUID."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(value)
    except ValueError:
        raise ValueError("Invalid UUID format")

def validate_datetime(value: Union[str, datetime]) -> datetime:
    """Validate datetime."""
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("Invalid datetime format") 