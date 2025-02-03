from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    """Base user schema."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False

class UserCreate(UserBase):
    """User creation schema."""
    email: EmailStr
    password: str
    full_name: str

class UserUpdate(UserBase):
    """User update schema."""
    password: Optional[str] = None

class UserResponse(UserBase):
    """User response schema."""
    id: int
    email: EmailStr
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""
        from_attributes = True 