from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict

DataT = TypeVar("DataT")

class Response(BaseModel, Generic[DataT]):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[DataT] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation successful",
                "data": None
            }
        }
    )

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": "Not Found",
                "detail": "Resource not found",
                "code": "NOT_FOUND"
            }
        }
    )

class PaginatedResponse(Response[DataT]):
    """Paginated response model."""
    page: int
    size: int
    total: int
    pages: int
    has_next: bool
    has_prev: bool
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": [],
                "page": 1,
                "size": 10,
                "total": 0,
                "pages": 0,
                "has_next": False,
                "has_prev": False
            }
        }
    )

def create_response(
    data: Optional[Any] = None,
    message: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Create a success response."""
    return {
        "success": True,
        "message": message,
        "data": data,
        **kwargs
    }

def create_error(
    error: str,
    detail: Optional[str] = None,
    code: Optional[str] = None
) -> Dict[str, Any]:
    """Create an error response."""
    return {
        "success": False,
        "error": error,
        "detail": detail,
        "code": code
    } 