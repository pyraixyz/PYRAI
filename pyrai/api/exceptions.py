from typing import Any, Dict, Optional
from fastapi import HTTPException, status

class APIError(HTTPException):
    """Base API error."""
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class NotFoundError(APIError):
    """Resource not found error."""
    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class ValidationError(APIError):
    """Validation error."""
    def __init__(self, detail: str = "Validation error") -> None:
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

class AuthenticationError(APIError):
    """Authentication error."""
    def __init__(self, detail: str = "Could not validate credentials") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class PermissionError(APIError):
    """Permission error."""
    def __init__(self, detail: str = "Not enough permissions") -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

class RateLimitError(APIError):
    """Rate limit error."""
    def __init__(self, detail: str = "Too many requests") -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": "60"},
        )

class ServerError(APIError):
    """Server error."""
    def __init__(self, detail: str = "Internal server error") -> None:
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail) 