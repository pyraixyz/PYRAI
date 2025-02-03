from typing import Any, Dict, Optional
from http import HTTPStatus

class PyRaiError(Exception):
    """Base error class for PyRai."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error": self.code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }

class ValidationError(PyRaiError):
    """Validation error."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            details=details
        )

class AuthenticationError(PyRaiError):
    """Authentication error."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=HTTPStatus.UNAUTHORIZED,
            details=details
        )

class AuthorizationError(PyRaiError):
    """Authorization error."""

    def __init__(
        self,
        message: str = "Permission denied",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=HTTPStatus.FORBIDDEN,
            details=details
        )

class NotFoundError(PyRaiError):
    """Resource not found error."""

    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=HTTPStatus.NOT_FOUND,
            details=details
        )

class RateLimitError(PyRaiError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            details={"retry_after": retry_after, **(details or {})}
        ) 