"""SDK exceptions."""

class PyRAIError(Exception):
    """Base exception for PyRAI SDK."""
    pass

class APIError(PyRAIError):
    """API request error."""
    pass

class ValidationError(PyRAIError):
    """Data validation error."""
    pass

class AuthenticationError(PyRAIError):
    """Authentication error."""
    pass

class TaskError(PyRAIError):
    """Task processing error."""
    pass 