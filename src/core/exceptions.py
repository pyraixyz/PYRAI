"""
Core exceptions for PYRAI.
Defines all custom exceptions used throughout the core engine.
"""

class PYRAIException(Exception):
    """Base exception for all PYRAI errors"""
    pass


class ModelError(PYRAIException):
    """Base class for model-related errors"""
    pass


class ResourceError(PYRAIException):
    """Base class for resource-related errors"""
    pass


class TrainingError(PYRAIException):
    """Base class for training-related errors"""
    pass


class StorageError(PYRAIException):
    """Base class for storage-related errors"""
    pass


class ConfigError(PYRAIException):
    """Configuration-related errors"""
    pass


class ValidationError(PYRAIException):
    """Validation-related errors"""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a model cannot be found"""
    pass


class ResourceExhaustedError(ResourceError):
    """Raised when system resources are exhausted"""
    pass


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted unexpectedly"""
    pass


class StorageQuotaExceededError(StorageError):
    """Raised when storage quota is exceeded"""
    pass 