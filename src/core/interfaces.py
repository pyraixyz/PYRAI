"""
Core interfaces for PYRAI.
Defines the base interfaces that all core components must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ModelInterface(ABC):
    """Base interface for AI model management"""
    
    @abstractmethod
    async def load(self, model_id: str) -> Any:
        """Load a model from storage"""
        pass
    
    @abstractmethod
    async def save(self, model_id: str, model: Any) -> bool:
        """Save a model to storage"""
        pass
    
    @abstractmethod
    async def delete(self, model_id: str) -> bool:
        """Delete a model from storage"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """List all available models"""
        pass


class ResourceInterface(ABC):
    """Base interface for resource management"""
    
    @abstractmethod
    async def allocate(self, requirements: Dict[str, Any]) -> str:
        """Allocate resources based on requirements"""
        pass
    
    @abstractmethod
    async def deallocate(self, resource_id: str) -> bool:
        """Deallocate resources"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        pass


class TrainingInterface(ABC):
    """Base interface for training coordination"""
    
    @abstractmethod
    async def start_training(self, config: Dict[str, Any]) -> str:
        """Start a training job"""
        pass
    
    @abstractmethod
    async def stop_training(self, job_id: str) -> bool:
        """Stop a training job"""
        pass
    
    @abstractmethod
    async def get_progress(self, job_id: str) -> Dict[str, Any]:
        """Get training progress"""
        pass


class StorageInterface(ABC):
    """Base interface for storage management"""
    
    @abstractmethod
    async def store(self, key: str, data: Any) -> bool:
        """Store data"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data"""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with given prefix"""
        pass 