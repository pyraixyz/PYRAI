"""
Storage Manager implementation for PYRAI.
Provides a unified interface for data storage operations across different storage backends.
"""

import os
import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import aiofiles
import json

from .interfaces import StorageInterface
from .exceptions import StorageError, StorageQuotaExceededError
from .config import config


class StorageManager(StorageInterface):
    """
    Storage Manager implementation that handles data persistence across different storage backends.
    Currently supports local file system storage with caching capabilities.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the storage manager.
        
        Args:
            storage_path: Optional custom storage path. If not provided, uses the config value.
        """
        self.storage_path = Path(storage_path or config.model_storage_path)
        self.cache: Dict[str, Any] = {}
        self.cache_size = 0
        self._setup_storage()

    def _setup_storage(self) -> None:
        """Setup the storage directory structure"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            (self.storage_path / "temp").mkdir(exist_ok=True)
            (self.storage_path / "cache").mkdir(exist_ok=True)
        except Exception as e:
            raise StorageError(f"Failed to setup storage directories: {str(e)}")

    def _get_file_path(self, key: str) -> Path:
        """
        Get the file path for a given key.
        
        Args:
            key: The storage key
            
        Returns:
            Path object for the file location
        """
        # Create a hash of the key to use as filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.storage_path / key_hash[:2] / key_hash[2:]

    async def _manage_cache(self, key: str, data: Any) -> None:
        """
        Manage the cache size and contents.
        
        Args:
            key: The cache key
            data: The data to cache
        """
        data_size = len(str(data).encode())
        
        # If adding this item would exceed cache size, remove old items
        while self.cache_size + data_size > config.cache_size:
            if not self.cache:
                raise StorageQuotaExceededError("Cache quota exceeded and no items to remove")
            # Remove the first item (FIFO)
            removed_key = next(iter(self.cache))
            removed_data = self.cache.pop(removed_key)
            self.cache_size -= len(str(removed_data).encode())

        self.cache[key] = data
        self.cache_size += data_size

    async def store(self, key: str, data: Any) -> bool:
        """
        Store data using the specified key.
        
        Args:
            key: The storage key
            data: The data to store
            
        Returns:
            bool: True if storage was successful
            
        Raises:
            StorageError: If storage operation fails
        """
        try:
            file_path = self._get_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert data to JSON string if it's not already a string
            if not isinstance(data, str):
                data = json.dumps(data)

            async with aiofiles.open(file_path, mode='w') as f:
                await f.write(data)

            # Update cache
            await self._manage_cache(key, data)
            return True

        except Exception as e:
            raise StorageError(f"Failed to store data: {str(e)}")

    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve data for the specified key.
        
        Args:
            key: The storage key
            
        Returns:
            The stored data or None if not found
            
        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            # Check cache first
            if key in self.cache:
                return self.cache[key]

            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None

            async with aiofiles.open(file_path, mode='r') as f:
                data = await f.read()

            # Try to parse as JSON
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON

            # Update cache
            await self._manage_cache(key, data)
            return data

        except Exception as e:
            raise StorageError(f"Failed to retrieve data: {str(e)}")

    async def delete(self, key: str) -> bool:
        """
        Delete data for the specified key.
        
        Args:
            key: The storage key
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            StorageError: If deletion operation fails
        """
        try:
            # Remove from cache if present
            if key in self.cache:
                data = self.cache.pop(key)
                self.cache_size -= len(str(data).encode())

            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                
                # Remove parent directory if empty
                if not any(file_path.parent.iterdir()):
                    file_path.parent.rmdir()
                    
            return True

        except Exception as e:
            raise StorageError(f"Failed to delete data: {str(e)}")

    async def list_keys(self, prefix: str = "") -> List[str]:
        """
        List all storage keys with the given prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of matching keys
            
        Raises:
            StorageError: If listing operation fails
        """
        try:
            keys = []
            for path in self.storage_path.rglob("*"):
                if path.is_file() and not path.name.startswith('.'):
                    key = path.stem
                    if key.startswith(prefix):
                        keys.append(key)
            return keys

        except Exception as e:
            raise StorageError(f"Failed to list keys: {str(e)}")

    async def clear_cache(self) -> None:
        """Clear the entire cache"""
        self.cache.clear()
        self.cache_size = 0 