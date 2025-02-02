"""
Tests for the Storage Manager implementation.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from src.core.storage_manager import StorageManager
from src.core.exceptions import StorageError, StorageQuotaExceededError


@pytest.fixture
async def storage_manager():
    """Create a temporary storage manager for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = StorageManager(storage_path=temp_dir)
        yield manager


@pytest.mark.asyncio
async def test_store_and_retrieve(storage_manager):
    """Test basic store and retrieve operations"""
    # Test with string data
    key = "test_key"
    data = "test_data"
    assert await storage_manager.store(key, data)
    retrieved = await storage_manager.retrieve(key)
    assert retrieved == data

    # Test with dictionary data
    key2 = "test_dict"
    data2 = {"name": "test", "value": 123}
    assert await storage_manager.store(key2, data2)
    retrieved2 = await storage_manager.retrieve(key2)
    assert retrieved2 == data2


@pytest.mark.asyncio
async def test_delete(storage_manager):
    """Test delete operation"""
    key = "test_delete"
    data = "test_data"
    
    # Store and verify data exists
    await storage_manager.store(key, data)
    assert await storage_manager.retrieve(key) == data
    
    # Delete and verify data is gone
    assert await storage_manager.delete(key)
    assert await storage_manager.retrieve(key) is None


@pytest.mark.asyncio
async def test_list_keys(storage_manager):
    """Test listing keys with prefix"""
    # Store multiple items
    test_data = {
        "test1": "data1",
        "test2": "data2",
        "other": "data3"
    }
    
    for key, data in test_data.items():
        await storage_manager.store(key, data)
    
    # List all keys
    all_keys = await storage_manager.list_keys()
    assert len(all_keys) == 3
    
    # List keys with prefix
    test_keys = await storage_manager.list_keys("test")
    assert len(test_keys) == 2


@pytest.mark.asyncio
async def test_cache_management(storage_manager):
    """Test cache management and quota enforcement"""
    # Override cache size to test quota management
    storage_manager.cache_size = 0
    storage_manager.config.cache_size = 100  # Small cache size for testing
    
    # Store data that should trigger cache management
    key = "test_cache"
    data = "x" * 200  # Data larger than cache size
    
    with pytest.raises(StorageQuotaExceededError):
        await storage_manager.store(key, data)


@pytest.mark.asyncio
async def test_concurrent_operations(storage_manager):
    """Test concurrent storage operations"""
    async def store_item(key: str, data: str):
        await storage_manager.store(key, data)
        return await storage_manager.retrieve(key)
    
    # Create multiple concurrent operations
    tasks = [
        store_item(f"key_{i}", f"data_{i}")
        for i in range(10)
    ]
    
    # Run concurrently and verify results
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    assert all(f"data_{i}" == results[i] for i in range(10)) 