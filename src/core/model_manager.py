"""
Model Manager implementation for PYRAI.
Handles AI model lifecycle management including loading, saving, and versioning.
"""

import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import torch
import tensorflow as tf

from .interfaces import ModelInterface
from .exceptions import ModelError, ModelNotFoundError
from .storage_manager import StorageManager
from .config import config


class ModelManager(ModelInterface):
    """
    Model Manager implementation that handles AI model lifecycle.
    Supports PyTorch and TensorFlow models with versioning capabilities.
    """

    def __init__(self, storage_manager: Optional[StorageManager] = None):
        """
        Initialize the model manager.
        
        Args:
            storage_manager: Optional storage manager instance. Creates new one if not provided.
        """
        self.storage = storage_manager or StorageManager()
        self._setup_metadata()

    async def _setup_metadata(self) -> None:
        """Initialize model metadata storage"""
        try:
            metadata = await self.storage.retrieve("model_metadata")
            if metadata is None:
                await self.storage.store("model_metadata", {})
        except Exception as e:
            raise ModelError(f"Failed to setup model metadata: {str(e)}")

    async def _get_metadata(self) -> Dict[str, Any]:
        """Retrieve model metadata"""
        try:
            metadata = await self.storage.retrieve("model_metadata")
            return metadata or {}
        except Exception as e:
            raise ModelError(f"Failed to get metadata: {str(e)}")

    async def _update_metadata(self, model_id: str, info: Dict[str, Any]) -> None:
        """Update metadata for a specific model"""
        try:
            metadata = await self._get_metadata()
            if model_id not in metadata:
                metadata[model_id] = {"versions": []}
            
            version_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "version": len(metadata[model_id]["versions"]) + 1,
                **info
            }
            
            metadata[model_id]["versions"].append(version_info)
            await self.storage.store("model_metadata", metadata)
        except Exception as e:
            raise ModelError(f"Failed to update metadata: {str(e)}")

    def _get_model_key(self, model_id: str, version: Optional[int] = None) -> str:
        """Generate storage key for model"""
        if version:
            return f"models/{model_id}/v{version}"
        return f"models/{model_id}/latest"

    async def load(self, model_id: str, version: Optional[int] = None) -> Any:
        """
        Load a model from storage.
        
        Args:
            model_id: The model identifier
            version: Optional specific version to load
            
        Returns:
            The loaded model
            
        Raises:
            ModelNotFoundError: If model is not found
            ModelError: If loading fails
        """
        try:
            key = self._get_model_key(model_id, version)
            model_data = await self.storage.retrieve(key)
            
            if model_data is None:
                raise ModelNotFoundError(f"Model {model_id} not found")
            
            model_type = model_data.get("type")
            weights = model_data.get("weights")
            
            if model_type == "pytorch":
                model = torch.load(weights)
            elif model_type == "tensorflow":
                model = tf.keras.models.load_model(weights)
            else:
                raise ModelError(f"Unsupported model type: {model_type}")
            
            return model
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")

    async def save(self, model_id: str, model: Any) -> bool:
        """
        Save a model to storage.
        
        Args:
            model_id: The model identifier
            model: The model to save
            
        Returns:
            bool: True if save was successful
            
        Raises:
            ModelError: If saving fails
        """
        try:
            # Determine model type
            if isinstance(model, torch.nn.Module):
                model_type = "pytorch"
                weights_path = f"temp/{model_id}_weights.pt"
                torch.save(model, weights_path)
            elif isinstance(model, tf.keras.Model):
                model_type = "tensorflow"
                weights_path = f"temp/{model_id}_weights.h5"
                model.save(weights_path)
            else:
                raise ModelError("Unsupported model type")
            
            # Store model data
            model_data = {
                "type": model_type,
                "weights": weights_path,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Save to storage
            key = self._get_model_key(model_id)
            await self.storage.store(key, model_data)
            
            # Update metadata
            await self._update_metadata(model_id, {
                "type": model_type,
                "size": os.path.getsize(weights_path)
            })
            
            # Cleanup temporary files
            os.remove(weights_path)
            
            return True
            
        except Exception as e:
            raise ModelError(f"Failed to save model: {str(e)}")

    async def delete(self, model_id: str, version: Optional[int] = None) -> bool:
        """
        Delete a model from storage.
        
        Args:
            model_id: The model identifier
            version: Optional specific version to delete
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            ModelError: If deletion fails
        """
        try:
            key = self._get_model_key(model_id, version)
            if not await self.storage.delete(key):
                raise ModelNotFoundError(f"Model {model_id} not found")
            
            if version is None:
                # If no version specified, delete all versions
                metadata = await self._get_metadata()
                if model_id in metadata:
                    del metadata[model_id]
                    await self.storage.store("model_metadata", metadata)
            
            return True
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelError(f"Failed to delete model: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their metadata.
        
        Returns:
            List of model information dictionaries
            
        Raises:
            ModelError: If listing fails
        """
        try:
            metadata = await self._get_metadata()
            return [
                {
                    "id": model_id,
                    "versions": info["versions"],
                    "latest_version": len(info["versions"])
                }
                for model_id, info in metadata.items()
            ]
        except Exception as e:
            raise ModelError(f"Failed to list models: {str(e)}")

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Model information dictionary or None if not found
            
        Raises:
            ModelError: If retrieval fails
        """
        try:
            metadata = await self._get_metadata()
            return metadata.get(model_id) 