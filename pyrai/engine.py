import os
from typing import Any, Dict, List, Optional, Union
import torch
import tensorflow as tf
from pathlib import Path
from .backends import ModelBackend
from .config import Config
from .deployment import Deployment
from .monitoring import Monitor
from .metrics import MetricCollection
from .logging import Logger

logger = Logger.get_logger("engine")

class Engine:
    """Main PyRAI engine class."""
    
    def __init__(
        self,
        backend: ModelBackend,
        config: Config
    ):
        """Initialize PyRAI engine.
        
        Args:
            backend: Model backend (PyTorch/TensorFlow)
            config: Engine configuration
        """
        self.backend = backend
        self.config = config
        self.device = config.get("model.device", "cpu")
        self.monitor = Monitor()
        
    def train(
        self,
        model: Any,
        train_data: Any,
        val_data: Optional[Any] = None,
        metrics: Optional[MetricCollection] = None,
        callbacks: Optional[list] = None
    ) -> Dict[str, list]:
        """Train model.
        
        Args:
            model: Model to train
            train_data: Training data
            val_data: Validation data
            metrics: Training metrics
            callbacks: Training callbacks
            
        Returns:
            Training history
        """
        logger.info("Starting model training")
        
        # Move model to device
        model = self.backend.to_device(model, self.device)
        
        # Get training parameters
        batch_size = self.config.get("training.batch_size", 32)
        epochs = self.config.get("training.epochs", 10)
        
        # Setup optimizer
        optimizer = self.backend.create_optimizer(
            model,
            self.config.get("training.optimizer", {})
        )
        
        # Setup scheduler
        scheduler = self.backend.create_scheduler(
            optimizer,
            self.config.get("training.scheduler", {})
        )
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        if metrics:
            for name in metrics.names:
                history[f"train_{name}"] = []
                history[f"val_{name}"] = []
        
        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch(
                model,
                train_data,
                optimizer,
                metrics,
                batch_size
            )
            
            for k, v in train_metrics.items():
                history[f"train_{k}"].append(v)
            
            # Validation
            if val_data is not None:
                val_metrics = self._validate(
                    model,
                    val_data,
                    metrics,
                    batch_size
                )
                
                for k, v in val_metrics.items():
                    history[f"val_{k}"].append(v)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}")
            for k, v in train_metrics.items():
                logger.info(f"train_{k}: {v:.4f}")
            if val_data is not None:
                for k, v in val_metrics.items():
                    logger.info(f"val_{k}: {v:.4f}")
        
        return history
    
    def deploy(
        self,
        model: Any,
        model_id: str,
        version: Optional[str] = None
    ) -> Deployment:
        """Deploy model.
        
        Args:
            model: Model to deploy
            model_id: Model identifier
            version: Model version
            
        Returns:
            Model deployment
        """
        logger.info(f"Deploying model: {model_id}")
        
        # Save model
        save_dir = self.config.get("model.save_dir", "models")
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(
            save_dir,
            f"{model_id}_{version}.pt" if version else f"{model_id}.pt"
        )
        
        self.backend.save_model(model, model_path)
        
        # Create deployment
        return Deployment(
            model=model,
            model_id=model_id,
            version=version,
            backend=self.backend,
            config=self.config
        )
    
    async def predict(
        self,
        inputs: Any,
        model_id: str,
        version_id: Optional[str] = None
    ) -> Any:
        """Make predictions.
        
        Args:
            inputs: Input data
            model_id: Model identifier
            version_id: Model version
            
        Returns:
            Model predictions
        """
        # Get deployment
        deployment = self.get_deployment(model_id, version_id)
        
        # Make prediction
        predictions = await deployment.predict(inputs)
        
        return predictions
    
    def get_status(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get model status.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Status information
        """
        deployment = self.get_deployment(model_id, version)
        return deployment.get_status()
    
    def get_deployment(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Deployment:
        """Get deployment object.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Deployment object
        """
        # Get latest version if not specified
        if not version:
            version = self._get_latest_version(model_id)
        
        deployment_id = f"{model_id}-{version}"
        return Deployment.get(deployment_id)
    
    def _get_latest_version(self, model_id: str) -> str:
        """Get latest model version."""
        # TODO: Implement version management
        return "latest"
    
    def _train_epoch(
        self,
        model: Any,
        data: Any,
        optimizer: Any,
        metrics: Optional[MetricCollection],
        batch_size: int
    ) -> Dict[str, float]:
        """Train single epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        if metrics:
            metrics.reset()
        
        for batch in self.backend.create_dataloader(data, batch_size):
            # Forward pass
            loss = self.backend.train_step(model, batch, optimizer)
            total_loss += loss
            num_batches += 1
            
            # Update metrics
            if metrics:
                metrics.update(
                    self.backend.get_predictions(model, batch),
                    self.backend.get_targets(batch)
                )
        
        # Compute metrics
        results = {"loss": total_loss / num_batches}
        if metrics:
            results.update(metrics.compute())
            
        return results
    
    def _validate(
        self,
        model: Any,
        data: Any,
        metrics: Optional[MetricCollection],
        batch_size: int
    ) -> Dict[str, float]:
        """Validate model."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        if metrics:
            metrics.reset()
        
        for batch in self.backend.create_dataloader(data, batch_size):
            # Forward pass
            loss = self.backend.validation_step(model, batch)
            total_loss += loss
            num_batches += 1
            
            # Update metrics
            if metrics:
                metrics.update(
                    self.backend.get_predictions(model, batch),
                    self.backend.get_targets(batch)
                )
        
        # Compute metrics
        results = {"loss": total_loss / num_batches}
        if metrics:
            results.update(metrics.compute())
            
        return results 