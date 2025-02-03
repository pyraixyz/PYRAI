from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import ModelBackend

class PyTorchBackend(ModelBackend):
    """PyTorch model backend implementation."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize PyTorch backend.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to_device(
        self,
        model: nn.Module,
        device: str
    ) -> nn.Module:
        """Move model to device."""
        return model.to(device)
    
    def create_optimizer(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer."""
        name = config.get("name", "adam").lower()
        
        if name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config.get("learning_rate", 0.001),
                weight_decay=config.get("weight_decay", 0)
            )
        elif name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.get("learning_rate", 0.001),
                weight_decay=config.get("weight_decay", 0.01)
            )
        elif name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.get("learning_rate", 0.01),
                momentum=config.get("momentum", 0),
                weight_decay=config.get("weight_decay", 0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any]
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if not config:
            return None
            
        name = config.get("name", "").lower()
        
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["T_max"],
                eta_min=config.get("eta_min", 0)
            )
        elif name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get("mode", "min"),
                factor=config.get("factor", 0.1),
                patience=config.get("patience", 10)
            )
        elif name == "one_cycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config["max_lr"],
                epochs=config["epochs"],
                steps_per_epoch=config["steps_per_epoch"]
            )
        else:
            return None
    
    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
    
    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Perform training step."""
        optimizer.zero_grad()
        
        # Forward pass
        inputs = self.get_inputs(batch)
        targets = self.get_targets(batch)
        outputs = model(inputs)
        loss = self._compute_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validation_step(
        self,
        model: nn.Module,
        batch: Any
    ) -> float:
        """Perform validation step."""
        with torch.no_grad():
            inputs = self.get_inputs(batch)
            targets = self.get_targets(batch)
            outputs = model(inputs)
            loss = self._compute_loss(outputs, targets)
            
        return loss.item()
    
    def get_predictions(
        self,
        model: nn.Module,
        batch: Any
    ) -> torch.Tensor:
        """Get model predictions."""
        with torch.no_grad():
            inputs = self.get_inputs(batch)
            return model(inputs)
    
    def get_targets(self, batch: Any) -> torch.Tensor:
        """Get target values."""
        if isinstance(batch, (tuple, list)):
            return batch[1]
        return batch.targets
    
    def get_inputs(self, batch: Any) -> torch.Tensor:
        """Get input values."""
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch.inputs
    
    def save_model(
        self,
        model: nn.Module,
        path: str
    ) -> None:
        """Save PyTorch model.
        
        Args:
            model: PyTorch model to save
            path: Save path
        """
        torch.save(model.state_dict(), path)
    
    def load_model(
        self,
        path: str,
        device: Optional[str] = None
    ) -> nn.Module:
        """Load PyTorch model.
        
        Args:
            path: Path to saved model
            device: Target device
            
        Returns:
            Loaded PyTorch model
        """
        state_dict = torch.load(path, map_location=device)
        model = self._create_model()
        model.load_state_dict(state_dict)
        return model.to(self.device)
    
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss value."""
        if outputs.shape[-1] == 1:  # Binary classification
            return nn.BCEWithLogitsLoss()(outputs, targets.float())
        else:  # Multi-class classification
            return nn.CrossEntropyLoss()(outputs, targets)
    
    def _create_model(self) -> nn.Module:
        """Create model instance."""
        raise NotImplementedError(
            "Subclasses must implement _create_model()"
        )
    
    def prepare_training(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ) -> None:
        """Prepare PyTorch model for training.
        
        Args:
            model: PyTorch model
            config: Training configuration
        """
        model.to(self.device)
        model.train()
        
        # Setup optimizer
        optimizer_config = config.get("optimizer", {})
        optimizer_type = optimizer_config.pop("type", "adam")
        
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                **optimizer_config
            )
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                **optimizer_config
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup loss function
        loss_type = config.get("loss", "cross_entropy")
        if loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_type}")
    
    def train_epoch(
        self,
        model: nn.Module,
        train_data: Any,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            model: PyTorch model
            train_data: Training data
            config: Training configuration
            
        Returns:
            Training metrics
        """
        model.train()
        total_loss = 0
        num_batches = 0
        
        dataloader = self._get_dataloader(train_data, config)
        
        for batch in dataloader:
            inputs, targets = self._prepare_batch(batch)
            
            # Forward pass
            outputs = model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches
        }
    
    def validate(
        self,
        model: nn.Module,
        val_data: Any,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate PyTorch model.
        
        Args:
            model: PyTorch model
            val_data: Validation data
            config: Validation configuration
            
        Returns:
            Validation metrics
        """
        model.eval()
        total_loss = 0
        num_batches = 0
        
        dataloader = self._get_dataloader(val_data, config)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self._prepare_batch(batch)
                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches
        }
    
    def _prepare_inputs(self, inputs: Any) -> torch.Tensor:
        """Prepare inputs for model."""
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        elif isinstance(inputs, (list, tuple)):
            return torch.tensor(inputs).to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
    
    def _prepare_batch(self, batch: Any) -> tuple:
        """Prepare batch for training/validation."""
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch
        else:
            inputs = batch
            targets = None
        
        inputs = self._prepare_inputs(inputs)
        if targets is not None:
            targets = self._prepare_inputs(targets)
        
        return inputs, targets
    
    def _get_dataloader(
        self,
        data: Any,
        config: Dict[str, Any]
    ) -> DataLoader:
        """Get DataLoader for training/validation."""
        if isinstance(data, DataLoader):
            return data
        
        batch_size = config.get("batch_size", 32)
        shuffle = config.get("shuffle", True)
        
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle
        ) 