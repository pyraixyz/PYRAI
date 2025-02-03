from typing import Any, Dict, List, Optional, Union, Callable
import torch
import torch.distributed as dist
import tensorflow as tf
import horovod.torch as hvd
import horovod.tensorflow as hvd_tf
from enum import Enum

class DistributedBackend(Enum):
    """Distributed training backend."""
    PYTORCH_DDP = "pytorch_ddp"
    TENSORFLOW_MIRROR = "tensorflow_mirror"
    HOROVOD = "horovod"

class DistributedTrainer:
    """Distributed training manager."""
    
    def __init__(
        self,
        backend: DistributedBackend,
        num_workers: int = 1,
        use_gpu: bool = True
    ):
        """Initialize distributed trainer.
        
        Args:
            backend: Distributed training backend
            num_workers: Number of workers
            use_gpu: Whether to use GPU
        """
        self.backend = backend
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self._setup_distributed()
        
    def _setup_distributed(self):
        """Setup distributed training environment."""
        if self.backend == DistributedBackend.PYTORCH_DDP:
            dist.init_process_group(backend="nccl" if self.use_gpu else "gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
        elif self.backend == DistributedBackend.TENSORFLOW_MIRROR:
            self.strategy = tf.distribute.MirroredStrategy()
            
        elif self.backend == DistributedBackend.HOROVOD:
            hvd.init()
            self.rank = hvd.rank()
            self.world_size = hvd.size()
            
    def prepare_model(
        self,
        model: Union[torch.nn.Module, tf.keras.Model]
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Prepare model for distributed training."""
        if isinstance(model, torch.nn.Module):
            return self._prepare_torch_model(model)
        else:
            return self._prepare_tf_model(model)
            
    def _prepare_torch_model(
        self,
        model: torch.nn.Module
    ) -> torch.nn.Module:
        """Prepare PyTorch model."""
        if self.use_gpu:
            if self.backend == DistributedBackend.PYTORCH_DDP:
                device = torch.device(f"cuda:{self.rank}")
                model = model.to(device)
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.rank]
                )
            elif self.backend == DistributedBackend.HOROVOD:
                device = torch.device(f"cuda:{hvd.local_rank()}")
                model = model.to(device)
                
        return model
        
    def _prepare_tf_model(
        self,
        model: tf.keras.Model
    ) -> tf.keras.Model:
        """Prepare TensorFlow model."""
        if self.backend == DistributedBackend.TENSORFLOW_MIRROR:
            with self.strategy.scope():
                model = tf.keras.models.clone_model(model)
                model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
        elif self.backend == DistributedBackend.HOROVOD:
            optimizer = hvd_tf.DistributedOptimizer(model.optimizer)
            model.compile(
                optimizer=optimizer,
                loss=model.loss,
                metrics=model.metrics
            )
            
        return model
        
    def prepare_data(
        self,
        dataset: Any,
        batch_size: int
    ) -> Any:
        """Prepare dataset for distributed training."""
        if isinstance(dataset, torch.utils.data.Dataset):
            return self._prepare_torch_data(dataset, batch_size)
        else:
            return self._prepare_tf_data(dataset, batch_size)
            
    def _prepare_torch_data(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int
    ) -> torch.utils.data.DataLoader:
        """Prepare PyTorch dataset."""
        if self.backend == DistributedBackend.PYTORCH_DDP:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler
            )
        elif self.backend == DistributedBackend.HOROVOD:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=hvd.size(),
                rank=hvd.rank()
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler
            )
            
        return dataloader
        
    def _prepare_tf_data(
        self,
        dataset: tf.data.Dataset,
        batch_size: int
    ) -> tf.data.Dataset:
        """Prepare TensorFlow dataset."""
        if self.backend == DistributedBackend.TENSORFLOW_MIRROR:
            return self.strategy.experimental_distribute_dataset(
                dataset.batch(batch_size)
            )
        elif self.backend == DistributedBackend.HOROVOD:
            dataset = dataset.shard(hvd.size(), hvd.rank())
            return dataset.batch(batch_size)
            
        return dataset.batch(batch_size)

    def train(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List[Any]] = None
    ) -> Union[torch.nn.Module, tf.keras.Model]:
        """Train model in distributed mode.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: Training callbacks
            
        Returns:
            Trained model
        """
        # Prepare model and data
        model = self.prepare_model(model)
        train_data = self.prepare_data(train_dataset, batch_size)
        if val_dataset is not None:
            val_data = self.prepare_data(val_dataset, batch_size)
        
        if isinstance(model, torch.nn.Module):
            return self._train_torch_model(
                model,
                train_data,
                val_data if val_dataset is not None else None,
                epochs,
                callbacks
            )
        else:
            return self._train_tf_model(
                model,
                train_data,
                val_data if val_dataset is not None else None,
                epochs,
                callbacks
            )
            
    def evaluate(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        test_dataset: Any,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Evaluate model in distributed mode."""
        model = self.prepare_model(model)
        test_data = self.prepare_data(test_dataset, batch_size)
        
        if isinstance(model, torch.nn.Module):
            return self._evaluate_torch_model(model, test_data)
        else:
            return self._evaluate_tf_model(model, test_data)
            
    def _train_torch_model(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader],
        epochs: int,
        callbacks: Optional[List[Any]]
    ) -> torch.nn.Module:
        """Train PyTorch model."""
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters())
        
        # Initialize callbacks
        if callbacks is None:
            callbacks = []
            
        # Training loop with fault tolerance
        for epoch in range(epochs):
            try:
                model.train()
                total_loss = 0.0
                
                for batch_idx, (x, y) in enumerate(train_data):
                    x, y = x.to(device), y.to(device)
                    
                    optimizer.zero_grad()
                    output = model(x)
                    loss = model.criterion(output, y)
                    loss.backward()
                    
                    # Synchronize gradients
                    if self.backend == DistributedBackend.HOROVOD:
                        optimizer.synchronize()
                        
                    optimizer.step()
                    total_loss += loss.item()
                    
                # Validation
                if val_data is not None:
                    val_metrics = self._evaluate_torch_model(model, val_data)
                    
                # Call callbacks
                for callback in callbacks:
                    callback.on_epoch_end(epoch, {
                        'loss': total_loss / len(train_data),
                        **val_metrics if val_data is not None else {}
                    })
                    
            except Exception as e:
                print(f"Error during training: {e}")
                if self.rank == 0:
                    # Save checkpoint for recovery
                    self._save_checkpoint(model, optimizer, epoch)
                continue
                
        return model
        
    def _train_tf_model(
        self,
        model: tf.keras.Model,
        train_data: tf.data.Dataset,
        val_data: Optional[tf.data.Dataset],
        epochs: int,
        callbacks: Optional[List[Any]]
    ) -> tf.keras.Model:
        """Train TensorFlow model."""
        if self.backend == DistributedBackend.TENSORFLOW_MIRROR:
            with self.strategy.scope():
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=epochs,
                    callbacks=callbacks
                )
        else:  # Horovod
            callbacks = callbacks or []
            callbacks.append(hvd_tf.callbacks.BroadcastGlobalVariablesCallback(0))
            
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks
            )
            
        return model
        
    def _evaluate_torch_model(
        self,
        model: torch.nn.Module,
        test_data: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate PyTorch model."""
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_data:
                x, y = x.to(device), y.to(device)
                output = model(x)
                total_loss += model.criterion(output, y).item()
                
                _, predicted = output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
        # Gather metrics from all processes
        if self.backend == DistributedBackend.PYTORCH_DDP:
            metrics = torch.tensor([total_loss, correct, total]).to(device)
            dist.all_reduce(metrics)
            total_loss, correct, total = metrics.tolist()
            
        return {
            'loss': total_loss / len(test_data),
            'accuracy': 100. * correct / total
        }
        
    def _evaluate_tf_model(
        self,
        model: tf.keras.Model,
        test_data: tf.data.Dataset
    ) -> Dict[str, float]:
        """Evaluate TensorFlow model."""
        if self.backend == DistributedBackend.TENSORFLOW_MIRROR:
            with self.strategy.scope():
                results = model.evaluate(test_data)
        else:  # Horovod
            results = model.evaluate(test_data)
            
        return dict(zip(model.metrics_names, results))
        
    def _save_checkpoint(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        optimizer: Any,
        epoch: int
    ):
        """Save training checkpoint."""
        if isinstance(model, torch.nn.Module):
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f'checkpoint_rank{self.rank}.pt')
        else:
            model.save(f'checkpoint_rank{self.rank}') 