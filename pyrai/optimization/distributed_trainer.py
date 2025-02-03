from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass
from enum import Enum
from .distillation import KnowledgeDistillation, DistillationConfig

class DistributionStrategy(Enum):
    """Distribution strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE = "pipeline"
    HYBRID = "hybrid"

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    strategy: DistributionStrategy
    world_size: int
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"
    pipeline_chunks: int = 4
    gradient_accumulation: int = 1

class DistributedTrainer:
    """Manages distributed knowledge distillation."""
    
    def __init__(
        self,
        distillation: KnowledgeDistillation,
        config: DistributedConfig
    ):
        """Initialize distributed trainer.
        
        Args:
            distillation: Knowledge distillation instance
            config: Distribution configuration
        """
        self.distillation = distillation
        self.config = config
        
    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        **kwargs
    ):
        """Start distributed training.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            **kwargs: Additional training arguments
        """
        mp.spawn(
            self._train_worker,
            args=(train_dataset, val_dataset, kwargs),
            nprocs=self.config.world_size,
            join=True
        )
        
    def _train_worker(
        self,
        rank: int,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset],
        kwargs: Dict[str, Any]
    ):
        """Worker process for distributed training."""
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
            world_size=self.config.world_size,
            rank=rank
        )
        
        # Setup models for distributed training
        if self.config.strategy == DistributionStrategy.DATA_PARALLEL:
            self._setup_data_parallel(rank)
        elif self.config.strategy == DistributionStrategy.MODEL_PARALLEL:
            self._setup_model_parallel(rank)
        elif self.config.strategy == DistributionStrategy.PIPELINE:
            self._setup_pipeline(rank)
        else:  # HYBRID
            self._setup_hybrid(rank)
            
        # Create distributed samplers
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.config.world_size,
            rank=rank
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=kwargs.get("batch_size", 32),
            sampler=train_sampler,
            num_workers=kwargs.get("num_workers", 4),
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.config.world_size,
                rank=rank
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=kwargs.get("batch_size", 32),
                sampler=val_sampler,
                num_workers=kwargs.get("num_workers", 4),
                pin_memory=True
            )
            
        # Training loop
        optimizer = torch.optim.Adam(
            self.distillation.student.parameters(),
            **kwargs.get("optimizer_config", {})
        )
        
        for epoch in range(kwargs.get("epochs", 10)):
            train_sampler.set_epoch(epoch)
            self._train_epoch(train_loader, optimizer, rank)
            
            if val_loader and rank == 0:
                metrics = self.distillation.evaluate(val_loader)
                print(f"Epoch {epoch}: ", metrics)
                
        dist.destroy_process_group()
        
    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        rank: int
    ):
        """Train single epoch."""
        self.distillation.student.train()
        
        for i, batch in enumerate(train_loader):
            # Forward pass and loss computation
            metrics = self.distillation.train_step(batch, optimizer)
            
            # Log metrics on rank 0
            if rank == 0 and i % 100 == 0:
                print(f"Step {i}: ", metrics)
                
    def _setup_data_parallel(self, rank: int):
        """Setup data parallel training."""
        device = torch.device(f"cuda:{rank}")
        self.distillation.teacher = self.distillation.teacher.to(device)
        self.distillation.student = self.distillation.student.to(device)
        
        self.distillation.student = DDP(
            self.distillation.student,
            device_ids=[rank]
        )
        
    def _setup_model_parallel(self, rank: int):
        """Setup model parallel training."""
        # Split student model across devices
        device = torch.device(f"cuda:{rank}")
        if rank == 0:
            self.distillation.teacher = self.distillation.teacher.to(device)
            # First half of student model
            self.distillation.student.encoder = (
                self.distillation.student.encoder.to(device)
            )
        else:
            # Second half of student model
            self.distillation.student.decoder = (
                self.distillation.student.decoder.to(device)
            )
            
    def _setup_pipeline(self, rank: int):
        """Setup pipeline parallel training."""
        device = torch.device(f"cuda:{rank}")
        chunks = self.config.pipeline_chunks
        
        if rank == 0:
            self.distillation.teacher = self.distillation.teacher.to(device)
            # First stage of pipeline
            self.distillation.student.stage1 = (
                self.distillation.student.stage1.to(device)
            )
        elif rank == self.config.world_size - 1:
            # Last stage of pipeline
            self.distillation.student.stage3 = (
                self.distillation.student.stage3.to(device)
            )
        else:
            # Middle stages
            self.distillation.student.stage2 = (
                self.distillation.student.stage2.to(device)
            )
            
    def _setup_hybrid(self, rank: int):
        """Setup hybrid parallel training."""
        device = torch.device(f"cuda:{rank}")
        world_size = self.config.world_size
        
        # Combine data and model parallelism
        if rank < world_size // 2:
            # First model parallel group
            self.distillation.teacher = self.distillation.teacher.to(device)
            self.distillation.student.part1 = DDP(
                self.distillation.student.part1.to(device),
                device_ids=[rank]
            )
        else:
            # Second model parallel group
            self.distillation.student.part2 = DDP(
                self.distillation.student.part2.to(device),
                device_ids=[rank]
            ) 