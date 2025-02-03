from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import tensorflow as tf
import os
import time
from .fault_tolerance import FaultHandler, FaultEvent, FaultType

class CheckpointHandler(FaultHandler):
    """Handles faults by saving and loading checkpoints."""
    
    def __init__(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        optimizer: Any,
        checkpoint_dir: str,
        max_checkpoints: int = 5
    ):
        """Initialize checkpoint handler.
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            checkpoint_dir: Checkpoint directory
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[str] = []
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def handle(self, event: FaultEvent) -> bool:
        """Handle fault by checkpointing or restoring."""
        try:
            if event.type in [FaultType.WORKER_FAILURE, FaultType.MEMORY_OVERFLOW]:
                return self._save_checkpoint(event)
            elif event.type == FaultType.COMMUNICATION_ERROR:
                return self._restore_checkpoint(event)
            return False
        except:
            return False
            
    def _save_checkpoint(self, event: FaultEvent) -> bool:
        """Save model checkpoint."""
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{int(event.timestamp)}.pt"
            )
            
            if isinstance(self.model, torch.nn.Module):
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'event': event
                }, checkpoint_path)
            else:
                self.model.save_weights(checkpoint_path)
                
            self.checkpoints.append(checkpoint_path)
            if len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    
            return True
        except:
            return False
            
    def _restore_checkpoint(self, event: FaultEvent) -> bool:
        """Restore from latest checkpoint."""
        try:
            if not self.checkpoints:
                return False
                
            latest_checkpoint = self.checkpoints[-1]
            if isinstance(self.model, torch.nn.Module):
                checkpoint = torch.load(latest_checkpoint)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.model.load_weights(latest_checkpoint)
                
            return True
        except:
            return False

class ResourceHandler(FaultHandler):
    """Handles resource-related faults."""
    
    def __init__(
        self,
        memory_threshold: float = 0.9,
        enable_gc: bool = True
    ):
        """Initialize resource handler.
        
        Args:
            memory_threshold: Memory usage threshold
            enable_gc: Whether to enable garbage collection
        """
        self.memory_threshold = memory_threshold
        self.enable_gc = enable_gc
        
    def handle(self, event: FaultEvent) -> bool:
        """Handle resource-related faults."""
        if event.type == FaultType.MEMORY_OVERFLOW:
            return self._handle_memory_overflow(event)
        return False
        
    def _handle_memory_overflow(self, event: FaultEvent) -> bool:
        """Handle memory overflow."""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Run garbage collection
            if self.enable_gc:
                import gc
                gc.collect()
                
            # Check if memory was freed
            if torch.cuda.is_available():
                device = event.details.get("device", 0)
                memory_used = torch.cuda.memory_allocated(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                return memory_used / memory_total < self.memory_threshold
                
            return True
        except:
            return False

class PerformanceHandler(FaultHandler):
    """Handles performance-related faults."""
    
    def __init__(
        self,
        scheduler: 'AdaptiveScheduler',
        performance_threshold: float = 0.7
    ):
        """Initialize performance handler.
        
        Args:
            scheduler: Adaptive scheduler
            performance_threshold: Performance degradation threshold
        """
        self.scheduler = scheduler
        self.performance_threshold = performance_threshold
        self.peak_performance = 0.0
        
    def handle(self, event: FaultEvent) -> bool:
        """Handle performance-related faults."""
        if event.type == FaultType.PERFORMANCE_DEGRADATION:
            return self._handle_performance_degradation(event)
        return False
        
    def _handle_performance_degradation(self, event: FaultEvent) -> bool:
        """Handle performance degradation."""
        try:
            # Update scheduling policy
            if self.scheduler.policy != SchedulingPolicy.PERFORMANCE_AWARE:
                self.scheduler.policy = SchedulingPolicy.PERFORMANCE_AWARE
                return True
            return False
        except:
            return False 