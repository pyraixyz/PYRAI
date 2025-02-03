from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import torch.distributed as dist
import tensorflow as tf
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum
from threading import Thread, Lock

class FaultType(Enum):
    """Types of faults that can occur."""
    WORKER_FAILURE = "worker_failure"
    MEMORY_OVERFLOW = "memory_overflow"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COMMUNICATION_ERROR = "communication_error"

@dataclass
class FaultEvent:
    """Fault event information."""
    type: FaultType
    timestamp: float
    worker_id: int
    details: Dict[str, Any]

class FaultHandler:
    """Base class for fault handlers."""
    
    def handle(self, event: FaultEvent) -> bool:
        """Handle fault event.
        
        Returns:
            True if fault was handled successfully
        """
        raise NotImplementedError

class FaultToleranceManager:
    """Manages fault detection and recovery."""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        checkpoint_dir: str = "checkpoints",
        monitoring_interval: float = 1.0
    ):
        """Initialize fault tolerance manager.
        
        Args:
            world_size: Number of workers
            rank: Current worker rank
            checkpoint_dir: Checkpoint directory
            monitoring_interval: Fault monitoring interval
        """
        self.world_size = world_size
        self.rank = rank
        self.checkpoint_dir = checkpoint_dir
        self.monitoring_interval = monitoring_interval
        self.handlers: Dict[FaultType, List[FaultHandler]] = {
            fault_type: [] for fault_type in FaultType
        }
        self.active_workers = set(range(world_size))
        self.lock = Lock()
        self._stop_flag = False
        
    def start(self):
        """Start fault monitoring."""
        self._stop_flag = False
        self._monitor_thread = Thread(target=self._monitoring_loop)
        self._monitor_thread.start()
        
    def stop(self):
        """Stop fault monitoring."""
        self._stop_flag = True
        self._monitor_thread.join()
        
    def register_handler(self, fault_type: FaultType, handler: FaultHandler):
        """Register fault handler."""
        with self.lock:
            self.handlers[fault_type].append(handler)
            
    def handle_fault(self, event: FaultEvent) -> bool:
        """Handle fault event."""
        with self.lock:
            handlers = self.handlers[event.type]
            for handler in handlers:
                if handler.handle(event):
                    return True
            return False
            
    def _monitoring_loop(self):
        """Main fault monitoring loop."""
        while not self._stop_flag:
            try:
                # Check worker health
                self._check_worker_health()
                
                # Check memory usage
                self._check_memory_usage()
                
                # Check performance
                self._check_performance()
                
                # Check communication
                self._check_communication()
                
            except Exception as e:
                print(f"Fault monitoring error: {e}")
                
            time.sleep(self.monitoring_interval)
            
    def _check_worker_health(self):
        """Check health of all workers."""
        try:
            # Ping all workers
            alive = torch.zeros(self.world_size)
            if self.rank == 0:
                alive[0] = 1
                for i in range(1, self.world_size):
                    try:
                        dist.send(alive, i)
                        alive[i] = 1
                    except:
                        if i in self.active_workers:
                            self._handle_worker_failure(i)
            else:
                try:
                    dist.recv(alive, 0)
                except:
                    self._handle_communication_error()
                    
        except Exception as e:
            print(f"Health check error: {e}")
            
    def _check_memory_usage(self):
        """Check GPU memory usage."""
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    if memory_used / memory_total > 0.95:  # 95% threshold
                        self.handle_fault(FaultEvent(
                            type=FaultType.MEMORY_OVERFLOW,
                            timestamp=time.time(),
                            worker_id=self.rank,
                            details={"device": i, "memory_used": memory_used}
                        ))
        except Exception as e:
            print(f"Memory check error: {e}")
            
    def _check_performance(self):
        """Check performance metrics."""
        # This should be implemented based on your monitoring system
        pass
        
    def _check_communication(self):
        """Check communication health."""
        try:
            tensor = torch.ones(1)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
                
            # All-reduce to test communication
            dist.all_reduce(tensor)
            if tensor.item() != self.world_size:
                self._handle_communication_error()
                
        except Exception as e:
            self._handle_communication_error()
            
    def _handle_worker_failure(self, worker_id: int):
        """Handle worker failure."""
        event = FaultEvent(
            type=FaultType.WORKER_FAILURE,
            timestamp=time.time(),
            worker_id=worker_id,
            details={}
        )
        if self.handle_fault(event):
            self.active_workers.remove(worker_id)
            
    def _handle_communication_error(self):
        """Handle communication error."""
        event = FaultEvent(
            type=FaultType.COMMUNICATION_ERROR,
            timestamp=time.time(),
            worker_id=self.rank,
            details={}
        )
        self.handle_fault(event) 