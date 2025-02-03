from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .fault_tolerance import FaultEvent, FaultType
from .resources import ResourceManager
from .coordinator import ResourceCoordinator

class RecoveryStrategy(Enum):
    """Recovery strategies."""
    CHECKPOINT = "checkpoint"
    REPLICATION = "replication"
    MIGRATION = "migration"
    ADAPTIVE = "adaptive"

@dataclass
class RecoveryPlan:
    """Recovery plan details."""
    strategy: RecoveryStrategy
    actions: List[Dict[str, Any]]
    estimated_time: float
    resource_cost: float

class RecoveryManager:
    """Manages fault recovery."""
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        coordinator: ResourceCoordinator,
        checkpoint_dir: str = "checkpoints",
        max_retry_attempts: int = 3
    ):
        """Initialize recovery manager.
        
        Args:
            resource_manager: Resource manager
            coordinator: Resource coordinator
            checkpoint_dir: Checkpoint directory
            max_retry_attempts: Maximum recovery attempts
        """
        self.resource_manager = resource_manager
        self.coordinator = coordinator
        self.checkpoint_dir = checkpoint_dir
        self.max_retry_attempts = max_retry_attempts
        self.recovery_history: List[Tuple[FaultEvent, RecoveryPlan]] = []
        self.active_recoveries: Dict[int, RecoveryPlan] = {}
        
    def handle_fault(self, event: FaultEvent) -> bool:
        """Handle fault event with recovery.
        
        Args:
            event: Fault event to handle
            
        Returns:
            Whether recovery was successful
        """
        # Check if already recovering
        if event.worker_id in self.active_recoveries:
            return False
            
        # Generate recovery plan
        plan = self._generate_recovery_plan(event)
        if not plan:
            return False
            
        # Execute recovery
        success = self._execute_recovery_plan(event, plan)
        if success:
            self.recovery_history.append((event, plan))
            
        return success
        
    def _generate_recovery_plan(self, event: FaultEvent) -> Optional[RecoveryPlan]:
        """Generate recovery plan for fault."""
        if event.type == FaultType.WORKER_FAILURE:
            return self._plan_worker_recovery(event)
        elif event.type == FaultType.MEMORY_OVERFLOW:
            return self._plan_memory_recovery(event)
        elif event.type == FaultType.PERFORMANCE_DEGRADATION:
            return self._plan_performance_recovery(event)
        elif event.type == FaultType.COMMUNICATION_ERROR:
            return self._plan_communication_recovery(event)
        return None
        
    def _execute_recovery_plan(
        self,
        event: FaultEvent,
        plan: RecoveryPlan
    ) -> bool:
        """Execute recovery plan."""
        try:
            self.active_recoveries[event.worker_id] = plan
            
            for action in plan.actions:
                if action["type"] == "checkpoint_restore":
                    success = self._restore_from_checkpoint(
                        action["checkpoint_path"],
                        action["target_device"]
                    )
                elif action["type"] == "resource_reallocation":
                    success = self._reallocate_resources(
                        action["resources"],
                        action["target_device"]
                    )
                elif action["type"] == "worker_migration":
                    success = self._migrate_worker(
                        action["worker_id"],
                        action["target_device"]
                    )
                else:
                    success = False
                    
                if not success:
                    return False
                    
            del self.active_recoveries[event.worker_id]
            return True
            
        except Exception as e:
            print(f"Recovery execution failed: {e}")
            return False
            
    def _plan_worker_recovery(self, event: FaultEvent) -> RecoveryPlan:
        """Plan recovery for worker failure."""
        # Find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint(event.worker_id)
        
        # Find available device
        target_device = self._find_available_device()
        
        if checkpoint_path and target_device is not None:
            return RecoveryPlan(
                strategy=RecoveryStrategy.CHECKPOINT,
                actions=[{
                    "type": "checkpoint_restore",
                    "checkpoint_path": checkpoint_path,
                    "target_device": target_device
                }],
                estimated_time=30.0,  # Estimated restore time
                resource_cost=1.0  # Relative cost
            )
        else:
            # Try replication strategy
            return self._plan_replication_recovery(event)
            
    def _plan_memory_recovery(self, event: FaultEvent) -> RecoveryPlan:
        """Plan recovery for memory overflow."""
        actions = []
        
        # Try to free memory
        actions.append({
            "type": "resource_reallocation",
            "resources": {"memory": -0.2},  # Reduce by 20%
            "target_device": event.worker_id
        })
        
        # Plan migration if needed
        target_device = self._find_available_device(
            memory_requirement=event.details.get("memory_used", 0)
        )
        if target_device is not None:
            actions.append({
                "type": "worker_migration",
                "worker_id": event.worker_id,
                "target_device": target_device
            })
            
        return RecoveryPlan(
            strategy=RecoveryStrategy.MIGRATION,
            actions=actions,
            estimated_time=10.0,
            resource_cost=0.5
        )
        
    def _find_latest_checkpoint(self, worker_id: int) -> Optional[str]:
        """Find latest checkpoint for worker."""
        import glob
        import os
        
        pattern = os.path.join(
            self.checkpoint_dir,
            f"worker_{worker_id}_*.pt"
        )
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None
            
        # Return most recent
        return max(checkpoints, key=os.path.getctime)
        
    def _find_available_device(
        self,
        memory_requirement: Optional[float] = None
    ) -> Optional[int]:
        """Find available device meeting requirements."""
        for device_id in range(self.resource_manager.num_devices):
            if memory_requirement is not None:
                available_memory = self.resource_manager.get_available(
                    "gpu_memory",
                    device_id
                )
                if available_memory < memory_requirement:
                    continue
                    
            # Check device health
            if self._check_device_health(device_id):
                return device_id
                
        return None
        
    def _check_device_health(self, device_id: int) -> bool:
        """Check if device is healthy."""
        try:
            if torch.cuda.is_available():
                # Test device
                torch.cuda.device(device_id)
                test_tensor = torch.zeros(1).cuda(device_id)
                del test_tensor
                return True
        except:
            pass
        return False 