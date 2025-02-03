from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .resources import ResourceManager, ResourceType
from .resource_predictor import ResourcePredictor
from .resource_adjuster import ResourceAdjuster, AdjustmentAction

class CoordinationStrategy(Enum):
    """Resource coordination strategies."""
    CENTRALIZED = "centralized"    # Single coordinator
    DECENTRALIZED = "decentralized"  # Peer-to-peer
    HIERARCHICAL = "hierarchical"   # Tree-based

@dataclass
class CoordinationState:
    """Global coordination state."""
    global_allocations: Dict[Tuple[ResourceType, int], float]
    worker_states: Dict[int, Dict[str, Any]]
    timestamp: float

class ResourceCoordinator:
    """Coordinates distributed resource management."""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        resource_manager: ResourceManager,
        predictor: ResourcePredictor,
        adjuster: ResourceAdjuster,
        strategy: CoordinationStrategy = CoordinationStrategy.HIERARCHICAL,
        sync_interval: float = 5.0
    ):
        """Initialize coordinator.
        
        Args:
            world_size: Number of workers
            rank: Worker rank
            resource_manager: Local resource manager
            predictor: Resource predictor
            adjuster: Resource adjuster
            strategy: Coordination strategy
            sync_interval: State sync interval
        """
        self.world_size = world_size
        self.rank = rank
        self.resource_manager = resource_manager
        self.predictor = predictor
        self.adjuster = adjuster
        self.strategy = strategy
        self.sync_interval = sync_interval
        self.state = CoordinationState(
            global_allocations={},
            worker_states={},
            timestamp=0.0
        )
        
    def coordinate(self, current_time: float) -> List[AdjustmentAction]:
        """Coordinate resource adjustments.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            List of adjustment actions
        """
        # Sync state if needed
        if current_time - self.state.timestamp >= self.sync_interval:
            self._sync_state(current_time)
            
        if self.strategy == CoordinationStrategy.CENTRALIZED:
            return self._centralized_coordinate()
        elif self.strategy == CoordinationStrategy.DECENTRALIZED:
            return self._decentralized_coordinate()
        else:  # Hierarchical
            return self._hierarchical_coordinate()
            
    def _sync_state(self, current_time: float):
        """Synchronize global state."""
        # Gather local state
        local_state = {
            "allocations": {
                (rt, did): self.resource_manager.get_utilization(rt, did)
                for rt in ResourceType
                for did in range(self.resource_manager.num_devices)
            },
            "predictions": {
                (rt, did): self.predictor.predict(rt, did)
                for rt in ResourceType
                for did in range(self.resource_manager.num_devices)
            }
        }
        
        # Gather states from all workers
        gathered_states = [None] * self.world_size
        dist.all_gather_object(gathered_states, local_state)
        
        # Update global state
        self.state.worker_states = {
            i: state for i, state in enumerate(gathered_states)
        }
        self.state.timestamp = current_time
        
        # Aggregate global allocations
        self.state.global_allocations = {}
        for worker_state in self.state.worker_states.values():
            for key, value in worker_state["allocations"].items():
                if key not in self.state.global_allocations:
                    self.state.global_allocations[key] = 0.0
                self.state.global_allocations[key] += value
                
    def _centralized_coordinate(self) -> List[AdjustmentAction]:
        """Centralized coordination strategy."""
        if self.rank == 0:  # Coordinator
            actions = []
            
            # Check global resource usage
            for (rt, did), usage in self.state.global_allocations.items():
                # Get predictions from all workers
                predictions = []
                for worker_state in self.state.worker_states.values():
                    if (rt, did) in worker_state["predictions"]:
                        predictions.extend(worker_state["predictions"][(rt, did)])
                        
                if predictions:
                    # Determine global adjustment
                    action = self.adjuster._determine_adjustment(rt, did, predictions)
                    if action:
                        actions.append(action)
                        
            # Broadcast actions to all workers
            dist.broadcast_object_list([actions], 0)
            return actions
        else:
            # Receive actions from coordinator
            actions = [None]
            dist.broadcast_object_list(actions, 0)
            return actions[0]
            
    def _decentralized_coordinate(self) -> List[AdjustmentAction]:
        """Decentralized coordination strategy."""
        local_actions = self.adjuster.adjust(self.state.timestamp)
        
        # Share proposed actions with neighbors
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size
        
        # Send to next, receive from previous
        send_req = dist.isend(local_actions, next_rank)
        recv_actions = [None]
        dist.recv(recv_actions, prev_rank)
        send_req.wait()
        
        # Merge actions
        return self._merge_actions(local_actions, recv_actions[0])
        
    def _hierarchical_coordinate(self) -> List[AdjustmentAction]:
        """Hierarchical coordination strategy."""
        local_actions = self.adjuster.adjust(self.state.timestamp)
        
        # Determine parent and children in tree
        parent = (self.rank - 1) // 2 if self.rank > 0 else None
        children = [
            2 * self.rank + i + 1
            for i in range(2)
            if 2 * self.rank + i + 1 < self.world_size
        ]
        
        # Gather actions from children
        child_actions = []
        for child in children:
            actions = [None]
            dist.recv(actions, child)
            child_actions.extend(actions[0])
            
        # Merge all actions
        merged_actions = local_actions
        for actions in child_actions:
            merged_actions = self._merge_actions(merged_actions, actions)
            
        # Send to parent if not root
        if parent is not None:
            dist.send(merged_actions, parent)
            
            # Wait for final actions from parent
            final_actions = [None]
            dist.recv(final_actions, parent)
            return final_actions[0]
            
        return merged_actions
        
    def _merge_actions(
        self,
        actions1: List[AdjustmentAction],
        actions2: List[AdjustmentAction]
    ) -> List[AdjustmentAction]:
        """Merge two lists of adjustment actions."""
        merged = {}
        
        # Helper to add action
        def add_action(action):
            key = (action.resource_type, action.device_id)
            if key not in merged:
                merged[key] = action
            else:
                # Take action with larger absolute change
                if abs(action.amount_change) > abs(merged[key].amount_change):
                    merged[key] = action
                    
        # Add all actions
        for action in actions1:
            add_action(action)
        for action in actions2:
            add_action(action)
            
        return list(merged.values()) 