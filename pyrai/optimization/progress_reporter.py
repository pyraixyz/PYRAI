from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import time
import threading
from dataclasses import dataclass
from enum import Enum
import tqdm

class ProgressStage(Enum):
    """Export progress stages."""
    PREPARING = "preparing"
    EXPORTING = "exporting"
    VALIDATING = "validating"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProgressState:
    """Progress state."""
    stage: ProgressStage
    current: int
    total: int
    message: str
    error: Optional[str] = None

class ProgressCallback:
    """Progress callback interface."""
    
    def on_progress(self, state: ProgressState):
        """Handle progress update."""
        raise NotImplementedError

class ConsoleProgress(ProgressCallback):
    """Console progress reporter."""
    
    def __init__(self):
        """Initialize console progress."""
        self.pbar = None
        
    def on_progress(self, state: ProgressState):
        """Update console progress."""
        if state.stage in [ProgressStage.COMPLETED, ProgressStage.FAILED]:
            if self.pbar:
                self.pbar.close()
                self.pbar = None
                
            status = "✓" if state.stage == ProgressStage.COMPLETED else "✗"
            print(f"{status} {state.message}")
            if state.error:
                print(f"Error: {state.error}")
                
        else:
            if not self.pbar:
                self.pbar = tqdm.tqdm(
                    total=state.total,
                    desc=state.stage.value.capitalize(),
                    unit="files"
                )
                
            self.pbar.n = state.current
            self.pbar.set_description(state.message)
            self.pbar.refresh()

class ProgressReporter:
    """Manages export progress reporting."""
    
    def __init__(
        self,
        callbacks: Optional[List[ProgressCallback]] = None
    ):
        """Initialize reporter.
        
        Args:
            callbacks: Progress callbacks
        """
        self.callbacks = callbacks or [ConsoleProgress()]
        self.state = ProgressState(
            stage=ProgressStage.PREPARING,
            current=0,
            total=0,
            message="Initializing..."
        )
        self._lock = threading.Lock()
        
    def update(
        self,
        stage: Optional[ProgressStage] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update progress state."""
        with self._lock:
            if stage:
                self.state.stage = stage
            if current is not None:
                self.state.current = current
            if total is not None:
                self.state.total = total
            if message:
                self.state.message = message
            if error is not None:
                self.state.error = error
                
            for callback in self.callbacks:
                callback.on_progress(self.state)
                
    def wrap_operation(
        self,
        operation: Callable,
        stage: ProgressStage,
        total: int,
        message: str
    ) -> Any:
        """Wrap operation with progress reporting.
        
        Args:
            operation: Operation to wrap
            stage: Progress stage
            total: Total steps
            message: Progress message
            
        Returns:
            Operation result
        """
        self.update(
            stage=stage,
            current=0,
            total=total,
            message=message
        )
        
        try:
            result = operation()
            
            self.update(
                stage=ProgressStage.COMPLETED,
                current=total,
                message=f"Completed: {message}"
            )
            
            return result
            
        except Exception as e:
            self.update(
                stage=ProgressStage.FAILED,
                message=f"Failed: {message}",
                error=str(e)
            )
            raise 