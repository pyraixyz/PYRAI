import time
import psutil
import threading
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

class Monitor:
    """System resource monitor."""
    
    def __init__(self, interval: float = 1.0):
        """Initialize monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.metrics = defaultdict(list)
        self._running = False
        self._thread = None
        
    def start(self):
        """Start monitoring."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitor)
        self._thread.daemon = True
        self._thread.start()
        
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None
            
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get collected metrics."""
        return dict(self.metrics)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get metrics summary."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "mean": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values),
                    "std": np.std(values)
                }
        return summary
    
    def _monitor(self):
        """Monitor system resources."""
        while self._running:
            # CPU usage
            self.metrics["cpu_percent"].append(
                psutil.cpu_percent(interval=None)
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["memory_percent"].append(memory.percent)
            self.metrics["memory_used"].append(memory.used / 1024**3)  # GB
            
            # GPU usage if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.metrics["gpu_memory_used"].append(
                    info.used / 1024**3  # GB
                )
                self.metrics["gpu_memory_percent"].append(
                    info.used / info.total * 100
                )
            except:
                pass
                
            time.sleep(self.interval) 