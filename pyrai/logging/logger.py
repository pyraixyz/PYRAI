import os
import sys
import logging
from typing import Optional
from datetime import datetime

class Logger:
    """PyRAI logger."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """Get or create logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path
            
        Returns:
            Logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
            
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def set_level(cls, name: str, level: int):
        """Set logger level.
        
        Args:
            name: Logger name
            level: Logging level
        """
        if name in cls._loggers:
            cls._loggers[name].setLevel(level)

class TrainingLogger:
    """Training progress logger."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None
    ):
        """Initialize training logger.
        
        Args:
            log_dir: Logging directory
            experiment_name: Experiment name
        """
        self.log_dir = log_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize logger
        log_file = os.path.join(self.experiment_dir, "training.log")
        self.logger = Logger.get_logger(
            f"training.{experiment_name}",
            log_file=log_file
        )
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics.
        
        Args:
            metrics: Metric values
            step: Training step
        """
        # Update metrics
        for name, value in metrics.items():
            self.metrics[name].append(value)
            
        # Log to file
        metric_str = " - ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
        )
        self.logger.info(f"Step {step}: {metric_str}")
        
    def save_metrics(self):
        """Save metrics to file."""
        import json
        
        metrics_file = os.path.join(
            self.experiment_dir,
            "metrics.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics.
        
        Args:
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(
            len(self.metrics),
            1,
            figsize=(10, 4 * len(self.metrics))
        )
        if len(self.metrics) == 1:
            axes = [axes]
            
        for ax, (name, values) in zip(axes, self.metrics.items()):
            ax.plot(values)
            ax.set_title(name)
            ax.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 