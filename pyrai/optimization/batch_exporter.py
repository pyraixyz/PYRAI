from typing import Any, Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from dataclasses import dataclass
from enum import Enum
from .export_manager import ExportManager, ExportFormat
from .experiment_manager import ExperimentResult

class BatchStatus(Enum):
    """Batch export status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchConfig:
    """Batch export configuration."""
    formats: List[ExportFormat]
    max_workers: int = 4
    timeout: Optional[float] = None
    retry_count: int = 3
    retry_delay: float = 1.0

@dataclass
class BatchResult:
    """Batch export result."""
    status: BatchStatus
    files: Dict[ExportFormat, str]
    errors: Dict[ExportFormat, str]
    duration: float

class BatchExporter:
    """Manages batch exports."""
    
    def __init__(
        self,
        export_manager: ExportManager,
        config: BatchConfig
    ):
        """Initialize batch exporter.
        
        Args:
            export_manager: Export manager
            config: Batch configuration
        """
        self.export_manager = export_manager
        self.config = config
        
    def export_batch(
        self,
        results: List[ExperimentResult],
        base_filename: Optional[str] = None
    ) -> BatchResult:
        """Run batch export.
        
        Args:
            results: Experiment results
            base_filename: Optional base filename
            
        Returns:
            Batch result
        """
        batch_result = BatchResult(
            status=BatchStatus.RUNNING,
            files={},
            errors={},
            duration=0.0
        )
        
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit export tasks
                future_to_format = {
                    executor.submit(
                        self._export_with_retry,
                        results,
                        format,
                        base_filename
                    ): format
                    for format in self.config.formats
                }
                
                # Collect results
                for future in future_to_format:
                    format = future_to_format[future]
                    try:
                        filepath = future.result(timeout=self.config.timeout)
                        batch_result.files[format] = filepath
                    except Exception as e:
                        batch_result.errors[format] = str(e)
                        
            # Set final status
            if batch_result.errors:
                if len(batch_result.errors) == len(self.config.formats):
                    batch_result.status = BatchStatus.FAILED
                else:
                    batch_result.status = BatchStatus.COMPLETED
            else:
                batch_result.status = BatchStatus.COMPLETED
                
        except Exception as e:
            batch_result.status = BatchStatus.FAILED
            batch_result.errors["batch"] = str(e)
            
        batch_result.duration = time.time() - start_time
        return batch_result
        
    def _export_with_retry(
        self,
        results: List[ExperimentResult],
        format: ExportFormat,
        base_filename: Optional[str]
    ) -> str:
        """Export with retry logic."""
        last_error = None
        
        for attempt in range(self.config.retry_count):
            try:
                return self.export_manager.export_results(
                    results,
                    format,
                    f"{base_filename}_{attempt}" if base_filename else None
                )
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)
                    
        raise last_error
        
    def get_export_summary(
        self,
        batch_result: BatchResult
    ) -> Dict[str, Any]:
        """Get export summary."""
        return {
            "status": batch_result.status.value,
            "duration": batch_result.duration,
            "successful_formats": list(batch_result.files.keys()),
            "failed_formats": list(batch_result.errors.keys()),
            "total_files": len(batch_result.files),
            "total_errors": len(batch_result.errors),
            "files": batch_result.files,
            "errors": batch_result.errors
        }
        
    def cleanup_exports(
        self,
        batch_result: BatchResult,
        keep_failed: bool = True
    ):
        """Clean up exported files."""
        for format, filepath in batch_result.files.items():
            if format in batch_result.errors and keep_failed:
                continue
                
            try:
                Path(filepath).unlink()
            except Exception:
                pass 