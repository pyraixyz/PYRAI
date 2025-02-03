from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from .parallel_tester import ParallelTester, ParallelTestConfig, TestStatus
from .ab_tester import TestConfig, TestResult
from .monitor import RuntimeStats

class ScheduleMode(Enum):
    """Test scheduling modes."""
    SEQUENTIAL = "sequential"  # Run tests in sequence
    PARALLEL = "parallel"     # Run tests in parallel
    ADAPTIVE = "adaptive"     # Adapt based on resource usage

@dataclass
class ScheduleConfig:
    """Test schedule configuration."""
    test_configs: List[TestConfig]
    mode: ScheduleMode
    max_concurrent: int = 3
    resource_threshold: float = 0.8
    retry_attempts: int = 3
    retry_delay: float = 300.0  # 5 minutes

class TestScheduler:
    """Manages test scheduling and execution."""
    
    def __init__(
        self,
        config: ScheduleConfig,
        resource_monitor: Any  # Resource monitor interface
    ):
        """Initialize scheduler.
        
        Args:
            config: Schedule configuration
            resource_monitor: Resource monitor
        """
        self.config = config
        self.resource_monitor = resource_monitor
        self.active_testers: Dict[str, ParallelTester] = {}
        self.completed_tests: Dict[str, TestResult] = {}
        self.failed_tests: Dict[str, str] = {}  # test_id -> error
        self.retry_counts: Dict[str, int] = {}
        
    async def schedule_tests(self):
        """Schedule and run tests."""
        if self.config.mode == ScheduleMode.SEQUENTIAL:
            await self._run_sequential()
        elif self.config.mode == ScheduleMode.PARALLEL:
            await self._run_parallel()
        else:  # ADAPTIVE
            await self._run_adaptive()
            
    async def _run_sequential(self):
        """Run tests sequentially."""
        for i, test_config in enumerate(self.config.test_configs):
            test_id = f"test_{i}"
            
            # Create single test parallel tester
            tester = ParallelTester(
                ParallelTestConfig(
                    test_configs=[test_config],
                    max_concurrent=1
                )
            )
            
            self.active_testers[test_id] = tester
            results = await tester.run_tests()
            
            # Handle results
            await self._handle_test_results(test_id, results)
            
    async def _run_parallel(self):
        """Run tests in parallel."""
        # Create parallel tester with all configs
        tester = ParallelTester(
            ParallelTestConfig(
                test_configs=self.config.test_configs,
                max_concurrent=self.config.max_concurrent
            )
        )
        
        results = await tester.run_tests()
        
        # Handle results
        for test_id, result in results.items():
            await self._handle_test_results(test_id, {test_id: result})
            
    async def _run_adaptive(self):
        """Run tests adaptively based on resource usage."""
        pending_configs = self.config.test_configs.copy()
        current_concurrent = self.config.max_concurrent
        
        while pending_configs:
            # Check resource usage
            resource_usage = await self._get_resource_usage()
            
            # Adjust concurrency
            if resource_usage > self.config.resource_threshold:
                current_concurrent = max(1, current_concurrent - 1)
            else:
                current_concurrent = min(
                    self.config.max_concurrent,
                    current_concurrent + 1
                )
                
            # Take next batch of configs
            batch_configs = pending_configs[:current_concurrent]
            pending_configs = pending_configs[current_concurrent:]
            
            # Run batch
            tester = ParallelTester(
                ParallelTestConfig(
                    test_configs=batch_configs,
                    max_concurrent=current_concurrent
                )
            )
            
            results = await tester.run_tests()
            
            # Handle results
            for test_id, result in results.items():
                await self._handle_test_results(test_id, {test_id: result})
                
    async def _handle_test_results(
        self,
        test_id: str,
        results: Dict[str, TestResult]
    ):
        """Handle test results."""
        result = results[test_id]
        
        if result.status == TestStatus.COMPLETED:
            self.completed_tests[test_id] = result.result
            
        elif result.status == TestStatus.FAILED:
            # Check if should retry
            retry_count = self.retry_counts.get(test_id, 0)
            if retry_count < self.config.retry_attempts:
                self.retry_counts[test_id] = retry_count + 1
                await asyncio.sleep(self.config.retry_delay)
                
                # Retry test
                config = self.config.test_configs[int(test_id.split("_")[1])]
                tester = ParallelTester(
                    ParallelTestConfig(
                        test_configs=[config],
                        max_concurrent=1
                    )
                )
                
                retry_results = await tester.run_tests()
                await self._handle_test_results(test_id, retry_results)
                
            else:
                self.failed_tests[test_id] = result.error
                
    async def _get_resource_usage(self) -> float:
        """Get current resource usage."""
        stats = await self.resource_monitor.get_stats()
        return max(
            np.mean(stats.gpu_utilization),
            np.mean(stats.gpu_memory)
        )
        
    def get_status(self) -> Dict[str, Any]:
        """Get current schedule status."""
        return {
            "mode": self.config.mode.value,
            "total_tests": len(self.config.test_configs),
            "completed": len(self.completed_tests),
            "failed": len(self.failed_tests),
            "active": len(self.active_testers),
            "results": {
                test_id: {
                    "improvement": result.improvement,
                    "significant": result.significant,
                    "winner": result.winner.value if result.winner else None
                }
                for test_id, result in self.completed_tests.items()
            },
            "failures": self.failed_tests
        } 