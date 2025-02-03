import time
import argparse
import numpy as np
from pathlib import Path
from pyrai import Engine
from pyrai.config import Config
from pyrai.utils import load_config
from pyrai.logging import Logger

logger = Logger.get_logger("benchmark")

def run_benchmark(
    model_path: str,
    data_path: str,
    batch_sizes: list,
    num_iterations: int = 100
):
    """Run performance benchmark.
    
    Args:
        model_path: Path to model
        data_path: Path to test data
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations per batch size
    """
    # Load model and data
    engine = Engine()
    model = engine.load_model(model_path)
    test_data = engine.load_data(data_path)
    
    results = {}
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        latencies = []
        throughputs = []
        memory_usage = []
        
        for _ in range(num_iterations):
            # Prepare batch
            batch = test_data.get_batch(batch_size)
            
            # Measure inference
            start_time = time.time()
            _ = engine.predict(model, batch)
            end_time = time.time()
            
            # Calculate metrics
            latency = (end_time - start_time) * 1000  # ms
            throughput = batch_size / (end_time - start_time)
            
            latencies.append(latency)
            throughputs.append(throughput)
            memory_usage.append(engine.get_memory_usage())
        
        # Compute statistics
        results[batch_size] = {
            "latency": {
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            },
            "throughput": {
                "mean": np.mean(throughputs),
                "std": np.std(throughputs)
            },
            "memory": {
                "mean": np.mean(memory_usage),
                "max": np.max(memory_usage)
            }
        }
        
        logger.info(f"Results for batch size {batch_size}:")
        logger.info(f"  Latency (ms): {results[batch_size]['latency']}")
        logger.info(f"  Throughput: {results[batch_size]['throughput']}")
        logger.info(f"  Memory (MB): {results[batch_size]['memory']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run performance benchmark")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--data-path", required=True, help="Path to test data")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128],
                      help="Batch sizes to test")
    parser.add_argument("--iterations", type=int, default=100,
                      help="Number of iterations per batch size")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        results = run_benchmark(
            args.model_path,
            args.data_path,
            args.batch_sizes,
            args.iterations
        )
        
        if args.output:
            import json
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
                
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 