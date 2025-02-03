#!/usr/bin/env python
import os
import click
import torch
import yaml
from pathlib import Path
from pyrai import Engine
from pyrai.config import Config
from pyrai.utils.model_utils import get_model_size, get_model_flops
from pyrai.logging import Logger

logger = Logger.get_logger("cli")

@click.group()
def cli():
    """PyRAI command line tools."""
    pass

@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--input-size", required=True, help="Input size (e.g. '1,3,224,224')")
@click.option("--device", default=None, help="Device to use")
def analyze(model_path: str, input_size: str, device: Optional[str]):
    """Analyze model architecture."""
    try:
        # Load model
        model = torch.load(model_path)
        
        # Parse input size
        input_size = tuple(map(int, input_size.split(',')))
        
        # Get model info
        size_bytes = get_model_size(model)
        flops = get_model_flops(model, input_size, device)
        
        # Print analysis
        click.echo("\nModel Analysis:")
        click.echo(f"Path: {model_path}")
        click.echo(f"Size: {size_bytes / 1024 / 1024:.2f} MB")
        click.echo(f"FLOPs: {flops / 1e6:.2f} M")
        click.echo(f"Parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Print architecture
        click.echo("\nModel Architecture:")
        click.echo(model)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--format", type=click.Choice(['onnx', 'torchscript']), required=True)
@click.option("--input-size", required=True, help="Input size (e.g. '1,3,224,224')")
def export(model_path: str, output_path: str, format: str, input_size: str):
    """Export model to different formats."""
    try:
        # Load model
        model = torch.load(model_path)
        model.eval()
        
        # Parse input size
        input_size = tuple(map(int, input_size.split(',')))
        dummy_input = torch.randn(input_size)
        
        # Export model
        if format == 'onnx':
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                verbose=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        elif format == 'torchscript':
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(traced_model, output_path)
            
        logger.info(f"Model exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path(), default="benchmark_results")
@click.option("--batch-sizes", default="1,8,32", help="Comma-separated batch sizes")
@click.option("--iterations", default=100, help="Number of iterations per batch size")
def benchmark(model_path: str, output_dir: str, batch_sizes: str, iterations: int):
    """Run model benchmarks."""
    try:
        from pyrai.scripts.benchmark import run_benchmark
        
        # Parse batch sizes
        batch_sizes = list(map(int, batch_sizes.split(',')))
        
        # Run benchmark
        results = run_benchmark(
            model_path=model_path,
            batch_sizes=batch_sizes,
            num_iterations=iterations
        )
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "benchmark_results.yaml")
        
        with open(output_path, 'w') as f:
            yaml.dump(results, f, indent=2)
            
        logger.info(f"Benchmark results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == "__main__":
    cli() 