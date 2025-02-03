import os
import click
import yaml
from typing import Dict, Optional
from pathlib import Path
from pyrai import Engine
from pyrai.backends import PyTorchBackend, TensorFlowBackend
from pyrai.config import Config
from pyrai.utils import load_config
from pyrai.logging import Logger

logger = Logger.get_logger("cli")

@click.group()
def cli():
    """PyRAI command line interface."""
    pass

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--model-id", required=True, help="Model identifier")
@click.option("--data-path", required=True, help="Training data path")
@click.option("--output-dir", default="models", help="Output directory")
def train(config_path: str, model_id: str, data_path: str, output_dir: str):
    """Train a model."""
    try:
        # Load configuration
        config = load_config(config_path)
        config["model"]["save_dir"] = output_dir
        
        # Initialize engine
        engine = Engine(config=Config(config))
        
        # Load data
        train_data, val_data = engine.load_data(data_path)
        
        # Train model
        model = engine.create_model(model_id)
        history = engine.train(
            model=model,
            train_data=train_data,
            val_data=val_data
        )
        
        logger.info(f"Training completed: {history}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument("model-path", type=click.Path(exists=True))
@click.option("--model-id", required=True, help="Model identifier")
@click.option("--version", help="Model version")
@click.option("--config-path", type=click.Path(exists=True), help="Deployment config")
def deploy(model_path: str, model_id: str, version: Optional[str], config_path: Optional[str]):
    """Deploy a model."""
    try:
        # Load configuration
        config = {}
        if config_path:
            config = load_config(config_path)
            
        # Initialize engine
        engine = Engine(config=Config(config))
        
        # Load and deploy model
        model = engine.load_model(model_path)
        deployment = engine.deploy(
            model=model,
            model_id=model_id,
            version=version
        )
        
        logger.info(f"Model deployed: {deployment.get_status()}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument("model-id")
@click.option("--version", help="Model version")
def status(model_id: str, version: Optional[str]):
    """Get model deployment status."""
    try:
        engine = Engine()
        deployment = engine.get_deployment(model_id, version)
        click.echo(deployment.get_status())
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument("model-id")
@click.option("--version", help="Model version")
def stop(model_id: str, version: Optional[str]):
    """Stop model deployment."""
    try:
        engine = Engine()
        deployment = engine.get_deployment(model_id, version)
        deployment.stop()
        logger.info(f"Deployment stopped: {model_id}")
        
    except Exception as e:
        logger.error(f"Stop failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == "__main__":
    cli() 