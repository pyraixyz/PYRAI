"""SDK utility functions."""

from typing import Dict, Any, Optional
import torch
import tensorflow as tf
from pathlib import Path
import json

def load_model_config(path: str) -> Dict[str, Any]:
    """Load model configuration from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path) as f:
        return json.load(f)

def export_torch_model(
    model: "torch.nn.Module",
    save_path: str,
    input_shape: Optional[tuple] = None
):
    """Export PyTorch model for distillation."""
    if input_shape:
        # Export traced model
        example_input = torch.randn(input_shape)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(save_path)
    else:
        # Export state dict
        torch.save(model.state_dict(), save_path)

def export_tf_model(
    model: "tf.keras.Model",
    save_path: str
):
    """Export TensorFlow model for distillation."""
    model.save(save_path) 