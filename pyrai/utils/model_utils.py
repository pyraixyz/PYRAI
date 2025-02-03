import os
import json
import torch
import numpy as np
from typing import Any, Dict, Optional, Union
from pathlib import Path

def save_model_info(
    path: Union[str, Path],
    model_info: Dict[str, Any]
) -> None:
    """Save model information.
    
    Args:
        path: Save path
        model_info: Model information dictionary
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(model_info, f, indent=2)

def load_model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Load model information.
    
    Args:
        path: Path to model info file
        
    Returns:
        Model information dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model info not found: {path}")
        
    with open(path) as f:
        return json.load(f)

def get_model_size(model: torch.nn.Module) -> int:
    """Get model size in bytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in bytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    return param_size + buffer_size

def get_model_flops(
    model: torch.nn.Module,
    input_size: tuple,
    device: Optional[str] = None
) -> int:
    """Estimate model FLOPs.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run calculation
        
    Returns:
        Estimated FLOPs
    """
    from thop import profile
    
    if device is None:
        device = next(model.parameters()).device
        
    input = torch.randn(input_size).to(device)
    flops, _ = profile(model, inputs=(input,))
    
    return int(flops) 