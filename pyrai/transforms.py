from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

class Transform:
    """Base transform class."""
    
    def __call__(self, data: Any) -> Any:
        """Apply transform to data."""
        raise NotImplementedError

class Compose(Transform):
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[Transform]):
        """Initialize transform composition.
        
        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms
        
    def __call__(self, data: Any) -> Any:
        """Apply transforms sequentially."""
        for t in self.transforms:
            data = t(data)
        return data

class ToTensor(Transform):
    """Convert data to tensor."""
    
    def __call__(self, data: Any) -> torch.Tensor:
        """Convert to tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Image.Image):
            return T.ToTensor()(data)
        elif isinstance(data, torch.Tensor):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

class Normalize(Transform):
    """Normalize tensor data."""
    
    def __init__(
        self,
        mean: Union[float, List[float]],
        std: Union[float, List[float]]
    ):
        """Initialize normalization.
        
        Args:
            mean: Mean value(s)
            std: Standard deviation value(s)
        """
        self.mean = mean if isinstance(mean, (list, tuple)) else [mean]
        self.std = std if isinstance(std, (list, tuple)) else [std]
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a tensor")
            
        mean = torch.tensor(self.mean, device=tensor.device)
        std = torch.tensor(self.std, device=tensor.device)
        
        return (tensor - mean[None, :]) / std[None, :]

class RandomHorizontalFlip(Transform):
    """Random horizontal flip."""
    
    def __init__(self, p: float = 0.5):
        """Initialize random flip.
        
        Args:
            p: Probability of flipping
        """
        self.p = p
        
    def __call__(self, data: Any) -> Any:
        """Apply random flip."""
        if torch.rand(1) < self.p:
            if isinstance(data, torch.Tensor):
                return torch.flip(data, [-1])
            elif isinstance(data, np.ndarray):
                return np.flip(data, -1)
            elif isinstance(data, Image.Image):
                return data.transpose(Image.FLIP_LEFT_RIGHT)
        return data 