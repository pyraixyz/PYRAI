from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

class BaseDataset(Dataset):
    """Base dataset class."""
    
    def __init__(self):
        """Initialize dataset."""
        super().__init__()
        
    def __len__(self) -> int:
        """Get dataset length."""
        raise NotImplementedError
        
    def __getitem__(self, idx: int) -> Any:
        """Get dataset item."""
        raise NotImplementedError

class TensorDataset(BaseDataset):
    """Dataset for tensor data."""
    
    def __init__(
        self,
        inputs: Union[np.ndarray, torch.Tensor, tf.Tensor],
        targets: Optional[Union[np.ndarray, torch.Tensor, tf.Tensor]] = None,
        transform: Optional[callable] = None
    ):
        """Initialize tensor dataset.
        
        Args:
            inputs: Input data
            targets: Target data
            transform: Data transform function
        """
        super().__init__()
        self.inputs = self._convert_tensor(inputs)
        self.targets = self._convert_tensor(targets) if targets is not None else None
        self.transform = transform
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.inputs)
        
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get dataset item."""
        inputs = self.inputs[idx]
        
        if self.transform:
            inputs = self.transform(inputs)
            
        if self.targets is not None:
            targets = self.targets[idx]
            return inputs, targets
        return inputs
    
    def _convert_tensor(self, data: Any) -> torch.Tensor:
        """Convert data to tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, tf.Tensor):
            return torch.from_numpy(data.numpy())
        elif isinstance(data, torch.Tensor):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

class DataLoader:
    """Data loader wrapper."""
    
    def __init__(
        self,
        dataset: Union[BaseDataset, tf.data.Dataset],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        backend: str = "pytorch"
    ):
        """Initialize data loader.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            backend: Backend framework
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.backend = backend
        
        self._init_loader()
        
    def _init_loader(self):
        """Initialize backend-specific loader."""
        if self.backend == "pytorch":
            if isinstance(self.dataset, tf.data.Dataset):
                raise ValueError("Cannot use TensorFlow dataset with PyTorch backend")
            self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers
            )
        elif self.backend == "tensorflow":
            if isinstance(self.dataset, BaseDataset):
                raise ValueError("Cannot use PyTorch dataset with TensorFlow backend")
            self.loader = self.dataset.batch(self.batch_size)
            if self.shuffle:
                self.loader = self.loader.shuffle(buffer_size=10000)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def __iter__(self):
        """Get data loader iterator."""
        return iter(self.loader)
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.loader) 