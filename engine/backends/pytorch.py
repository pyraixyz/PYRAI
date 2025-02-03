import torch
from typing import Any, Dict, Optional
from ..core.base import ModelBackend, Trainer

class PyTorchBackend(ModelBackend):
    """PyTorch implementation of ModelBackend."""
    
    def load_model(self, path: str) -> torch.nn.Module:
        return torch.load(path)
    
    def save_model(self, model: torch.nn.Module, path: str) -> None:
        torch.save(model, path)
    
    def predict(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(inputs)

class PyTorchTrainer(Trainer):
    """PyTorch implementation of Trainer."""
    
    def __init__(self, 
                 optimizer_cls: Any = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.device = device
    
    def train(self,
              model: torch.nn.Module,
              train_data: torch.utils.data.DataLoader,
              valid_data: Optional[torch.utils.data.DataLoader] = None,
              epochs: int = 10,
              criterion: Any = torch.nn.CrossEntropyLoss(),
              **kwargs) -> Dict[str, Any]:
        model = model.to(self.device)
        optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_kwargs)
        
        history = {'train_loss': [], 'valid_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_data):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_data)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if valid_data is not None:
                model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for data, target in valid_data:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = criterion(output, target)
                        valid_loss += loss.item()
                
                avg_valid_loss = valid_loss / len(valid_data)
                history['valid_loss'].append(avg_valid_loss)
        
        return history
    
    def evaluate(self,
                model: torch.nn.Module,
                test_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        model = model.to(self.device)
        model.eval()
        
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_data)
        accuracy = correct / len(test_data.dataset)
        
        return {
            'test_loss': test_loss,
            'accuracy': accuracy
        } 