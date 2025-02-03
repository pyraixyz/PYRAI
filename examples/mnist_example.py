import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pyrai import Engine
from pyrai.config import Config
from pyrai.backends import PyTorchBackend
from pyrai.data import TensorDataset
from pyrai.metrics import Accuracy, MetricCollection

class MNISTModel(nn.Module):
    """Simple MNIST classifier."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True,
                                transform=transform)
    test_dataset = datasets.MNIST('data', train=False,
                               transform=transform)
    
    # Create engine
    config = Config({
        "training": {
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": {
                "name": "adam",
                "weight_decay": 1e-5
            },
            "scheduler": {
                "name": "cosine",
                "T_max": 10
            }
        },
        "model": {
            "type": "pytorch",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_dir": "models/mnist"
        }
    })
    
    engine = Engine(
        backend=PyTorchBackend(),
        config=config
    )
    
    # Create model and metrics
    model = MNISTModel()
    metrics = MetricCollection({
        "accuracy": Accuracy()
    })
    
    # Train model
    history = engine.train(
        model=model,
        train_data=train_dataset,
        val_data=test_dataset,
        metrics=metrics
    )
    
    print("Training history:", history)
    
    # Deploy model
    deployment = engine.deploy(
        model=model,
        model_id="mnist",
        version="v1"
    )
    
    # Test predictions
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )
    
    images, labels = next(iter(test_loader))
    predictions = deployment.predict(images)
    accuracy = (predictions.argmax(dim=1) == labels).float().mean()
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Stop deployment
    deployment.stop()

if __name__ == "__main__":
    main() 