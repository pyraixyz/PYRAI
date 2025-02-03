import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pyrai import Engine
from pyrai.config import Config
from pyrai.backends import PyTorchBackend
from pyrai.metrics import Accuracy, Precision, MetricCollection

class ResNetClassifier(nn.Module):
    """ResNet-based image classifier."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Use pretrained ResNet18
        self.resnet = torchvision.models.resnet18(pretrained=True)
        
        # Replace final layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def main():
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=val_transform
    )
    
    # Create engine
    config = Config({
        "training": {
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 0.001,
            "optimizer": {
                "name": "adamw",
                "weight_decay": 0.01
            },
            "scheduler": {
                "name": "one_cycle",
                "max_lr": 0.001,
                "epochs": 20,
                "steps_per_epoch": len(train_dataset) // 32
            }
        },
        "model": {
            "type": "pytorch",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_dir": "models/cifar10"
        }
    })
    
    engine = Engine(
        backend=PyTorchBackend(),
        config=config
    )
    
    # Create model and metrics
    model = ResNetClassifier(num_classes=10)
    metrics = MetricCollection({
        "accuracy": Accuracy(),
        "precision": Precision(average="macro")
    })
    
    # Train model
    history = engine.train(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        metrics=metrics
    )
    
    print("\nTraining history:")
    for metric, values in history.items():
        print(f"{metric}: {values[-1]:.4f}")
    
    # Deploy model
    deployment = engine.deploy(
        model=model,
        model_id="cifar10",
        version="v1"
    )
    
    # Test predictions
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False
    )
    
    total_correct = 0
    total_samples = 0
    
    for images, labels in val_loader:
        predictions = deployment.predict(images)
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Stop deployment
    deployment.stop()

if __name__ == "__main__":
    main() 