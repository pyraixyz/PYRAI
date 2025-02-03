# Training Guide

## Basic Training

### Single GPU Training

```python
from pyrai import Engine
from pyrai.backends import PyTorchBackend

# Initialize engine
engine = Engine(
    backend=PyTorchBackend(),
    config={
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "optimizer": {
                "type": "adam",
                "lr": 0.001
            }
        }
    }
)

# Train model
history = engine.train(
    model=model,
    train_data=train_data,
    validation_data=val_data
)
```

### Multi-GPU Training

```python
# Configure for multi-GPU
config = {
    "backend": {
        "type": "pytorch",
        "device": "cuda"
    },
    "distributed": {
        "enabled": True,
        "backend": "nccl",
        "world_size": torch.cuda.device_count()
    }
}

# Train with multiple GPUs
engine.train(
    model=model,
    train_data=train_data,
    distributed=True
)
```

## Advanced Training

### Custom Training Loop

```python
@engine.training_loop
def custom_training(model, data_loader, optimizer):
    for epoch in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            
# Use custom training loop
engine.train(
    model=model,
    train_data=train_data,
    training_loop=custom_training
)
```

### Callbacks

```python
from pyrai.callbacks import ModelCheckpoint, EarlyStopping

callbacks = [
    ModelCheckpoint(
        monitor='val_loss',
        save_best_only=True
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=3
    )
]

engine.train(
    model=model,
    train_data=train_data,
    callbacks=callbacks
)
``` 