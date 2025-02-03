# Core Concepts

## Distributed Training

PyRAI uses a distributed training architecture that combines:

- Data Parallelism
- Model Parallelism
- Pipeline Parallelism

### Data Parallelism
Each node processes different batches of data in parallel, then synchronizes gradients.

### Model Parallelism
Large models are split across multiple devices, with each device handling different layers.

### Pipeline Parallelism
Different stages of the model are processed in parallel on different devices.

## Resource Management

PyRAI manages compute resources through:

### Resource Tokenization
Computing resources are tokenized on the blockchain, allowing:
- Resource trading
- Usage tracking
- Cost optimization

### Auto-scaling
Dynamic resource allocation based on:
- Model complexity
- Batch size
- Training progress
- Resource utilization

## Blockchain Integration

### Smart Contracts
PyRAI uses Solana smart contracts for:
- Model verification
- Training certification
- Resource allocation
- Payment processing

### Cross-chain Operations
Support for multiple blockchains:
- Solana (primary)
- Ethereum
- Other EVM-compatible chains 