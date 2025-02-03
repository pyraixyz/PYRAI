# Blockchain Integration Tutorial

This tutorial explains how to integrate blockchain features in your AI training workflow.

## Prerequisites

- PyRAI installed
- Solana CLI tools
- Solana wallet with devnet tokens

## Setup Blockchain Environment

1. Create Solana wallet:
```bash
solana-keygen new -o wallet.json
```

2. Configure PyRAI:
```python
from pyrai import PyRAIClient

client = PyRAIClient(
    api_key="your-api-key",
    chain="solana"
)

client.configure_chain(
    network="devnet",
    wallet="wallet.json"
)
```

## Resource Tokenization

1. Mint compute tokens:
```python
tokens = client.mint_compute_tokens(
    cpu_hours=100,
    gpu_hours=50
)
```

2. Check token balance:
```python
balance = client.get_token_balance()
print(f"Available compute: {balance}")
```

## Smart Contracts

1. Deploy model verification contract:
```python
contract = client.deploy_contract(
    "model_verification",
    {
        "min_accuracy": 0.9,
        "max_model_size": 1000  # MB
    }
)
```

2. Verify model updates:
```python
result = contract.verify_update(
    model_id="model1",
    update_data={
        "accuracy": 0.95,
        "model_size": 800
    }
)
```

## Cross-Chain Operations

1. Bridge tokens between chains:
```python
client.bridge_tokens(
    from_chain="solana",
    to_chain="ethereum",
    amount=100
)
``` 