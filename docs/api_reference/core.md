# Core API Reference

## PyRAIClient

Main client interface for interacting with PyRAI.

### Methods

#### `__init__(api_key: str, chain: str = "solana")`
Initialize PyRAI client.

**Parameters:**
- `api_key`: API authentication key
- `chain`: Blockchain to use (default: "solana")

#### `train(model: str, dataset: str, config: Dict[str, Any]) -> Task`
Start model training task.

**Parameters:**
- `model`: Model identifier or configuration
- `dataset`: Dataset identifier or configuration
- `config`: Training configuration

**Returns:**
- `Task`: Training task object

#### `deploy(model: str, resources: Dict[str, Any]) -> Deployment`
Deploy trained model.

**Parameters:**
- `model`: Model identifier
- `resources`: Resource requirements

**Returns:**
- `Deployment`: Model deployment object

## DistributedCoordinator

Coordinates distributed training across nodes.

### Methods

#### `register_node(node_id: str, config: NodeConfig)`
Register new training node.

#### `train_distributed(model_config: Dict[str, Any], data_config: Dict[str, Any]) -> Dict[str, Any]`
Execute distributed training. 