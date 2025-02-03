# Blockchain API Reference

## Solana Integration

### SolanaAdapter

Interface for Solana blockchain operations.

#### Methods

##### `__init__(network: str = "devnet", wallet_path: Optional[str] = None)`
Initialize Solana adapter.

**Parameters:**
- `network`: Solana network ("mainnet", "testnet", "devnet")
- `wallet_path`: Path to wallet keypair file

##### `deploy_contract(program_id: str, config: Dict[str, Any]) -> str`
Deploy smart contract to Solana.

**Parameters:**
- `program_id`: Program identifier
- `config`: Contract configuration

**Returns:**
- Contract address

## Resource Tokenization

### ResourceToken

Manages compute resource tokenization.

#### Methods

##### `mint_tokens(amount: float, resource_type: str) -> str`
Mint new compute tokens.

**Parameters:**
- `amount`: Token amount
- `resource_type`: Resource type ("CPU", "GPU", "MEMORY")

**Returns:**
- Transaction signature

## Cross-chain Bridge

### ChainBridge

Manages cross-chain token transfers.

#### Methods

##### `bridge_tokens(from_chain: str, to_chain: str, amount: float)`
Transfer tokens between chains.

**Parameters:**
- `from_chain`: Source chain
- `to_chain`: Target chain
- `amount`: Token amount 