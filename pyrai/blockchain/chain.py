"""Blockchain implementation for PyRAI."""

from typing import List, Dict, Any, Optional
from time import time
import json

from .block import Block
from .consensus import ProofOfWork

class Blockchain:
    """PyRAI blockchain implementation.
    
    Provides decentralized storage and verification for:
    - Model metadata
    - Training data provenance
    - Distillation results
    - Model updates
    """
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.difficulty = difficulty
        self.consensus = ProofOfWork(difficulty)
        
        # Create genesis block
        self.create_genesis_block()
        
    def create_genesis_block(self):
        """Create and add genesis block."""
        genesis_block = Block(
            index=0,
            timestamp=time(),
            transactions=[],
            previous_hash="0"
        )
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        
    def get_latest_block(self) -> Block:
        """Get last block in chain."""
        return self.chain[-1]
        
    def add_transaction(self, transaction: Dict[str, Any]):
        """Add new transaction to pending transactions."""
        self.pending_transactions.append(transaction)
        
    def mine_pending_transactions(self, miner_address: str):
        """Create new block with pending transactions and mine it."""
        block = Block(
            index=len(self.chain),
            timestamp=time(),
            transactions=self.pending_transactions,
            previous_hash=self.get_latest_block().hash
        )
        
        # Mine block
        block.mine_block(self.difficulty)
        
        # Add block to chain
        self.chain.append(block)
        
        # Clear pending transactions and reward miner
        self.pending_transactions = [{
            "from": "network",
            "to": miner_address,
            "amount": 1.0  # Mining reward
        }]
        
    def is_chain_valid(self) -> bool:
        """Validate entire blockchain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Validate current block
            if not current.validate():
                return False
                
            # Validate chain link
            if current.previous_hash != previous.hash:
                return False
                
        return True
        
    def replace_chain(self, new_chain: List[Block]):
        """Replace chain with new longer valid chain."""
        if len(new_chain) > len(self.chain) and self.is_valid_chain(new_chain):
            self.chain = new_chain
            
    def get_model_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get history of model updates and training."""
        history = []
        for block in self.chain:
            for tx in block.transactions:
                if tx.get("model_id") == model_id:
                    history.append({
                        "timestamp": block.timestamp,
                        "action": tx.get("action"),
                        "data": tx.get("data")
                    })
        return history
        
    def save_chain(self, filepath: str):
        """Save blockchain to file."""
        with open(filepath, "w") as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2)
            
    @classmethod
    def load_chain(cls, filepath: str) -> "Blockchain":
        """Load blockchain from file."""
        blockchain = cls()
        with open(filepath) as f:
            chain_data = json.load(f)
            blockchain.chain = [
                Block(**block_data)
                for block_data in chain_data
            ]
        return blockchain 