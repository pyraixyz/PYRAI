"""Block implementation for PyRAI blockchain."""

import hashlib
import json
from time import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class Block:
    """Block structure for PyRAI blockchain.
    
    Stores model metadata, training data, and distillation results
    in a tamper-proof way using cryptographic hashing.
    """
    
    index: int
    timestamp: float
    transactions: List[Dict[str, Any]]
    previous_hash: str
    nonce: int = 0
    hash: Optional[str] = None
    
    def __post_init__(self):
        """Initialize block hash if not provided."""
        if self.hash is None:
            self.hash = self.calculate_hash()
            
    def calculate_hash(self) -> str:
        """Calculate block hash using SHA-256."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
        
    def mine_block(self, difficulty: int):
        """Mine block by finding nonce that satisfies difficulty."""
        while self.hash[:difficulty] != "0" * difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()
            
    def validate(self) -> bool:
        """Validate block hash."""
        return self.hash == self.calculate_hash()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        } 