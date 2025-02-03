"""Consensus mechanisms for PyRAI blockchain."""

from abc import ABC, abstractmethod
from typing import List, Dict
import hashlib
import time

from .block import Block

class ConsensusProtocol(ABC):
    """Base class for consensus protocols."""
    
    @abstractmethod
    def validate_block(self, block: Block) -> bool:
        """Validate block according to consensus rules."""
        pass
        
    @abstractmethod
    def validate_chain(self, chain: List[Block]) -> bool:
        """Validate entire blockchain."""
        pass

class ProofOfWork(ConsensusProtocol):
    """Proof of Work consensus implementation."""
    
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty
        
    def validate_block(self, block: Block) -> bool:
        """Validate block hash meets difficulty requirement."""
        return (
            block.hash.startswith("0" * self.difficulty) and
            block.hash == block.calculate_hash()
        )
        
    def validate_chain(self, chain: List[Block]) -> bool:
        """Validate entire blockchain."""
        for i in range(1, len(chain)):
            current = chain[i]
            previous = chain[i-1]
            
            if not self.validate_block(current):
                return False
                
            if current.previous_hash != previous.hash:
                return False
                
        return True

class ProofOfStake(ConsensusProtocol):
    """Proof of Stake consensus implementation."""
    
    def __init__(self, min_stake: float = 100.0):
        self.min_stake = min_stake
        self.stakes: Dict[str, float] = {}
        
    def add_stake(self, address: str, amount: float):
        """Add stake for validator."""
        self.stakes[address] = self.stakes.get(address, 0) + amount
        
    def validate_block(self, block: Block) -> bool:
        """Validate block creator has sufficient stake."""
        creator = block.transactions[0].get("from")
        return self.stakes.get(creator, 0) >= self.min_stake
        
    def validate_chain(self, chain: List[Block]) -> bool:
        """Validate entire blockchain."""
        for block in chain[1:]:
            if not self.validate_block(block):
                return False
        return True 