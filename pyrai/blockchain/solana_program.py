"""Solana program interface for PyRAI blockchain."""

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.transaction import Transaction
from solana.system_program import TransactionInstruction, transfer
from solana.keypair import Keypair
from anchorpy import Program, Provider
import json
from typing import Dict, Any, Optional

class PyRAISolanaProgram:
    """Interface for PyRAI Solana program.
    
    Handles interaction with Solana blockchain for:
    - Model metadata storage
    - Distillation verification
    - Training data provenance
    - Model updates tracking
    """
    
    def __init__(
        self,
        program_id: str,
        rpc_url: str = "https://api.devnet.solana.com"
    ):
        """Initialize Solana program interface.
        
        Args:
            program_id: Solana program ID
            rpc_url: Solana RPC endpoint URL
        """
        self.program_id = program_id
        self.client = AsyncClient(rpc_url, Confirmed)
        self.provider = Provider.local()
        
    async def store_model_metadata(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        payer: Keypair
    ):
        """Store model metadata on Solana blockchain.
        
        Args:
            model_id: Unique model identifier
            metadata: Model metadata to store
            payer: Transaction payer keypair
        """
        instruction = TransactionInstruction(
            program_id=self.program_id,
            data=self._encode_model_metadata(model_id, metadata),
            keys=[]  # Add required account keys
        )
        
        transaction = Transaction().add(instruction)
        await self.client.send_transaction(transaction, payer)
        
    async def verify_distillation(
        self,
        teacher_id: str,
        student_id: str,
        results: Dict[str, Any],
        payer: Keypair
    ):
        """Verify and record distillation results.
        
        Args:
            teacher_id: Teacher model ID
            student_id: Student model ID
            results: Distillation results
            payer: Transaction payer keypair
        """
        instruction = TransactionInstruction(
            program_id=self.program_id,
            data=self._encode_distillation_results(
                teacher_id,
                student_id,
                results
            ),
            keys=[]  # Add required account keys
        )
        
        transaction = Transaction().add(instruction)
        await self.client.send_transaction(transaction, payer)
        
    async def track_data_provenance(
        self,
        data_id: str,
        provenance: Dict[str, Any],
        payer: Keypair
    ):
        """Track training data provenance.
        
        Args:
            data_id: Data identifier
            provenance: Provenance information
            payer: Transaction payer keypair
        """
        instruction = TransactionInstruction(
            program_id=self.program_id,
            data=self._encode_data_provenance(data_id, provenance),
            keys=[]  # Add required account keys
        )
        
        transaction = Transaction().add(instruction)
        await self.client.send_transaction(transaction, payer)
        
    def _encode_model_metadata(
        self,
        model_id: str,
        metadata: Dict[str, Any]
    ) -> bytes:
        """Encode model metadata for Solana instruction."""
        return json.dumps({
            "type": "store_model_metadata",
            "model_id": model_id,
            "metadata": metadata
        }).encode()
        
    def _encode_distillation_results(
        self,
        teacher_id: str,
        student_id: str,
        results: Dict[str, Any]
    ) -> bytes:
        """Encode distillation results for Solana instruction."""
        return json.dumps({
            "type": "verify_distillation",
            "teacher_id": teacher_id,
            "student_id": student_id,
            "results": results
        }).encode()
        
    def _encode_data_provenance(
        self,
        data_id: str,
        provenance: Dict[str, Any]
    ) -> bytes:
        """Encode data provenance for Solana instruction."""
        return json.dumps({
            "type": "track_data_provenance",
            "data_id": data_id,
            "provenance": provenance
        }).encode() 