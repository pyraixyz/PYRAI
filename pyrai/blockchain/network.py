"""P2P network implementation for PyRAI blockchain."""

import asyncio
import json
from typing import Set, Dict, Any, List
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol

from .chain import Blockchain
from .block import Block

class Node:
    """Blockchain network node implementation."""
    
    def __init__(self, host: str, port: int, blockchain: Blockchain):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.peers: Set[str] = set()
        
    async def start(self):
        """Start node server."""
        server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        await server.wait_closed()
        
    async def connect_to_peer(self, peer_address: str):
        """Connect to peer node."""
        try:
            async with websockets.connect(peer_address) as websocket:
                await self.handle_peer(websocket)
        except Exception as e:
            print(f"Failed to connect to peer {peer_address}: {e}")
            
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all peers."""
        for peer in self.peers:
            try:
                async with websockets.connect(peer) as websocket:
                    await websocket.send(json.dumps(message))
            except Exception as e:
                print(f"Failed to broadcast to {peer}: {e}")
                
    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle incoming connection."""
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(websocket, data)
        except Exception as e:
            print(f"Connection error: {e}")
            
    async def handle_peer(self, websocket: WebSocketClientProtocol):
        """Handle peer connection."""
        # Exchange chain data
        await websocket.send(json.dumps({
            "type": "chain_request"
        }))
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(websocket, data)
        except Exception as e:
            print(f"Peer connection error: {e}")
            
    async def handle_message(
        self,
        websocket: WebSocketServerProtocol,
        data: Dict[str, Any]
    ):
        """Handle received message."""
        message_type = data.get("type")
        
        if message_type == "chain_request":
            # Send blockchain data
            await websocket.send(json.dumps({
                "type": "chain_response",
                "chain": [b.to_dict() for b in self.blockchain.chain]
            }))
            
        elif message_type == "chain_response":
            # Validate and potentially update chain
            new_chain = [
                Block(**block_data)
                for block_data in data["chain"]
            ]
            self.blockchain.replace_chain(new_chain)
            
        elif message_type == "new_block":
            # Add new block
            block = Block(**data["block"])
            if block.validate():
                self.blockchain.chain.append(block)
                # Broadcast to other peers
                await self.broadcast({
                    "type": "new_block",
                    "block": block.to_dict()
                }) 