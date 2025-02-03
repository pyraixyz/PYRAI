import asyncio
import aiohttp
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Message:
    """Communication message."""
    sender: str
    receiver: str
    msg_type: str
    payload: Any

class Communicator:
    """Distributed training communicator."""
    
    def __init__(self, node_id: str, nodes: List[str]):
        """Initialize communicator.
        
        Args:
            node_id: Current node ID
            nodes: List of node addresses
        """
        self.node_id = node_id
        self.nodes = nodes
        self.session: Optional[aiohttp.ClientSession] = None
        self._message_queue = asyncio.Queue()
        
    async def start(self):
        """Start communicator."""
        self.session = aiohttp.ClientSession()
        asyncio.create_task(self._message_loop())
        
    async def stop(self):
        """Stop communicator."""
        if self.session:
            await self.session.close()
            
    async def send(self, msg: Message):
        """Send message to node.
        
        Args:
            msg: Message to send
        """
        if not self.session:
            raise RuntimeError("Communicator not started")
            
        url = f"http://{msg.receiver}/message"
        async with self.session.post(url, json=msg.payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to send message: {resp.status}")
                
    async def receive(self) -> Message:
        """Receive message.
        
        Returns:
            Received message
        """
        return await self._message_queue.get()
        
    async def _message_loop(self):
        """Message processing loop."""
        while True:
            try:
                msg = await self._message_queue.get()
                # Process message
                print(f"Received message: {msg}")
            except Exception as e:
                print(f"Message processing error: {e}")
            await asyncio.sleep(0.1) 