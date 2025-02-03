"""WebSocket service implementation."""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio

class WebSocketManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, client_id: str, websocket: WebSocket):
        """Handle new WebSocket connection."""
        await websocket.accept()
        if client_id not in self.connections:
            self.connections[client_id] = set()
        self.connections[client_id].add(websocket)
        
    async def disconnect(self, client_id: str, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        self.connections[client_id].remove(websocket)
        
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections."""
        for connections in self.connections.values():
            for websocket in connections:
                await websocket.send_json(message) 