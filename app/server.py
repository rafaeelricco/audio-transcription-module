import os
import asyncio
from picows import (
    ws_create_server,
    WSFrame,
    WSTransport,
    WSListener,
    WSCloseCode,
    WSMsgType,
    WSUpgradeRequest,
)
from app.utils.logger import Logger


class ServerClientListener(WSListener):
    def on_ws_connected(self, transport: WSTransport):
        transport.send(WSMsgType.TEXT, b"Connection established. Listen for messages.")

    def on_ws_frame(self, transport: WSTransport, frame: WSFrame):
        Logger.log(True, f"Received: {frame.get_payload_as_ascii_text()}")
        transport.send(frame.msg_type, frame.get_payload_as_bytes())

    def on_ws_closed(
        self, transport: WSTransport, close_code: WSCloseCode, close_reason: str
    ):
        Logger.log(True, f"Connection closed: {close_code} - {close_reason}")
        transport.close()


async def main():
    def listener_factory(r: WSUpgradeRequest):
        return ServerClientListener()
    
    # Get WebSocket port from environment variables or use default
    websocket_port = int(os.environ.get('WEBSOCKET_PORT', 9090))
    host = "127.0.0.1"
    
    try:
        server = await ws_create_server(listener_factory, host, websocket_port)
        for s in server.sockets:
            Logger.log(True, f"Server started on ws://{host}:{websocket_port}")
    except OSError as e:
        Logger.log(True, f"Error starting WebSocket server: {e}")
        # Try alternative port if specified one is in use
        alternative_port = websocket_port + 1
        Logger.log(True, f"Trying alternative port: {alternative_port}")
        server = await ws_create_server(listener_factory, host, alternative_port)
        for s in server.sockets:
            Logger.log(True, f"Server started on ws://{host}:{alternative_port}")

    # Always serve forever once we have a server
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
