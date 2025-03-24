import os
import asyncio
import logging
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

# Disable logging for this module
logging.getLogger(__name__).setLevel(logging.CRITICAL)


class ServerClientListener(WSListener):
    def on_ws_connected(self, transport: WSTransport):
        transport.send(WSMsgType.TEXT, b"Connection established. Listen for messages.")

    def on_ws_frame(self, transport: WSTransport, frame: WSFrame):
        transport.send(frame.msg_type, frame.get_payload_as_bytes())

    def on_ws_closed(
        self, transport: WSTransport, close_code: WSCloseCode, close_reason: str
    ):
        transport.close()


async def main():
    def listener_factory(r: WSUpgradeRequest):
        return ServerClientListener()

    websocket_port = int(os.environ.get("WEBSOCKET_PORT", 9090))
    host = "127.0.0.1"

    try:
        server = await ws_create_server(listener_factory, host, websocket_port)
    except OSError as e:
        alternative_port = websocket_port + 1
        server = await ws_create_server(listener_factory, host, alternative_port)

    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
