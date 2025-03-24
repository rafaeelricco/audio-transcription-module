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

    server = await ws_create_server(listener_factory, "127.0.0.1", 9001)
    for s in server.sockets:
        Logger.log(True, f"Server started on ws://127.0.0.1:9001")

    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
