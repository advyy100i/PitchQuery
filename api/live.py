"""Phase 11: the bridge between the Kafka consumer and connected browsers.

The consumer is synchronous (confluent-kafka's poll loop is blocking C), and
FastAPI is asyncio. So the consumer runs on a worker thread and hands each
message to the event loop with `call_soon_threadsafe`. Running it inside the API
process rather than as a separate service is a deliberate simplification: the
alternative is a second broker between the consumer and the socket, which would
mean two hops of the same message for no benefit on one machine.

Off unless PITCHQUERY_STREAM=1. Redpanda is a development container and the
hosted deployment has neither it nor the memory for it, so the default has to be
that the API works without any of this.

The `replay` label is not decoration. Every message carries `"source":
"replay"`, and the frontend prints it — recorded data presented as live is the
one claim in this project that would be a lie.
"""
import asyncio
import os
import sys
import threading
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STREAM_ENABLED = os.getenv("PITCHQUERY_STREAM", "0").lower() not in ("0", "false", "no")
BOOTSTRAP = os.getenv("PITCHQUERY_KAFKA", "localhost:19092")

# Replayed to a browser that connects late, so a new tab shows the possession in
# progress instead of a blank panel until the next event. Small on purpose: this
# is a viewport, not a history.
BACKLOG = 40


class Hub:
    """Fan-out to every connected WebSocket, plus a short backlog."""

    def __init__(self):
        self.clients: set = set()
        self.backlog: deque = deque(maxlen=BACKLOG)
        self.loop: asyncio.AbstractEventLoop = None
        self.thread: threading.Thread = None
        self.status = "not started"
        self._stop = threading.Event()

    # -- called from the asyncio side ---------------------------------------

    async def connect(self, websocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)
        for message in list(self.backlog):
            await websocket.send_json(message)

    def disconnect(self, websocket) -> None:
        self.clients.discard(websocket)

    async def _fanout(self, message: dict) -> None:
        self.backlog.append(message)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                # A browser tab that closed mid-send. Collect and drop rather
                # than let one dead socket stop the broadcast to the others.
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # -- called from the consumer thread ------------------------------------

    def publish(self, message: dict) -> None:
        message = {**message, "source": "replay"}
        if self.loop is None:
            self.backlog.append(message)
            return
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._fanout(message)))

    # -- lifecycle -----------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> str:
        if not STREAM_ENABLED:
            self.status = "disabled (set PITCHQUERY_STREAM=1 and start Redpanda)"
            return self.status
        try:
            import confluent_kafka  # noqa: F401
        except ImportError:
            self.status = "confluent-kafka is not installed"
            return self.status

        self.loop = loop
        self.thread = threading.Thread(target=self._run, name="pitchquery-live",
                                       daemon=True)
        self.thread.start()
        self.status = f"consuming match_events from {BOOTSTRAP}"
        return self.status

    def _run(self) -> None:
        from stream.consumer import consume

        try:
            consume(bootstrap=BOOTSTRAP, on_message=self.publish)
        except Exception as exc:
            # Never take the API down with it. A broker that is not running is
            # the normal case for this endpoint, not an outage.
            self.status = f"stopped: {type(exc).__name__}: {exc}"
            print(f"live consumer stopped: {exc}")


hub = Hub()
