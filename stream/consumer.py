"""Phase 11: rebuild possessions from the replayed stream, one event at a time.

The batch pipeline groups a whole match into possessions after the fact, which
is easy — every event is already there. Doing it incrementally is the part worth
building: the possession is only known to be over when the next one starts, so
the consumer has to hold state, emit a token as each event arrives, and close
the passage on a turnover.

Three pieces of the batch pipeline are reused rather than reimplemented, which
is the point of the exercise:

  * `core.zones.token` writes the same token the index is built from, so a
    passage watched live and the same passage searched later are the same
    string;
  * `core.features.shot_row` derives the same geometry;
  * `core.xg.XGModel` is the same 599 KB artefact the API serves.

A second implementation of any of them would be a second thing to keep correct,
and the streaming version would drift first because nothing tests it.

Run standalone (prints to the terminal):
  python stream/consumer.py

Or let the API run it and push to /live — see api/main.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.features import shot_row  # noqa: E402
from core.zones import MIN_EVENTS, token as grammar_token  # noqa: E402

TOPIC = "match_events"
BOOTSTRAP = "localhost:19092"
GROUP = "pitchquery-live"

REPLAY_END = "__replay_end__"


class PossessionBuilder:
    """Incremental possession state for one match.

    `feed()` returns a list of messages to broadcast. Everything it emits is
    already JSON-serialisable, so the WebSocket layer does no thinking.
    """

    def __init__(self, xg=None):
        self.xg = xg
        self.reset()

    def reset(self) -> None:
        self.possession = None
        self.team = None
        self.opponent = None
        self.tokens = []
        self.zones = []
        self.shots = []
        self.started_at = None
        self.last_at = None
        self.closed = 0

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _seconds(ev: dict) -> float:
        return (ev.get("minute") or 0) * 60 + (ev.get("second") or 0)

    def _score(self, ev: dict):
        """This project's xG for a shot event, or None with the reason."""
        row = shot_row(ev)
        if row is None:
            return None, "not a shot with a location"
        if self.xg is None:
            return None, "xG model not loaded"
        value = self.xg.predict_one(row)
        if value is None:
            return None, ("penalties are excluded from training"
                          if row.get("shot_type") == "Penalty"
                          else "no shot geometry recorded")
        return float(value), None

    def _snapshot(self) -> dict:
        return {
            "possession": self.possession,
            "team": self.team,
            "opponent": self.opponent,
            "tokens": list(self.tokens),
            "token_string": " ".join(self.tokens),
            "zone_path": " ".join(self.zones),
            "n_tokens": len(self.tokens),
            "duration_s": round((self.last_at or 0) - (self.started_at or 0), 1),
            "shots": list(self.shots),
            "my_xg_sum": round(sum(s["my_xg"] for s in self.shots
                                   if s["my_xg"] is not None), 4) or None,
        }

    def _close(self) -> list:
        """Emit the finished possession, if it was long enough to be one.

        Same MIN_EVENTS threshold the batch builder uses. Without it the live
        panel fills with two-token throw-in fragments, which is exactly the
        noise `ingest/04_build_possessions.py` drops for the index.
        """
        if self.possession is None or not self.tokens:
            return []
        out = self._snapshot()
        out["type"] = "possession_closed"
        out["kept"] = len(self.tokens) >= MIN_EVENTS
        self.closed += 1
        return [out]

    # -- the one public method -----------------------------------------------

    def feed(self, ev: dict) -> list:
        etype = (ev.get("type") or {}).get("name")
        if etype == REPLAY_END:
            msgs = self._close()
            self.reset()
            return msgs + [{"type": "replay_end"}]

        pid = ev.get("possession")
        if pid is None:
            return []

        messages = []
        if pid != self.possession:
            # The turnover. A possession is only known to be over when the next
            # one starts — there is no end-of-possession event to wait for.
            messages += self._close()
            self.possession = pid
            self.team = (ev.get("possession_team") or {}).get("name")
            self.tokens, self.zones, self.shots = [], [], []
            self.started_at = self._seconds(ev)
            messages.append({"type": "possession_opened", "possession": pid,
                             "team": self.team,
                             "minute": ev.get("minute"), "second": ev.get("second")})

        self.last_at = self._seconds(ev)
        acting = (ev.get("team") or {}).get("name")
        if acting and acting != self.team:
            # The defence intervening. Recorded as the opponent's name and then
            # dropped, for the same reason the batch builder drops it:
            # interleaving both teams' events destroys n-gram similarity.
            self.opponent = acting
            return messages

        tok = grammar_token(ev)
        if tok is None:
            return messages

        self.tokens.append(tok)
        self.zones.append(tok.split("@", 1)[1].rstrip("+>^"))

        if etype == "Shot":
            value, note = self._score(ev)
            self.shots.append({
                "player": (ev.get("player") or {}).get("name"),
                "minute": ev.get("minute"),
                "my_xg": round(value, 4) if value is not None else None,
                "note": note,
                "outcome": ((ev.get("shot") or {}).get("outcome") or {}).get("name"),
            })

        messages.append({**self._snapshot(), "type": "token", "token": tok,
                         "minute": ev.get("minute"), "second": ev.get("second"),
                         "player": (ev.get("player") or {}).get("name")})
        return messages


def load_xg():
    try:
        from core.xg import XGModel

        return XGModel.load()
    except Exception as exc:
        print(f"xG model unavailable ({type(exc).__name__}: {exc}) — "
              f"live possessions will show tokens without a chance value")
        return None


def consume(*, topic: str = TOPIC, bootstrap: str = BOOTSTRAP, group: str = GROUP,
            on_message=None, timeout: float = 1.0, from_start: bool = True):
    """Consume forever, calling `on_message(dict)` for everything the builder emits.

    A generator would be tidier, but the API drives this from an asyncio task
    and a callback is what fits there without a second thread queue.
    """
    from confluent_kafka import Consumer

    builder = PossessionBuilder(load_xg())
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group,
        # A replay is only interesting from the kick-off, and a demo restarted
        # after a crash should show the match again rather than resume in the
        # 70th minute against an empty screen.
        "auto.offset.reset": "earliest" if from_start else "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([topic])
    try:
        while True:
            msg = consumer.poll(timeout)
            if msg is None:
                continue
            if msg.error():
                print(f"kafka: {msg.error()}")
                continue
            ev = json.loads(msg.value())
            for out in builder.feed(ev):
                if on_message:
                    on_message(out)
    finally:
        consumer.close()


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--latest", action="store_true",
                    help="start at the end of the topic instead of the kick-off")
    args = ap.parse_args()

    def show(m):
        if m["type"] == "token":
            xg = f"  xG {m['my_xg_sum']:.3f}" if m.get("my_xg_sum") else ""
            print(f"  {m['minute']:>3}'  {m['team'][:18]:<18} "
                  f"{m['n_tokens']:>2} tokens  {m['token']}{xg}")
        elif m["type"] == "possession_closed" and m["kept"]:
            print(f"      ---- closed: {m['team']}, {m['n_tokens']} tokens, "
                  f"{m['duration_s']:.0f}s"
                  + (f", xG {m['my_xg_sum']:.3f}" if m.get("my_xg_sum") else "")
                  + " ----")
        elif m["type"] == "replay_end":
            print("      ==== replay finished ====")

    print("watching the replay (this is recorded data, not a live feed)")
    consume(topic=args.topic, bootstrap=args.bootstrap, group=args.group,
            on_message=show, from_start=not args.latest)


if __name__ == "__main__":
    cli()
