"""Phase 11: replay one recorded match onto a Kafka topic, in match time.

This is a REPLAY of events that were recorded years ago. It is not a live feed,
there is no live feed, and every surface that shows it says so. StatsBomb open
data is published after the fact; claiming a live pipeline over it is the
fastest way to lose an interview, and the replay is worth exactly as much
without the claim — the consumer, the incremental possession state and the
per-shot xG all do the same work either way.

Keyed by match_id, so all of one match lands on one partition and arrives in
order. Ordering is the whole point here: the consumer closes a possession when
the possession id changes, and events out of order would open and close the
same possession repeatedly.

Run:
  docker compose --profile stream up -d
  python stream/producer.py --match 3869685 --speed 60
  python stream/producer.py --match 3869685 --speed 0     # as fast as possible
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import RAW_DIR  # noqa: E402

TOPIC = "match_events"
BOOTSTRAP = "localhost:19092"

# Fields the consumer actually reads. Publishing the whole StatsBomb event would
# put the freeze frame — up to 21 players — on every shot message, and the
# consumer scores xG from the columns core/features.py derives, not from the
# raw frame. Trimming here keeps a message around 400 bytes.
KEEP = ("id", "index", "period", "minute", "second", "type", "possession",
        "possession_team", "team", "player", "position", "play_pattern",
        "location", "under_pressure", "duration", "pass", "carry", "shot",
        "dribble")


def seconds(ev: dict) -> float:
    return (ev.get("minute") or 0) * 60 + (ev.get("second") or 0)


def trim(ev: dict) -> dict:
    out = {k: ev[k] for k in KEEP if k in ev}
    # The freeze frame is the largest thing in a shot event and the only feature
    # the consumer needs from it is already computed by core/features.py, which
    # runs consumer-side on the location fields. Keep it: n_def_in_cone comes
    # out of it and the xG model reads that column.
    return out


def load_match(match_id: int) -> list:
    path = RAW_DIR / "events" / f"{match_id}.json"
    if not path.exists():
        raise SystemExit(
            f"no cached events for match {match_id} at {path}\n"
            f"  fetch it first:  python ingest/02_fetch.py --comp <id>:<season>")
    events = json.loads(path.read_text(encoding="utf-8"))
    return sorted(events, key=lambda e: e.get("index", 0))


def produce(match_id: int, *, speed: float = 60.0, topic: str = TOPIC,
            bootstrap: str = BOOTSTRAP, limit: int = None,
            progress_every: int = 250) -> dict:
    """Publish one match. `speed` is the multiple of real time; 0 means no sleeps."""
    from confluent_kafka import Producer

    events = load_match(match_id)
    if limit:
        events = events[:limit]

    producer = Producer({
        "bootstrap.servers": bootstrap,
        "client.id": f"pitchquery-replay-{match_id}",
        # Small batches: this is a demo where a message that arrives half a
        # second late is visible on screen, not a throughput problem.
        "linger.ms": 5,
    })
    key = str(match_id).encode()

    failures = []

    def delivery(err, _msg):
        if err is not None:
            failures.append(str(err))

    t0 = time.time()
    previous = None
    for i, ev in enumerate(events, 1):
        now = seconds(ev)
        if previous is not None and speed > 0:
            # Real match time divided by the speed factor. Clamped at 5 s so a
            # half-time break does not stall the replay for 15 real minutes.
            gap = min(max(now - previous, 0.0), 5.0 * speed) / speed
            if gap > 0:
                time.sleep(gap)
        previous = now

        producer.produce(topic, key=key,
                         value=json.dumps(trim(ev)).encode("utf-8"),
                         callback=delivery)
        producer.poll(0)
        if i % progress_every == 0:
            print(f"  {i}/{len(events)} events  "
                  f"({ev.get('minute')}:{(ev.get('second') or 0):02d} match time, "
                  f"{time.time() - t0:.0f}s wall)")

    # Mark the end of the match explicitly. Without it the consumer cannot tell
    # a finished replay from a producer that stopped, and the last possession
    # would sit open forever.
    producer.produce(topic, key=key,
                     value=json.dumps({"type": {"name": "__replay_end__"},
                                       "match_id": match_id}).encode("utf-8"))
    producer.flush(30)

    if failures:
        raise RuntimeError(f"{len(failures)} messages failed to deliver: "
                           f"{failures[:3]}")
    took = time.time() - t0
    print(f"published {len(events):,} events for match {match_id} in {took:.0f}s "
          f"({'as fast as possible' if speed <= 0 else f'{speed:g}x match time'})")
    return {"rows": len(events), "match_id": match_id, "seconds": round(took, 1)}


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--match", type=int, required=True, help="a cached match id")
    ap.add_argument("--speed", type=float, default=60.0,
                    help="multiple of real time; 0 replays with no sleeps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    args = ap.parse_args()
    produce(args.match, speed=args.speed, topic=args.topic,
            bootstrap=args.bootstrap, limit=args.limit)


if __name__ == "__main__":
    cli()
