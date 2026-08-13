"""The zone grid and the token grammar (plan §5).

This is the compression step: a possession becomes one line of text that a
sparse n-gram ranker can match on. Too fine and nothing matches, too coarse and
everything matches. Changing anything in here invalidates every stored
token_string, so re-run 04_build_possessions.py and 05_embed.py after edits.

Grid: 3 bands along x (40m each) x 5 channels across y (16m each) = 15 zones.
Coordinates assume the attacking direction verified in docs/probes.md — the
acting team always attacks towards x=120.

    x=0          x=40         x=80        x=120
y=0  +------------+------------+------------+
     |    D-L     |    M-L     |    F-L     |   left touchline
y=16 +------------+------------+------------+
     |    D-LI    |    M-LI    |    F-LI    |   left inside channel
y=32 +------------+------------+------------+
     |    D-C     |    M-C     |    F-C     |  <- goal attacked
y=48 +------------+------------+------------+
     |    D-RI    |    M-RI    |    F-RI    |   right inside channel
y=64 +------------+------------+------------+
     |    D-R     |    M-R     |    F-R     |   right touchline
y=80 +------------+------------+------------+
"""
from typing import Optional

BANDS = ("D", "M", "F")                    # x: 0-40, 40-80, 80-120
CHANNELS = ("L", "LI", "C", "RI", "R")     # y: 0-16, 16-32, 32-48, 48-64, 64-80

BAND_WIDTH = 40.0
CHANNEL_WIDTH = 16.0

# Modifier thresholds. Tuned against eval/queries.yaml in Phase 6 — record the
# before/after P@5 when you change them.
PROGRESSIVE_M = 12.0        # '+' : end_x - x at least this, or a band crossed
BOX_X = 102.0               # '>' : ends inside the 18-yard box
BOX_Y = (18.0, 62.0)
SWITCH_DY = 30.0            # SWITCH: lateral travel at least this,
SWITCH_DX = 20.0            #         with forward travel under this
MIN_EVENTS = 3              # possessions shorter than this are throw-in noise


def band_index(x: float) -> int:
    return max(0, min(int(x // BAND_WIDTH), 2))


def band(x: float) -> str:
    return BANDS[band_index(x)]


def channel(y: float) -> str:
    return CHANNELS[max(0, min(int(y // CHANNEL_WIDTH), 4))]


def zone(x: float, y: float) -> str:
    """'F-LI' etc. Out-of-range coordinates clamp to the edge zones."""
    return f"{band(x)}-{channel(y)}"


# --- action vocabulary -------------------------------------------------------
# Ordered derivation: the first rule that fires wins (plan §5).

ACTIONS = ("PASS", "CROSS", "THROUGH", "SWITCH", "CARRY", "DRIB", "SHOT",
           "RECV", "RECOV", "DUEL", "CLR", "INT", "LOSS", "SETP")

_SIMPLE = {
    "Carry": "CARRY",
    "Dribble": "DRIB",
    "Shot": "SHOT",
    "Ball Receipt*": "RECV",
    "Ball Recovery": "RECOV",
    "Duel": "DUEL",
    "Clearance": "CLR",
    "Interception": "INT",
    "Miscontrol": "LOSS",
    "Dispossessed": "LOSS",
}

# Event types that carry no tactical information for retrieval. Dropping them
# keeps n-grams dense — 'Pressure' alone is 8% of all events, and it is always
# the defending team anyway.
_SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}

_IGNORED = {
    "Pressure", "Starting XI", "Half Start", "Half End", "Substitution",
    "Tactical Shift", "Injury Stoppage", "Referee Ball-Drop", "Camera On",
    "Camera off", "Player On", "Player Off", "Shield", "Error",
}


def action(ev: dict) -> Optional[str]:
    """Map one StatsBomb event to a token action, or None to drop it."""
    etype = (ev.get("type") or {}).get("name")
    if etype in _IGNORED:
        return None

    if etype == "Pass":
        p = ev.get("pass") or {}
        # Set-piece restarts first. Without this a corner reads as SWITCH: the
        # delivery travels far across y and barely at all along x, which is
        # exactly the switch-of-play test. play_pattern keeps the finer
        # distinction (From Corner vs From Free Kick) as a hard filter.
        if (p.get("type") or {}).get("name") in _SET_PIECES:
            return "SETP"
        if p.get("cross"):
            return "CROSS"
        if (p.get("technique") or {}).get("name") == "Through Ball":
            return "THROUGH"
        loc, end = ev.get("location") or [], p.get("end_location") or []
        if len(loc) >= 2 and len(end) >= 2:
            dy = abs(float(end[1]) - float(loc[1]))
            dx = abs(float(end[0]) - float(loc[0]))
            if dy >= SWITCH_DY and dx < SWITCH_DX:
                return "SWITCH"
        return "PASS"

    return _SIMPLE.get(etype)


def modifiers(action_name: str, ev: dict, x: float, y: float,
              end_x: Optional[float], end_y: Optional[float]) -> str:
    """'+' progressive, '>' ends in the box, '^' under pressure. Order is fixed
    so the same passage always produces the same string.

    Shots get '^' only. A shot's end_location is where the ball finished, so
    '+' and '>' would encode whether it flew towards the goal — that is the
    outcome leaking into the token, and it splinters the single most important
    token in the grammar into variants that no longer match each other.
    """
    m = ""
    if end_x is not None and action_name != "SHOT":
        # "Progressive" is directional: a long ball backwards is not progress,
        # so compare band indices rather than testing for any band change.
        if end_x - x >= PROGRESSIVE_M or band_index(end_x) > band_index(x):
            m += "+"
        if end_y is not None and end_x >= BOX_X and BOX_Y[0] <= end_y <= BOX_Y[1]:
            m += ">"
    if ev.get("under_pressure"):
        m += "^"
    return m


def token(ev: dict) -> Optional[str]:
    """'CROSS@F-R>' for one event, or None if the event carries no signal."""
    act = action(ev)
    if act is None:
        return None
    loc = ev.get("location") or []
    if len(loc) < 2:
        return None
    x, y = float(loc[0]), float(loc[1])

    key = {"Pass": "pass", "Carry": "carry", "Shot": "shot"}.get(
        (ev.get("type") or {}).get("name"))
    end = ((ev.get(key) or {}).get("end_location") or []) if key else []
    end_x = float(end[0]) if len(end) >= 2 else None
    end_y = float(end[1]) if len(end) >= 2 else None

    return f"{act}@{zone(x, y)}{modifiers(act, ev, x, y, end_x, end_y)}"


def tsv_word(tok: str) -> str:
    """Rewrite a token as a single Postgres lexeme.

    Postgres' text-search parser splits on '@', '-', '+' and friends, so
    'CROSS@F-R>' would become several useless lexemes. Fold the punctuation into
    the word instead: 'CROSS@F-R>^' -> 'cross_f_r_box_prs'. The Python-side
    TF-IDF ranker reads the original token_string; only token_tsv uses this.
    """
    return (tok.lower()
            .replace("@", "_").replace("-", "_")
            .replace("+", "_prg").replace(">", "_box").replace("^", "_prs"))


def tsv_string(tokens) -> str:
    return " ".join(tsv_word(t) for t in tokens)
