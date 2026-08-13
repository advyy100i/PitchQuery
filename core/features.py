"""Shot geometry and freeze-frame features.

Everything here is derived from the event JSON alone. `statsbomb_xg` is NEVER
read in this module — it is the comparison column, never a feature (§8 rule 3).

Coordinates follow the convention verified in docs/probes.md: 120 x 80 pitch,
origin top-left, acting team always attacks left->right, so the goal being
attacked is always at (120, 40).
"""
import math
from typing import Optional, Sequence

from core.config import GOAL_CENTRE, GOAL_POSTS

GX, GY = GOAL_CENTRE
(PX1, PY1), (PX2, PY2) = GOAL_POSTS   # (120, 36) and (120, 44)


def distance_to_goal(x: float, y: float) -> float:
    """Euclidean distance from the shot location to the centre of the goal."""
    return math.hypot(GX - x, GY - y)


def goal_angle(x: float, y: float) -> float:
    """Angle in radians subtended by the goal mouth at the shot location.

    Standard xG feature: the wider the visible goal, the better the chance.
    Uses the cosine rule on the triangle (shot, near post, far post). Returns 0
    for a shot taken on the goal line outside the posts, where the goal has no
    visible width.
    """
    a = math.hypot(PX1 - x, PY1 - y)
    b = math.hypot(PX2 - x, PY2 - y)
    c = abs(PY2 - PY1)                      # goal width, 8 yards
    if a == 0 or b == 0:
        return 0.0
    cos = (a * a + b * b - c * c) / (2 * a * b)
    return math.acos(max(-1.0, min(1.0, cos)))


def _in_cone(px: float, py: float, sx: float, sy: float) -> bool:
    """Is (px, py) inside the triangle (shot, post, post)?

    Barycentric sign test. A defender in the cone is physically between the
    ball and the goal mouth, which is what actually blocks a shot.
    """
    def sign(ax, ay, bx, by, cx, cy):
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)

    d1 = sign(px, py, sx, sy, PX1, PY1)
    d2 = sign(px, py, PX1, PY1, PX2, PY2)
    d3 = sign(px, py, PX2, PY2, sx, sy)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def freeze_frame_features(freeze_frame: Optional[Sequence[dict]],
                          x: float, y: float) -> dict:
    """Context features from the shot-moment freeze frame.

    A freeze frame entry looks like:
        {"location": [x, y], "player": {...}, "position": {"name": "..."},
         "teammate": false}

    Returns None for every field when there is no usable frame, so that the
    model sees a missing value rather than a fabricated one. LightGBM handles
    NaN natively; do not impute these.

    - n_def_in_cone     opponents inside the shot/posts triangle, GK excluded
    - dist_nearest_def  nearest opponent, GK excluded (the GK gets its own pair)
    - gk_dist_to_goal   opponent keeper's distance from the goal centre
    - gk_off_line       how far the keeper has advanced off his line (120 - x)
    """
    out = {"n_def_in_cone": None, "dist_nearest_def": None,
           "gk_dist_to_goal": None, "gk_off_line": None}
    if not freeze_frame:
        return out

    n_cone = 0
    nearest = None
    for p in freeze_frame:
        if p.get("teammate", True):
            continue
        loc = p.get("location") or []
        if len(loc) < 2:
            continue
        px, py = float(loc[0]), float(loc[1])
        is_gk = (p.get("position") or {}).get("name") == "Goalkeeper"
        if is_gk:
            out["gk_dist_to_goal"] = math.hypot(GX - px, GY - py)
            out["gk_off_line"] = GX - px
            continue
        if _in_cone(px, py, x, y):
            n_cone += 1
        d = math.hypot(px - x, py - y)
        if nearest is None or d < nearest:
            nearest = d

    out["n_def_in_cone"] = n_cone
    out["dist_nearest_def"] = nearest
    return out


def shot_row(ev: dict) -> Optional[dict]:
    """Flatten one StatsBomb Shot event into the `shots` table shape.

    Returns None if the event is not a shot or has no location.
    """
    if (ev.get("type") or {}).get("name") != "Shot":
        return None
    loc = ev.get("location") or []
    if len(loc) < 2:
        return None
    x, y = float(loc[0]), float(loc[1])
    s = ev.get("shot") or {}
    ff = s.get("freeze_frame")

    row = {
        "event_id": ev["id"],
        "team": (ev.get("team") or {}).get("name"),
        "player": (ev.get("player") or {}).get("name"),
        "x": x,
        "y": y,
        "distance": distance_to_goal(x, y),
        "angle": goal_angle(x, y),
        "body_part": (s.get("body_part") or {}).get("name"),
        "technique": (s.get("technique") or {}).get("name"),
        "shot_type": (s.get("type") or {}).get("name"),
        "first_time": bool(s.get("first_time", False)),
        "under_pressure": bool(ev.get("under_pressure", False)),
        "play_pattern": (ev.get("play_pattern") or {}).get("name"),
        "is_goal": (s.get("outcome") or {}).get("name") == "Goal",
        "statsbomb_xg": s.get("statsbomb_xg"),
        "freeze_frame": ff,
    }
    row.update(freeze_frame_features(ff, x, y))
    return row
