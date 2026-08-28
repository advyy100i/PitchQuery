"""Phase 4: Pandera schemas for the frames that move between pipeline steps.

dbt tests cover tables that are already in the warehouse. These cover the
dataframes in between — the window where a bad coordinate or a malformed token
is still in Python and has not yet become 1.6M committed rows.

The strongest check here is TOKEN_GRAMMAR_RE. It is built from `core.zones`
rather than typed out, so the vocabulary the writer emits and the vocabulary the
validator accepts cannot drift apart, and it guarantees that nothing enters the
index which `Retriever.sparse_rank` cannot tokenise. A token string that fails
this regex would not raise anywhere downstream; it would just quietly never
match a query.

Every call site passes lazy=True, so a bad batch reports every offending column
and row count at once instead of stopping at the first one and hiding the rest.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import pandera.pandas as pa  # noqa: E402
from pandera.pandas import Check, Column  # noqa: E402

from core.config import PITCH_MARGIN, PITCH_X, PITCH_Y  # noqa: E402
from core.zones import ACTIONS, BANDS, CHANNELS, MIN_EVENTS  # noqa: E402

# --- the token grammar, as a regex -------------------------------------------

_ZONES = [f"{b}-{c}" for b in BANDS for c in CHANNELS]

# Modifiers are emitted in a fixed order by core.zones.modifiers: '+' then '>'
# then '^'. Accepting them in any order here would let a reordering bug through
# silently while splitting one token into variants that no longer match.
TOKEN_RE = (r"(?:" + "|".join(ACTIONS) + r")@(?:"
            + "|".join(z.replace("-", r"\-") for z in _ZONES) + r")\+?>?\^?")
TOKEN_GRAMMAR_RE = rf"^{TOKEN_RE}(?: {TOKEN_RE})*$"
_TOKEN_GRAMMAR = re.compile(TOKEN_GRAMMAR_RE)

ZONE_PATH_RE = r"^(?:" + "|".join(z.replace("-", r"\-") for z in _ZONES) + r")(?: (?:" \
               + "|".join(z.replace("-", r"\-") for z in _ZONES) + r"))*$"

# Closed set on purpose. StatsBomb adding a new event type is a thing this
# project wants to hear about — core/zones.py has to decide whether it is a
# token, a set piece or noise — so an unrecognised type fails the run rather
# than being silently dropped by the `_IGNORED` fallthrough.
ALLOWED_TYPES = frozenset({
    "50/50", "Bad Behaviour", "Ball Receipt*", "Ball Recovery", "Block",
    "Camera off", "Camera On", "Carry", "Clearance", "Dispossessed", "Dribble",
    "Dribbled Past", "Duel", "Error", "Foul Committed", "Foul Won",
    "Goal Keeper", "Half End", "Half Start", "Injury Stoppage", "Interception",
    "Miscontrol", "Offside", "Own Goal Against", "Own Goal For", "Pass",
    "Player Off", "Player On", "Pressure", "Referee Ball-Drop", "Shield",
    "Shot", "Starting XI", "Substitution", "Tactical Shift",
})

# --- schemas ------------------------------------------------------------------

# Coordinates are nullable because plenty of real events genuinely have none —
# Half Start, Substitution, Starting XI. What must never happen is a coordinate
# that exists and is nowhere near the pitch, which is what these bounds catch.
# The PITCH_MARGIN slack is there because two shots in this corpus are struck
# from on the goal line and come back at x = 120.5; see core/config.py.
X_RANGE = Check.in_range(-PITCH_MARGIN, PITCH_X + PITCH_MARGIN)
Y_RANGE = Check.in_range(-PITCH_MARGIN, PITCH_Y + PITCH_MARGIN)
EventSchema = pa.DataFrameSchema(
    {
        "event_id": Column(str, nullable=False, unique=True),
        "match_id": Column("int64", nullable=False),
        "idx": Column("int64", nullable=False, checks=Check.gt(0)),
        "type": Column(str, Check.isin(ALLOWED_TYPES), nullable=True),
        "x": Column(float, X_RANGE, nullable=True),
        "y": Column(float, Y_RANGE, nullable=True),
        "end_x": Column(float, X_RANGE, nullable=True),
        "end_y": Column(float, Y_RANGE, nullable=True),
        "token": Column(str, Check.str_matches(rf"^{TOKEN_RE}$"), nullable=True),
    },
    name="events",
    strict=False,          # the loader carries more columns than are checked
    coerce=True,
)

ShotSchema = pa.DataFrameSchema(
    {
        "event_id": Column(str, nullable=False, unique=True),
        "x": Column(float, X_RANGE, nullable=False),
        "y": Column(float, Y_RANGE, nullable=False),
        # 134.5 is the far corner (0, 80) to the goal centre; nothing legal is
        # further away, and anything longer means the mirroring convention broke.
        "distance": Column(float, Check.in_range(0.0, 135.0), nullable=False),
        "angle": Column(float, Check.in_range(0.0, 3.1416), nullable=False),
        "is_goal": Column(bool, nullable=False),
        # A comparison column, never a feature — but if it ever arrives outside
        # [0, 1] the benchmark in docs/benchmark.md is nonsense.
        "statsbomb_xg": Column(float, Check.in_range(0.0, 1.0), nullable=True),
        "n_def_in_cone": Column("Int64", Check.in_range(0, 11), nullable=True),
    },
    name="shots",
    strict=False,
    coerce=True,
)

PossessionSchema = pa.DataFrameSchema(
    {
        "possession_uid": Column(str, nullable=False, unique=True),
        "match_id": Column("int64", nullable=False),
        "team": Column(str, nullable=False),
        "token_string": Column(str, Check.str_matches(TOKEN_GRAMMAR_RE), nullable=False),
        "zone_path": Column(str, Check.str_matches(ZONE_PATH_RE), nullable=False),
        "n_events": Column("int64", Check.ge(MIN_EVENTS), nullable=False),
        "duration_s": Column(float, Check.ge(0.0), nullable=False),
        "xg_sum": Column(float, Check.in_range(0.0, 10.0), nullable=False),
        "ended_in_shot": Column(bool, nullable=False),
        "ended_in_goal": Column(bool, nullable=False),
    },
    name="possessions",
    strict=False,
    coerce=True,
    checks=[
        # A possession that ended in a goal but not in a shot is a contradiction
        # in the builder, not in the data. No single column can express it.
        Check(lambda df: ~(df["ended_in_goal"] & ~df["ended_in_shot"]),
              element_wise=False, error="ended_in_goal without ended_in_shot"),
        Check(lambda df: df["token_string"].str.split().str.len() == df["n_events"],
              element_wise=False, error="n_events disagrees with the token count"),
    ],
)


def token_ok(token_string: str) -> bool:
    """One string against the grammar. Used by tests and by eval fixtures."""
    return bool(_TOKEN_GRAMMAR.match(token_string or ""))


def check(schema: pa.DataFrameSchema, df, *, where: str = "") -> "pd.DataFrame":
    """Validate lazily and re-raise with the step name attached.

    Pandera's own message names the column and the failure count, which is what
    Phase 4 is for; this only adds where in the pipeline it happened, because
    the same schema is checked at more than one point.
    """
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases
        summary = (cases.groupby(["column", "check"], dropna=False).size()
                   .sort_values(ascending=False).head(10)
                   if len(cases) else cases)
        raise ValueError(
            f"{schema.name} contract failed{f' at {where}' if where else ''}: "
            f"{len(cases)} failing cases\n{summary}"
        ) from err


def frame(rows: list, columns: list):
    """Build a frame from the loader's positional tuples, keeping only `columns`.

    The insert statements are positional, so the contract has to be told the
    column order once rather than guessing it. Passing the same list that the
    INSERT uses keeps the two in step.
    """
    return pd.DataFrame(rows, columns=columns)
