"""Phase 4: the pipeline contracts, checked without a database.

Two things worth pinning down. The token grammar regex is generated from
core.zones, so these tests are what stops it generating something permissive by
accident — a regex that accepts everything fails nothing and would look exactly
like a passing pipeline. And the coordinate bounds carry a deliberate two-unit
slack for shots struck from on the goal line, which is the kind of allowance
that quietly widens over time unless something asserts where it stops.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest

from core.zones import ACTIONS, token
from pipeline.contracts import (EventSchema, PossessionSchema, check,
                                token_ok)

GOOD = "RECOV@D-C PASS@D-L+ CARRY@M-L SWITCH@M-L+ CROSS@F-R>^ SHOT@F-C"


def test_grammar_accepts_a_real_token_string():
    assert token_ok(GOOD)


@pytest.mark.parametrize("bad", [
    "",                       # empty
    "NOPE@D-C PASS@D-L",      # action outside the vocabulary
    "PASS@Z-Q",               # zone outside the grid
    "PASS@D-C^+",             # modifiers out of order
    "PASS@D-C  SHOT@F-C",     # double space: an empty token between them
    "pass@d-c",               # lower case
    "PASS@D-C SHOT",          # a token with no zone
])
def test_grammar_rejects(bad):
    assert not token_ok(bad)


def test_every_action_in_the_vocabulary_is_expressible():
    """A token the writer can emit that the validator rejects would be invisible
    until a query failed to match it, so check the whole vocabulary."""
    for act in ACTIONS:
        assert token_ok(f"{act}@F-C {act}@D-L+ {act}@M-RI>^")


def test_zones_module_and_grammar_agree():
    """The regex is generated from core.zones; this proves it against the
    function that actually writes the tokens rather than against itself."""
    ev = {"type": {"name": "Pass"}, "location": [30.0, 20.0],
          "pass": {"end_location": [60.0, 22.0], "cross": True},
          "under_pressure": True}
    assert token_ok(token(ev))


def _events(**overrides):
    row = {"event_id": "3b8f0f9e-0000-4000-8000-000000000001", "match_id": 1,
           "idx": 1, "type": "Pass", "x": 60.0, "y": 40.0, "end_x": 70.0,
           "end_y": 41.0, "token": "PASS@M-C"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_a_coordinate_off_the_scale_fails_and_names_the_column():
    """Phase 4's own acceptance test, without needing to corrupt a cached file."""
    with pytest.raises(ValueError) as e:
        check(EventSchema, _events(x=200.0), where="unit test")
    assert "x" in str(e.value)
    assert "unit test" in str(e.value)


def test_a_shot_from_the_goal_line_is_allowed():
    """x = 120.5 is real StatsBomb data — two rows in this corpus. The bound has
    to clear it, or the contract fails on football that happened."""
    check(EventSchema, _events(x=120.5, y=79.6))


def test_an_unknown_event_type_fails():
    with pytest.raises(ValueError):
        check(EventSchema, _events(type="Teleport"))


def _possession(**overrides):
    row = {"possession_uid": "1:1", "match_id": 1, "team": "A",
           "token_string": GOOD, "zone_path": "D-C D-L M-L M-L F-R F-C",
           "n_events": 6, "duration_s": 12.0, "xg_sum": 0.2,
           "ended_in_shot": True, "ended_in_goal": False}
    row.update(overrides)
    return pd.DataFrame([row])


def test_possession_contract_passes_a_real_row():
    check(PossessionSchema, _possession())


def test_a_goal_without_a_shot_fails():
    with pytest.raises(ValueError):
        check(PossessionSchema, _possession(ended_in_goal=True, ended_in_shot=False))


def test_token_count_must_match_n_events():
    with pytest.raises(ValueError):
        check(PossessionSchema, _possession(n_events=99))


def test_an_untokenisable_possession_never_reaches_the_index():
    """The single most important check in the pipeline: a string the searcher
    cannot parse would match nothing and raise nowhere."""
    with pytest.raises(ValueError):
        check(PossessionSchema, _possession(token_string="PASS@D-C GIBBERISH"))
