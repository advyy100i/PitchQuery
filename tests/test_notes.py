"""The scouting note's central guarantee: a claim cannot cite what it did not use.

`core/notes.py` argues that citation-checking is unnecessary because each
sentence is computed from the possessions it cites. That is only worth
believing if something enforces it, which is what these tests do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.notes import MIN_SUPPORT, scouting_note, verify  # noqa: E402


def row(uid, **kw):
    base = dict(possession_uid=uid, match_id=1, team="Spain", opponent="Italy",
                competition="UEFA Euro", season="2024", play_pattern="Regular Play",
                zone_path="D-C M-C F-C", token_string="PASS@D-C RECV@M-C SHOT@F-C",
                n_events=10, duration_s=12.0, ended_in_shot=True,
                ended_in_goal=False, xg_sum=0.10)
    base.update(kw)
    return base


def sample(n=8, **kw):
    return [row(f"m:{i}", **kw) for i in range(n)]


def test_every_cited_uid_is_in_the_result_set():
    rows = sample()
    assert verify(scouting_note(rows), rows) == []


def test_citations_are_never_empty():
    for claim in scouting_note(sample()):
        assert claim.uids, f"claim with no evidence: {claim.text!r}"


def test_a_claim_about_goals_cites_only_the_goals():
    rows = sample(6)
    rows[0]["ended_in_goal"] = True
    rows[1]["ended_in_goal"] = True
    goal_uids = {r["possession_uid"] for r in rows if r["ended_in_goal"]}

    goal_claims = [c for c in scouting_note(rows) if "goal" in c.text or "scored" in c.text]
    assert goal_claims, "expected a claim about the goals"
    for claim in goal_claims:
        # The sentence counts goals, so its evidence must be goals — not the
        # whole result set with a plausible number attached.
        assert set(claim.uids) <= goal_uids


def test_a_claim_about_one_team_cites_only_that_team():
    rows = sample(8)
    for r in rows[:3]:
        r["team"] = "Portugal"
    for claim in scouting_note(rows):
        if "Portugal" in claim.text:
            cited = {r["possession_uid"] for r in rows if r["team"] == "Portugal"}
            assert set(claim.uids) <= cited


def test_query_constraints_are_not_restated():
    """Filtering on a field and then being told about it is an echo, not a note."""
    rows = sample(8, play_pattern="From Corner")
    stated = scouting_note(rows, given={"play_pattern", "ended_in_shot"})
    assert not any("corner" in c.text.lower() for c in stated)
    assert not any("end in a shot" in c.text for c in stated)


def test_too_few_results_produce_no_note():
    assert scouting_note(sample(MIN_SUPPORT - 1)) == []


def test_note_stays_short():
    assert len(scouting_note(sample(8))) <= 5
