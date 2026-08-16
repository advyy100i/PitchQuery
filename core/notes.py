"""Deterministic scouting notes with citations that cannot be wrong.

Plan §7 specified a second LLM call: pass the top results as text, ask for 3-4
sentences, and require every claim to cite a possession_uid — then drop
sentences whose citations don't check out.

This does the same job by construction instead of by inspection. Each sentence
is *computed from* a subset of the results, and that subset IS the citation. A
claim and its evidence are produced by the same expression, so there is no
verification step to run and no way for the two to disagree:

    goals = [r for r in rows if r["ended_in_goal"]]
    Claim(f"{len(goals)} end in a goal", uids=[r["possession_uid"] for r in goals])

The citation-checking the plan describes is not implemented here because it
cannot fail. That is a stronger guarantee than an LLM plus a checker, not a
weaker one — the checker exists to catch a generator that can lie, and this
generator cannot.

The honest trade: it says only what these functions know how to say. It will
never notice something surprising in the data the way a model might.
"""
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

# A claim needs at least this much support to be worth stating. Below it the
# sentence would be describing noise.
MIN_SUPPORT = 2


@dataclass
class Claim:
    text: str
    uids: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"text": self.text, "uids": self.uids}


def _uids(rows) -> list:
    return [r["possession_uid"] for r in rows]


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return one if n == 1 else (many or one + "s")


def _xg(row: dict) -> float:
    """This project's xG for a possession, falling back to StatsBomb's.

    The result cards show our model, so the note beside them has to quote the
    same one. A sentence reading "they average 0.14 xG" next to badges reading
    0.09 is the kind of quiet disagreement nobody reports and everybody
    distrusts.

    The fallback is not decoration: `my_xg_sum` is null wherever the model is
    not loaded, and null for a possession whose only shot is a penalty. Reading
    those as zero would understate a set of chances rather than describe it.
    """
    mine = row.get("my_xg_sum")
    return float(mine if mine is not None else (row.get("xg_sum") or 0.0))


def scouting_note(rows: list, query: str = "", given=frozenset()) -> list:
    """Build a short note about a result set. Returns ordered claims.

    Every claim carries the uids it was derived from; the UI renders those as
    clickable clips, so a reader can always check a sentence against the
    footage it describes.

    `given` names the things the QUERY already fixed — filtering on
    `ended_in_shot` and then being told "8 of 8 end in a shot" is not an
    observation, it is an echo. Constrained fields are skipped, or narrowed to
    the part that still carries information (the goals within the shots).
    """
    rows = [r for r in rows if r]
    if len(rows) < MIN_SUPPORT:
        return []

    claims: list = []
    n = len(rows)

    # -- who -------------------------------------------------------------------
    teams = Counter(r["team"] for r in rows)
    top_team, top_n = teams.most_common(1)[0]
    team_rows = [r for r in rows if r["team"] == top_team]
    if "team" in given:
        pass
    elif top_n >= max(MIN_SUPPORT, n // 2):
        claims.append(Claim(
            f"{top_n} of the {n} come from {top_team}.", _uids(team_rows)))
    elif len(teams) >= 3:
        claims.append(Claim(
            f"These span {len(teams)} teams, most often {top_team} "
            f"({top_n} of {n}).", _uids(rows)))

    # -- how they start --------------------------------------------------------
    patterns = Counter(r["play_pattern"] for r in rows if r["play_pattern"])
    if patterns and "play_pattern" not in given:
        pat, pat_n = patterns.most_common(1)[0]
        pat_rows = [r for r in rows if r["play_pattern"] == pat]
        if pat_n >= MIN_SUPPORT and pat_n > n // 3:
            claims.append(Claim(
                f"{pat_n} begin from {pat.lower().replace('from ', '')}.",
                _uids(pat_rows)))

    # -- what happens ----------------------------------------------------------
    shots = [r for r in rows if r["ended_in_shot"]]
    goals = [r for r in rows if r["ended_in_goal"]]
    # One subject per sentence. A sentence that counted goals while citing every
    # shot was the bug tests/test_notes.py caught: the claim and its evidence
    # described different sets. Splitting them keeps each citation exact.
    if shots:
        xg = sum(_xg(r) for r in shots) / len(shots)
        if "ended_in_shot" in given:
            # The shot was asked for; only the chance quality is news.
            claims.append(Claim(
                f"They average {xg:.2f} xG per possession.", _uids(shots)))
        elif len(shots) >= MIN_SUPPORT:
            claims.append(Claim(
                f"{len(shots)} of {n} end in a shot, averaging {xg:.2f} xG.",
                _uids(shots)))

    if goals and "ended_in_goal" not in given:
        verb = "is" if len(goals) == 1 else "are"
        goal_xg = sum(_xg(r) for r in goals) / len(goals)
        claims.append(Claim(
            f"{len(goals)} of the {n} {verb} scored, from {goal_xg:.2f} xG.",
            _uids(goals)))

    # -- where they finish -----------------------------------------------------
    ends = Counter(r["zone_path"].split()[-1] for r in rows if r["zone_path"])
    if ends and "end_zone" not in given:
        zone, zone_n = ends.most_common(1)[0]
        zone_rows = [r for r in rows if r["zone_path"].split()[-1] == zone]
        if zone_n >= MIN_SUPPORT:
            claims.append(Claim(
                f"Most finish in {zone} ({zone_n} of {n}).", _uids(zone_rows)))

    # -- tempo -----------------------------------------------------------------
    lengths = [r["n_events"] for r in rows]
    durations = [r["duration_s"] for r in rows]
    if len(lengths) >= MIN_SUPPORT:
        med_len, med_dur = median(lengths), median(durations)
        shape = ("short and direct" if med_len <= 8 else
                 "sustained" if med_len >= 20 else "moderate")
        claims.append(Claim(
            f"They are {shape}: a median of {med_len:.0f} "
            f"{_plural(int(med_len), 'touch', 'touches')} over "
            f"{med_dur:.0f} seconds.", _uids(rows)))

    # -- the standout ----------------------------------------------------------
    # A single-possession claim is allowed because it names its evidence
    # outright rather than generalising from one case.
    best = max(rows, key=_xg)
    if _xg(best) > 0.15:
        claims.append(Claim(
            f"The clearest chance is {best['team']} against "
            f"{best['opponent']} at {_xg(best):.2f} xG.",
            [best["possession_uid"]]))

    return claims[:5]


def verify(claims: list, rows: list) -> list:
    """Confirm every cited uid is actually in the result set.

    This should never find anything — the claims are built from these rows.
    It exists so the guarantee is enforced by a test rather than by trust, and
    it is what `tests/test_notes.py` asserts.
    """
    known = {r["possession_uid"] for r in rows}
    return [c for c in claims if not set(c.uids).issubset(known)]
