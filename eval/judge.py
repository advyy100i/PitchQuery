"""Programmatic relevance rubrics — one predicate per query in queries.yaml.

This replaces hand-judging every pooled result. Each predicate is a direct
translation of the `rubric` field into a test over the token string, so a label
is reproducible, applies to all 66,817 possessions rather than a pooled sample,
and does not drift as a human gets tired at query 22.

What it costs: the judgement-y clauses in some rubrics ("reads as controlled
circulation rather than a scramble") cannot be expressed this way. Where a
proxy is used, the docstring says so. `eval/audit.py` measures how often these
predicates agree with a human on a random sample, which is the honest way to
report how much that cost.

Two limitations are structural rather than fixable, and are reported by
--stats so they end up in the write-up rather than being quietly absorbed:

  * Tokens record where an event STARTED, not where it ended. So "a pass that
    lands in the final third" has to be inferred from the next token's zone.
  * The 15-zone grid cannot express "outside the box": F-C spans x 80-102 as
    well as the box itself. q14 is therefore approximated, not answered.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOX_SHOT = ("SHOT@F-LI", "SHOT@F-C", "SHOT@F-RI")
WIDE_F = ("F-L", "F-LI", "F-RI", "F-R")


# --- small helpers -----------------------------------------------------------

def toks(p) -> list:
    return p["token_string"].split()


def zones(p) -> list:
    return p["zone_path"].split()


def zone_of(tok: str) -> str:
    return tok.split("@", 1)[1].rstrip("+>^") if "@" in tok else ""


def band_of(tok: str) -> str:
    z = zone_of(tok)
    return z.split("-")[0] if z else ""


def side(zone: str) -> str:
    """Which flank a zone sits on. 'C' for the central channel."""
    ch = zone.split("-")[-1]
    if ch in ("L", "LI"):
        return "L"
    if ch in ("R", "RI"):
        return "R"
    return "C"


def any_tok(p, *prefixes) -> bool:
    return any(t.startswith(prefixes) for t in toks(p))


def first_index(p, *prefixes):
    for i, t in enumerate(toks(p)):
        if t.startswith(prefixes):
            return i
    return None


def shot_indices(p) -> list:
    return [i for i, t in enumerate(toks(p)) if t.startswith("SHOT")]


def cross_then_shot(p, *cross_prefixes) -> bool:
    """A qualifying cross that reaches the box, with a shot after it."""
    ts = toks(p)
    shots = shot_indices(p)
    if not shots:
        return False
    for i, t in enumerate(ts):
        if t.startswith(cross_prefixes) and ">" in t and any(s > i for s in shots):
            return True
    return False


# --- one predicate per query -------------------------------------------------
# Note: 'CROSS@F-R' as a prefix also matches 'CROSS@F-RI', which is what the
# rubrics ask for — both are the right-hand side of the pitch.

def q01(p):
    """Cross from F-R/F-RI into the box, shot afterwards."""
    return cross_then_shot(p, "CROSS@F-R")


def q02(p):
    """Mirror of q01 on the left."""
    return cross_then_shot(p, "CROSS@F-L")


def q03(p):
    """One pass tagged SWITCH. The grammar already encodes the 30m/20m test."""
    return any_tok(p, "SWITCH")


def q04(p):
    """From Counter, ends in a shot, and stays short.

    PROXY: 'a counter that stalls into a 40-pass possession' is capped at 25
    tokens rather than judged.
    """
    return p["ended_in_shot"] and p["n_events"] <= 25


def q05(p):
    """Corner producing a shot from inside the box, including second phase."""
    return any_tok(p, *BOX_SHOT)


def q06(p):
    """A THROUGH ball played outside the defensive third."""
    return any(t.startswith("THROUGH") and band_of(t) != "D" for t in toks(p))


def q07(p):
    """Goal kick whose possession reaches the final third."""
    return any(band_of(t) == "F" for t in toks(p))


def q08(p):
    """25+ tokens ending in a shot, and the shot is in the final third.

    Excludes the 'padded with defensive recycling then a hopeful effort' case
    by requiring the shot itself to be in an F- zone.
    """
    return any(t.startswith("SHOT") and band_of(t) == "F" for t in toks(p))


def q09(p):
    """Recovery or interception in the final third, shot soon after."""
    start = toks(p)[:2]
    if not any(t.startswith(("RECOV@F", "INT@F")) for t in start):
        return False
    shots = shot_indices(p)
    return bool(shots) and min(shots) <= 8


def q10(p):
    """Dribble in the final third followed by a carry or shot in the box."""
    ts = toks(p)
    for i, t in enumerate(ts):
        if t.startswith("DRIB") and band_of(t) == "F":
            for later in ts[i + 1:]:
                if ">" in later or later.startswith(BOX_SHOT):
                    return True
    return False


def q11(p):
    """Arrives in F-L/F-LI having gone through an inside-left channel zone."""
    if zones(p)[-1] not in ("F-L", "F-LI"):
        return False
    return any(zone_of(t) in ("M-LI", "F-LI") for t in toks(p))


def q12(p):
    """PSG counter that actually reaches the final third, and stays direct."""
    return any(band_of(t) == "F" for t in toks(p)) and p["n_events"] <= 25


def q13(p):
    """Goal begun by a recovery/interception in the attacking half."""
    return any(t.startswith(("RECOV@M", "RECOV@F", "INT@M", "INT@F"))
               for t in toks(p)[:2])


def q14(p):
    """Shot from outside the box.

    APPROXIMATION: the 15-zone grid has no box boundary — F-C spans x 80-102
    as well as the box. A shot in an M- zone is unambiguously outside; a shot
    in F-C may or may not be. Only the unambiguous case is counted, so this
    query's precision is a lower bound, and the write-up says so.
    """
    shots = shot_indices(p)
    return bool(shots) and band_of(toks(p)[shots[-1]]) == "M"


def q15(p):
    """Ball goes wide right in the final third, then is squared into the box."""
    ts = toks(p)
    if not any(zone_of(t) == "F-R" for t in ts):
        return False
    shots = shot_indices(p)
    for i, t in enumerate(ts):
        if zone_of(t) in ("F-R", "F-RI") and ">" in t and any(s > i for s in shots):
            return True
    return False


def q16(p):
    """Throw-in taken in the final third that produces a shot."""
    return band_of(toks(p)[0]) == "F"


def q17(p):
    """Free kick delivered into the box."""
    return any(">" in t for t in toks(p))


def q18(p):
    """Retreat to the defensive third, then re-attack on the opposite flank."""
    zs = zones(p)
    for i, z in enumerate(zs):
        if z.startswith(("M-", "F-")) and side(z) != "C":
            for j in range(i + 1, len(zs)):
                if zs[j].startswith("D-"):
                    for k in range(j + 1, len(zs)):
                        if (zs[k].startswith("F-") and side(zs[k]) != "C"
                                and side(zs[k]) != side(z)):
                            return True
                    break
    return False


def q19(p):
    """Four consecutive touches in the central corridor, progressing forward."""
    zs = zones(p)
    run = 0
    for z in zs:
        run = run + 1 if z in ("M-C", "F-C") else 0
        if run >= 4:
            return any(z2.startswith("F-") for z2 in zs)
    return False


def q20(p):
    """Goal finished centrally in the final third."""
    shots = shot_indices(p)
    return bool(shots) and zone_of(toks(p)[shots[-1]]) == "F-C"


def q21(p):
    """Four or more tokens under pressure, and still escapes the defensive third."""
    ts = toks(p)
    if sum(1 for t in ts if "^" in t) < 4:
        return False
    return band_of(ts[0]) == "D" and any(band_of(t) in ("M", "F") for t in ts)


def q22(p):
    """One event in a D- zone immediately followed by one in an F- zone.

    PROXY for 'a single long pass from defence into the final third': tokens
    carry the start zone only, so a D->F jump between consecutive events is
    the available evidence that the ball travelled that far in one action.
    """
    zs = zones(p)
    return any(zs[i].startswith("D-") and zs[i + 1].startswith("F-")
               for i in range(len(zs) - 1))


def q23(p):
    """England possession finishing on the left of the final third."""
    return zones(p)[-1] in ("F-L", "F-LI")


def q24(p):
    """Long Spain possession ending in a shot, without turnover churn.

    PROXY for 'controlled circulation rather than a scramble': at most two
    LOSS/DUEL tokens.
    """
    return sum(1 for t in toks(p) if t.startswith(("LOSS", "DUEL"))) <= 2


def q25(p):
    """Two shots with a recovery or duel between them."""
    ts = toks(p)
    shots = shot_indices(p)
    if len(shots) < 2:
        return False
    for a, b in zip(shots, shots[1:]):
        if any(ts[i].startswith(("RECOV", "DUEL")) for i in range(a + 1, b)):
            return True
    return False


def q26(p):
    """Travels down the right, finishes with a shot in F-RI."""
    ts = toks(p)
    if not any(zone_of(t) in ("F-R", "M-R") for t in ts):
        return False
    shots = shot_indices(p)
    return bool(shots) and zone_of(ts[shots[-1]]) == "F-RI"


def q27(p):
    """Interception in midfield that reaches the final third quickly."""
    if not any(t.startswith("INT@M") for t in toks(p)[:2]):
        return False
    return any(band_of(t) == "F" for t in toks(p)[:8])


def q28(p):
    """Cutback: a wide final-third event into the box, then a central shot."""
    ts = toks(p)
    for i, t in enumerate(ts):
        if zone_of(t) in WIDE_F and ">" in t:
            if any(ts[j].startswith("SHOT@F-C") for j in (i + 1, i + 2)
                   if j < len(ts)):
                return True
    return False


def q29(p):
    """Two passes inside the inside-left channel, progressing to the final third."""
    passes = sum(1 for t in toks(p)
                 if t.startswith("PASS") and zone_of(t) in ("M-LI", "F-LI"))
    return passes >= 2 and any(band_of(t) == "F" for t in toks(p))


def q30(p):
    """High-xG open-play chance finished from inside the box."""
    return any_tok(p, *BOX_SHOT)


PREDICATES = {name: fn for name, fn in list(globals().items())
              if name.startswith("q") and len(name) == 3 and callable(fn)}

# Queries where the SQL filter alone already satisfies the rubric for >= 90% of
# the rows it returns, measured by --stats rather than guessed. Ranking cannot
# help on these — any five survivors score ~1.0 — so the report gives the mean
# with and without them. The honest headline is the number without.
#   q04 97.8%   q05 99.7%   q08 100%   q24 97.3%   q30 100%
FILTER_DOMINATED = {"q04", "q05", "q08", "q24", "q30"}


def judge(qid: str, row: dict) -> int:
    fn = PREDICATES.get(qid)
    if fn is None:
        raise KeyError(f"no predicate for {qid}")
    return 1 if fn(row) else 0


def main():
    """--stats: how many possessions in the whole corpus satisfy each rubric.

    A predicate matching 0 rows is broken; one matching most of the corpus is
    not discriminating. Both are worth seeing before trusting any P@5.
    """
    import argparse

    import yaml

    from core import db
    from core.retrieval import Filters

    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.parse_args()

    queries = yaml.safe_load(
        (Path(__file__).parent / "queries.yaml").read_text(encoding="utf-8"))["queries"]
    conn = db.connect()

    print(f"{'id':5} {'filtered':>9} {'relevant':>9} {'rate':>7}  query")
    for q in queries:
        f = Filters(**(q.get("filters") or {}))
        sql, params = f.where()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT possession_uid, token_string, zone_path, n_events, "
                f"       ended_in_shot, ended_in_goal, xg_sum, play_pattern "
                f"FROM possessions WHERE {sql}", params)
            cols = [d.name for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        rel = sum(judge(q["id"], r) for r in rows)
        rate = rel / len(rows) * 100 if rows else 0.0
        mark = " *" if q["id"] in FILTER_DOMINATED else ""
        print(f"{q['id']:5} {len(rows):9,} {rel:9,} {rate:6.1f}%  {q['text'][:52]}{mark}")
    print("\n* rubric adds little beyond the SQL filters — precision is high by construction")
    conn.close()


if __name__ == "__main__":
    main()
