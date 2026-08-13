"""Phase 0: verify the six assumptions the whole pipeline rests on (plan §2).

Downloads a small sample of event files (default 12 matches, ~5 MB each) and
checks A1-A6 empirically. Writes docs/probes.md.

Run:  python ingest/01_probe_assumptions.py
      python ingest/01_probe_assumptions.py --per-comp 6
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import DOCS_DIR  # noqa: E402
from ingest import _http  # noqa: E402

random.seed(7)


def pick_sample(per_comp: int):
    """Two competitions with 360, two without, `per_comp` matches from each."""
    comps = _http.competitions()
    scored = []
    for c in comps:
        ms = _http.matches(c["competition_id"], c["season_id"])
        n360 = sum(1 for m in ms if m.get("match_status_360") == "available")
        scored.append((c, ms, n360))

    with360 = sorted([s for s in scored if s[2] > 0], key=lambda s: -s[2])[:2]
    without = sorted([s for s in scored if s[2] == 0], key=lambda s: -len(s[1]))[:2]

    sample = []
    for c, ms, n360 in with360 + without:
        pool = [m for m in ms if m.get("match_status_360") == "available"] or ms
        for m in random.sample(pool, min(per_comp, len(pool))):
            sample.append((c, m))
    return sample, with360, without


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-comp", type=int, default=3,
                    help="matches sampled per competition (4 competitions)")
    args = ap.parse_args()

    sample, with360, without = pick_sample(args.per_comp)
    print(f"sampling {len(sample)} matches:")
    for c, m in sample:
        print(f"  {c['competition_name']} {c['season_name']} — {m['match_id']} "
              f"{m['home_team']['home_team_name']} v {m['away_team']['away_team_name']}")

    xs, ys = [], []
    n_events = 0
    missing_possession = 0
    missing_possession_team = 0
    shots = 0
    shots_with_xg = 0
    shots_with_ff = 0
    shot_x_by_period = defaultdict(list)
    index_disagreements = 0
    timestamp_resets = 0
    matches_checked = 0
    type_counter = Counter()

    for c, m in sample:
        ev = _http.events(m["match_id"])
        if not ev:
            print(f"  ! no events for {m['match_id']}")
            continue
        matches_checked += 1
        n_events += len(ev)

        for e in ev:
            type_counter[e.get("type", {}).get("name")] += 1
            loc = e.get("location")
            if isinstance(loc, list) and len(loc) >= 2:
                xs.append(loc[0])
                ys.append(loc[1])
            if e.get("possession") is None:
                missing_possession += 1
            if e.get("possession_team") is None:
                missing_possession_team += 1
            if e.get("type", {}).get("name") == "Shot":
                shots += 1
                s = e.get("shot", {})
                if s.get("statsbomb_xg") is not None:
                    shots_with_xg += 1
                if s.get("freeze_frame"):
                    shots_with_ff += 1
                if isinstance(loc, list):
                    shot_x_by_period[e.get("period")].append(loc[0])

        # A6: index order vs (period, timestamp) order
        by_index = [e["id"] for e in sorted(ev, key=lambda e: e["index"])]
        by_time = [e["id"] for e in sorted(ev, key=lambda e: (e["period"], e["timestamp"]))]
        if by_index != by_time:
            index_disagreements += 1
        firsts = {}
        for e in ev:
            firsts.setdefault(e["period"], e["timestamp"])
        if len(firsts) > 1 and all(t.startswith("00:00:0") for t in firsts.values()):
            timestamp_resets += 1

    # A5: does three-sixty exist only where flagged?
    a5_lines = []
    flagged = [m for c, m in sample if m.get("match_status_360") == "available"][:5]
    unflagged = [m for c, m in sample if m.get("match_status_360") != "available"][:5]
    for label, group in (("flagged", flagged), ("unflagged", unflagged)):
        hits = 0
        for m in group:
            if _http.three_sixty(m["match_id"]) is not None:
                hits += 1
        a5_lines.append(f"{label}: {hits}/{len(group)} have a three-sixty file")

    pct = lambda n, d: f"{100.0 * n / d:.1f}%" if d else "n/a"
    fwd = sum(1 for p, v in shot_x_by_period.items() for x in v if x > 60)
    all_shot_x = [x for v in shot_x_by_period.values() for x in v]

    rows = [
        ("A1", "Pitch is 120 x 80, origin top-left",
         f"x in [{min(xs):.1f}, {max(xs):.1f}], y in [{min(ys):.1f}, {max(ys):.1f}] "
         f"over {len(xs):,} located events",
         "PASS" if max(xs) <= 120.5 and max(ys) <= 80.5 else "CHECK"),
        ("A2", "Acting team always attacks left->right, both halves",
         f"{pct(fwd, len(all_shot_x))} of {len(all_shot_x):,} shots have x > 60; "
         + ", ".join(f"P{p}: {pct(sum(1 for x in v if x > 60), len(v))}"
                     for p, v in sorted(shot_x_by_period.items()) if v),
         "PASS" if len(all_shot_x) and fwd / len(all_shot_x) > 0.97 else "CHECK — flip needed"),
        ("A3", "Every event has possession + possession_team",
         f"missing possession {missing_possession}, missing possession_team "
         f"{missing_possession_team}, of {n_events:,} events",
         "PASS" if missing_possession == 0 and missing_possession_team == 0 else "CHECK"),
        ("A4", "Shots carry statsbomb_xg and mostly freeze_frame",
         f"{shots:,} shots — xg {pct(shots_with_xg, shots)}, "
         f"freeze_frame {pct(shots_with_ff, shots)}",
         "PASS" if shots and shots_with_xg / shots > 0.97 else "CHECK"),
        ("A5", "360 files exist only for flagged matches", "; ".join(a5_lines), "see result"),
        ("A6", "index is authoritative order; timestamp resets per period",
         f"index vs (period,timestamp) disagree in {index_disagreements}/{matches_checked} "
         f"matches; timestamp restarts at 00:00 each period in "
         f"{timestamp_resets}/{matches_checked}",
         "PASS — use index" if timestamp_resets else "CHECK"),
    ]

    out = ["# Probes — assumptions verified against the data", "",
           f"Generated by `ingest/01_probe_assumptions.py` over {matches_checked} sampled "
           "matches. Data source: StatsBomb.", "",
           "| ID | Assumption | Check | Result |", "|---|---|---|---|"]
    for i, a, ck, res in rows:
        out.append(f"| {i} | {a} | {ck} | **{res}** |")
    out += ["", "## Sampled matches", ""]
    for c, m in sample:
        out.append(f"- `{m['match_id']}` {c['competition_name']} {c['season_name']} — "
                   f"{m['home_team']['home_team_name']} v {m['away_team']['away_team_name']}"
                   f" (360: {m.get('match_status_360')})")
    out += ["", "## Event type frequency in the sample", "",
            "| type | count |", "|---|--:|"]
    for t, n in type_counter.most_common(25):
        out.append(f"| {t} | {n:,} |")
    out.append("")

    path = DOCS_DIR / "probes.md"
    path.write_text("\n".join(out), encoding="utf-8")

    print()
    for i, a, ck, res in rows:
        print(f"{i}  {res:<22} {ck}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
