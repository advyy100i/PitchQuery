"""Phase 1: selectively download event files for chosen competitions.

Everything is cached under PITCHQUERY_DATA, so re-running skips what you have.
Start small (--limit 50) and confirm the loader works before pulling everything.

`main()` is a plain function, not an argparse wrapper: pipeline/flows.py imports
it and reads the returned dict. A DAG built out of `subprocess.run` cannot pass
anything between steps except an exit code, which is how orchestration ends up
re-deriving in step N what step N-1 already knew.

Run:
  python ingest/02_fetch.py --comp 55:43 --limit 20 --dry-run
  python ingest/02_fetch.py --comp 55:43 --comp 43:106
  python ingest/02_fetch.py --comp 55:43 --with-360     # also pull three-sixty files (BIG)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import RAW_DIR  # noqa: E402
from ingest import _http  # noqa: E402


def mb(path: Path) -> float:
    return path.stat().st_size / 1e6 if path.exists() else 0.0


def parse_comp(spec: str) -> tuple:
    """'55:43' -> (55, 43)."""
    cid, sid = (int(v) for v in spec.split(":"))
    return cid, sid


def main(comps: list, *, limit: int = None, with_360: bool = False,
         dry_run: bool = False, since: dict = None) -> dict:
    """Fetch event files for `comps` (['55:43', '43:106', ...]).

    `since` maps (competition_id, season_id) -> the highest match id already
    loaded, from pipeline/watermark.py. Matches at or below it are dropped here
    rather than downstream, so the second run of a flow does no work at all
    instead of doing all of it and discovering nothing changed (Phase 2).

    Returns {"rows": new files fetched, "match_ids": every match in scope,
             "cached", "missing", "by_comp"}. `match_ids` is what the loader
    takes as its argument — it is the list of matches this run is responsible
    for, cached or not, so a rerun after a failed load still loads them.
    """
    since = since or {}
    targets = []
    for spec in comps:
        cid, sid = parse_comp(spec)
        ms = _http.matches(cid, sid)
        floor = since.get((cid, sid))
        held = 0
        if floor is not None:
            before = len(ms)
            ms = [m for m in ms if m["match_id"] > floor]
            held = before - len(ms)
        if limit:
            # By match id, not match date. The watermark is a high-water mark
            # over ids, so a limited run has to take the lowest ids available —
            # taking the earliest *dates* could skip an id and then advance the
            # mark past it, and that match would never be loaded again.
            ms = sorted(ms, key=lambda m: m["match_id"])[: limit]
        targets.append((cid, sid, ms))
        note = f" ({held} at or below watermark {floor})" if floor is not None else ""
        print(f"comp {cid} season {sid}: {len(ms)} matches selected{note}")

    if dry_run:
        total = sum(len(ms) for _, _, ms in targets)
        print(f"\ndry run — would fetch {total} event files "
              f"(~{total * 4:.0f} MB) into {RAW_DIR}")
        return {"rows": 0, "match_ids": [], "cached": 0, "missing": 0,
                "by_comp": {}, "dry_run": True}

    fetched = cached = failed = 0
    by_comp: dict = {}
    for cid, sid, ms in targets:
        got = []
        for i, m in enumerate(ms, 1):
            mid = m["match_id"]
            local = RAW_DIR / f"events/{mid}.json"
            already = local.exists()
            ev = _http.events(mid)
            if ev is None:
                failed += 1
                print(f"  ! {mid} missing upstream")
                continue
            cached += already
            fetched += not already
            got.append(mid)
            if with_360 and m.get("match_status_360") == "available":
                _http.three_sixty(mid)
            if i % 25 == 0 or i == len(ms):
                print(f"  comp {cid}/{sid}: {i}/{len(ms)} "
                      f"({fetched} new, {cached} cached, {failed} missing)")
        by_comp[f"{cid}:{sid}"] = sorted(got)

    size = sum(mb(p) for p in (RAW_DIR / "events").glob("*.json"))
    print(f"\ndone. {fetched} new, {cached} already cached, {failed} missing.")
    print(f"events cache: {size:.0f} MB in {RAW_DIR / 'events'}")

    match_ids = sorted({mid for ids in by_comp.values() for mid in ids})
    return {"rows": fetched, "match_ids": match_ids, "cached": cached,
            "missing": failed, "by_comp": by_comp}


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", action="append", required=True,
                    metavar="COMP_ID:SEASON_ID", help="repeatable, e.g. --comp 55:43")
    ap.add_argument("--limit", type=int, default=None, help="max matches per competition")
    ap.add_argument("--with-360", action="store_true",
                    help="also fetch three-sixty files (10-40 MB each — only if you need them)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = ap.parse_args()
    main(args.comp, limit=args.limit, with_360=args.with_360, dry_run=args.dry_run)


if __name__ == "__main__":
    cli()
