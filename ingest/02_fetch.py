"""Phase 1: selectively download event files for chosen competitions.

Everything is cached under PITCHQUERY_DATA, so re-running skips what you have.
Start small (--limit 50) and confirm the loader works before pulling everything.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", action="append", required=True,
                    metavar="COMP_ID:SEASON_ID", help="repeatable, e.g. --comp 55:43")
    ap.add_argument("--limit", type=int, default=None, help="max matches per competition")
    ap.add_argument("--with-360", action="store_true",
                    help="also fetch three-sixty files (10-40 MB each — only if you need them)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = ap.parse_args()

    targets = []
    for spec in args.comp:
        cid, sid = (int(v) for v in spec.split(":"))
        ms = _http.matches(cid, sid)
        if args.limit:
            ms = sorted(ms, key=lambda m: m.get("match_date", ""))[: args.limit]
        targets.append((cid, sid, ms))
        print(f"comp {cid} season {sid}: {len(ms)} matches selected")

    if args.dry_run:
        total = sum(len(ms) for _, _, ms in targets)
        print(f"\ndry run — would fetch {total} event files "
              f"(~{total * 4:.0f} MB) into {RAW_DIR}")
        return

    fetched = cached = failed = 0
    for cid, sid, ms in targets:
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
            if args.with_360 and m.get("match_status_360") == "available":
                _http.three_sixty(mid)
            if i % 25 == 0 or i == len(ms):
                print(f"  comp {cid}/{sid}: {i}/{len(ms)} "
                      f"({fetched} new, {cached} cached, {failed} missing)")

    size = sum(mb(p) for p in (RAW_DIR / "events").glob("*.json"))
    print(f"\ndone. {fetched} new, {cached} already cached, {failed} missing.")
    print(f"events cache: {size:.0f} MB in {RAW_DIR / 'events'}")


if __name__ == "__main__":
    main()
