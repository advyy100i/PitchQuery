"""Phase 6: eval scripts write JSON as well as markdown.

Markdown is for a human reading the repo. JSON is for `ci/compare_metrics.py`,
which has to compare this run against a committed baseline and fail a pull
request — and parsing numbers back out of a markdown table is how that job
starts working and then breaks the first time a column moves.

Every payload carries a corpus fingerprint. That is this project's answer to
"how do I know these two numbers were measured on the same data", and it is
deliberately not DVC: a row count plus an MD5 of the sorted possession uids is
one query, costs nothing, and travels inside the metrics file it describes.
A comparison across two different fingerprints is not a regression, it is a
different experiment, and compare_metrics.py says so rather than failing the
build over it.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402

OUT_DIR = REPO_ROOT / "eval" / "out"
BASELINE_DIR = REPO_ROOT / "eval" / "baselines"


def corpus_fingerprint(conn=None) -> dict:
    """{'rows': n, 'hash': 'ab12cd34'} for the possession corpus.

    Cheap enough to compute on every eval run: one indexed scan of a text
    column. The hash is truncated to eight hex characters — enough to notice a
    different corpus, short enough to read in a PR comment.
    """
    close = False
    if conn is None:
        from core import db
        try:
            conn = db.connect()
            close = True
        except Exception:
            return {"rows": None, "hash": "no-database"}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM possessions")
            rows = cur.fetchone()[0]
            cur.execute("SELECT possession_uid FROM possessions ORDER BY possession_uid")
            h = hashlib.md5()
            for (uid,) in cur:
                h.update(uid.encode("ascii"))
        return {"rows": int(rows), "hash": h.hexdigest()[:8]}
    finally:
        if close:
            conn.close()


def write(name: str, payload: dict, conn=None) -> Path:
    """Write eval/out/<name>.json with the fingerprint and timestamp attached."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["corpus"] = corpus_fingerprint(conn)
    body["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def load_baseline(name: str) -> dict:
    path = BASELINE_DIR / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def promote_to_baseline(name: str) -> Path:
    """Copy eval/out/<name>.json over eval/baselines/<name>.json.

    A separate, explicit step. If the eval scripts wrote the baseline themselves
    there would be no gate at all — every run would agree with itself.
    """
    src = OUT_DIR / f"{name}.json"
    dst = BASELINE_DIR / f"{name}.json"
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"baseline updated: {dst}")
    return dst


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="promote an eval run to the committed baseline")
    ap.add_argument("name", nargs="+", choices=["retrieval", "xg", "planner"],
                    help="which metrics files to promote")
    for n in ap.parse_args().name:
        promote_to_baseline(n)
