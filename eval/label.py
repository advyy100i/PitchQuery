"""Hand-judge retrieval results. This is the evening of work that turns a vibe
into a metric (plan §10, Phase 3).

Judging is POOLED: for each query the tool collects the top-N from the sparse
ranker, the dense ranker and the fused list, deduplicates, and shows them in a
shuffled order with the ranker that produced them hidden. You cannot tell which
system surfaced a result while you judge it, which is the point — otherwise the
labels drift towards whichever ranker you expect to win.

Judge against the `rubric` field in queries.yaml, which was written before any
results existed. The rubric is shown above every batch.

Progress is written after every keystroke, so quit any time with 'q' and rerun
to pick up where you left off.

Run it in a real terminal (it reads stdin):
  python eval/label.py
  python eval/label.py --query q07        # redo one query
  python eval/label.py --pool 8           # judge fewer per query
"""
import argparse
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from core import db  # noqa: E402
from core.retrieval import Filters, Retriever, hydrate  # noqa: E402

HERE = Path(__file__).resolve().parent
QUERIES = HERE / "queries.yaml"
LABELS = HERE / "labels.yaml"

HELP = """
  y = relevant     n = not relevant     s = skip (undecided)
  ? = show the rubric again             q = save and quit
"""


def load_queries() -> list:
    return yaml.safe_load(QUERIES.read_text(encoding="utf-8"))["queries"]


def load_labels() -> dict:
    if LABELS.exists():
        return yaml.safe_load(LABELS.read_text(encoding="utf-8")) or {}
    return {}


def save_labels(labels: dict) -> None:
    LABELS.write_text(
        "# Hand-judged relevance, pooled over sparse/dense/fused.\n"
        "# 1 = relevant, 0 = not relevant. Written by eval/label.py.\n"
        + yaml.safe_dump(labels, sort_keys=True, default_flow_style=False),
        encoding="utf-8")


def show(row: dict, n: int, total: int) -> None:
    flags = []
    if row["ended_in_goal"]:
        flags.append("GOAL")
    elif row["ended_in_shot"]:
        flags.append("shot")
    if row["xg_sum"]:
        flags.append(f"xg {row['xg_sum']:.2f}")
    print(f"\n[{n}/{total}] {row['possession_uid']}  "
          f"{row['team']} v {row['opponent']}  {row['competition']} {row['season']}")
    print(f"    {row['play_pattern']}, {row['n_events']} tokens, "
          f"{row['duration_s']:.0f}s  {'  '.join(flags)}")
    for line in textwrap.wrap(row["token_string"], width=96):
        print(f"    {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", action="append", help="only these query ids")
    ap.add_argument("--pool", type=int, default=10, help="results per ranker to pool")
    ap.add_argument("--seed", type=int, default=7, help="shuffle seed, for reproducibility")
    args = ap.parse_args()

    queries = load_queries()
    if args.query:
        wanted = set(args.query)
        queries = [q for q in queries if q["id"] in wanted]

    labels = load_labels()
    conn = db.connect()
    r = Retriever()
    rng = random.Random(args.seed)

    for q in queries:
        qid = q["id"]
        labels.setdefault(qid, {})
        f = Filters(**(q.get("filters") or {}))
        out = r.search(conn, sequence_hint=q["sequence_hint"], filters=f,
                       limit=args.pool, pool=max(50, args.pool * 5))

        pool = list(dict.fromkeys(
            out["sparse"][:args.pool] + out["dense"][:args.pool] + out["results"][:args.pool]))
        todo = [u for u in pool if u not in labels[qid]]
        if not todo:
            print(f"\n{qid}: already judged ({len(labels[qid])} labels)")
            continue

        rng.shuffle(todo)
        rows = {row["possession_uid"]: row for row in hydrate(conn, todo)}

        print("\n" + "=" * 98)
        print(f"{qid}  \"{q['text']}\"")
        print(f"  filters: {q.get('filters') or 'none'}")
        print(f"  candidates passing filters: {out['n_candidates']:,}")
        print("  rubric: " + " ".join(q["rubric"].split()))
        print("=" * 98)
        print(HELP)

        for i, uid in enumerate(todo, 1):
            row = rows.get(uid)
            if row is None:
                continue
            show(row, i, len(todo))
            while True:
                ans = input("    relevant? [y/n/s/?/q] ").strip().lower()
                if ans == "?":
                    print("  rubric: " + " ".join(q["rubric"].split()))
                    continue
                if ans == "q":
                    save_labels(labels)
                    print(f"\nsaved {sum(len(v) for v in labels.values())} labels to {LABELS}")
                    return
                if ans in ("y", "n", "s"):
                    break
                print(HELP)
            if ans != "s":
                labels[qid][uid] = 1 if ans == "y" else 0
                save_labels(labels)

    save_labels(labels)
    judged = sum(len(v) for v in labels.values())
    rel = sum(sum(v.values()) for v in labels.values())
    print(f"\ndone. {judged} labels over {len(labels)} queries, {rel} marked relevant.")
    print(f"now run:  python eval/score_retrieval.py")


if __name__ == "__main__":
    main()
