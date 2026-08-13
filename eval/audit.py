"""Measure how often the programmatic rubrics agree with a human.

The predicates in judge.py replace an evening of hand-judging. This is the price
of that: a small blind audit that says how trustworthy they are. Without it,
"programmatic relevance rubric" is an unverified claim.

Batched by query: one rubric at the top of each screen, several possessions
below it, one answer line for all of them. Judging five examples against a rule
you have just read is much faster than re-reading the rule five times.

You are never shown what the predicate decided, so you cannot anchor to it.

Run it in a real terminal:
  python eval/audit.py                      # 40 items, 5 per screen, ~10 min
  python eval/audit.py --n 20 --per-query 5 # shorter
  python eval/audit.py --report-only        # just the agreement number
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
from eval.judge import judge  # noqa: E402

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit_labels.yaml"

PROMPT = """
  Type the NUMBERS of the ones that match the rubric, separated by spaces.
    e.g.  1 3 4     ->  1, 3 and 4 match; the rest do not
          <enter>   ->  none of them match
          a         ->  all of them match
          s         ->  skip this screen entirely
          q         ->  save and quit
"""


def load() -> dict:
    if AUDIT.exists():
        return yaml.safe_load(AUDIT.read_text(encoding="utf-8")) or {}
    return {}


def save(d: dict) -> None:
    AUDIT.write_text(
        "# Human audit of the programmatic rubrics in eval/judge.py.\n"
        "# 1 = relevant, 0 = not. Written by eval/audit.py.\n"
        + yaml.safe_dump(d, sort_keys=True, default_flow_style=False),
        encoding="utf-8")


def parse(answer: str, n: int):
    """Returns a set of 1-based indices marked relevant, or a control string."""
    a = answer.strip().lower()
    if a in ("q", "s"):
        return a
    if a == "a":
        return set(range(1, n + 1))
    if a == "":
        return set()
    try:
        picked = {int(tok) for tok in a.replace(",", " ").split()}
    except ValueError:
        return None
    if any(i < 1 or i > n for i in picked):
        return None
    return picked


def show_item(i: int, row: dict) -> None:
    tag = "GOAL" if row["ended_in_goal"] else ("shot" if row["ended_in_shot"] else "")
    print(f"\n  {i}. {row['team']} v {row['opponent']}  —  {row['play_pattern']}, "
          f"{row['n_events']} tokens{', ' + tag if tag else ''}")
    for line in textwrap.wrap(row["token_string"], width=86):
        print(f"       {line}")


def report(labels: dict, rows: dict, queries: dict) -> None:
    agree = disagree = 0
    misses = []
    for key, human in labels.items():
        qid, uid = key.split("|", 1)
        row = rows.get(uid)
        if row is None:
            continue
        auto = judge(qid, row)
        if auto == human:
            agree += 1
        else:
            disagree += 1
            misses.append((qid, uid, human, auto, row))

    total = agree + disagree
    if not total:
        print("nothing judged yet")
        return
    print(f"\n{'=' * 90}")
    print(f"AGREEMENT: {agree}/{total} = {agree / total * 100:.0f}%")
    print(f"{'=' * 90}")
    if misses:
        print(f"\n{len(misses)} disagreement(s) — worth reading, they show where the "
              f"rubric and a human part company:\n")
        for qid, uid, human, auto, row in misses:
            print(f"  {qid} {uid}   you: {'relevant' if human else 'not relevant'}   "
                  f"rubric: {'relevant' if auto else 'not relevant'}")
            print(f"    query: {queries[qid]['text']}")
            for line in textwrap.wrap(row["token_string"], width=84):
                print(f"    {line}")
            print()
        print("  If the predicate is wrong, fix it in eval/judge.py and rerun the audit.")
        print("  If your call was the loose one, leave it — the disagreement rate is the")
        print("  number worth reporting either way.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="total items to audit")
    ap.add_argument("--per-query", type=int, default=5, help="items shown per screen")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    queries = {q["id"]: q for q in yaml.safe_load(
        (HERE / "queries.yaml").read_text(encoding="utf-8"))["queries"]}
    labels = load()
    conn = db.connect()
    r = Retriever()
    rng = random.Random(args.seed)

    # Sample from what the system actually returns. Auditing random corpus rows
    # would mostly surface obvious non-matches and flatter the agreement number.
    print("building the audit sample (loads MiniLM once, ~15s)...")
    batches = []
    for qid, q in queries.items():
        f = Filters(**(q.get("filters") or {}))
        out = r.search(conn, sequence_hint=q["sequence_hint"], filters=f, limit=10)
        pool = list(dict.fromkeys(out["sparse"][:6] + out["dense"][:6]))
        rng.shuffle(pool)
        if pool:
            batches.append((qid, pool[: args.per_query]))
    rng.shuffle(batches)

    # Every uid ever judged, not just this run's sample. The sample depends on
    # --per-query and on the ranking, so a label given under different settings
    # would otherwise be silently dropped from the agreement number.
    all_uids = list({u for _, pool in batches for u in pool}
                    | {k.split("|", 1)[1] for k in labels})
    all_rows = {row["possession_uid"]: row for row in hydrate(conn, all_uids)}

    if args.report_only:
        report(labels, all_rows, queries)
        return

    # Drop anything already judged, then take enough screens to reach --n.
    pending, budget = [], args.n
    for qid, pool in batches:
        todo = [u for u in pool if f"{qid}|{u}" not in labels]
        if not todo or budget <= 0:
            continue
        todo = todo[:budget]
        pending.append((qid, todo))
        budget -= len(todo)

    if not pending:
        print("audit already complete for this sample")
        report(labels, all_rows, queries)
        return

    total_items = sum(len(t) for _, t in pending)
    print(f"\n{total_items} possessions across {len(pending)} queries, "
          f"{len(pending)} screens.")
    print("You are NOT shown what the rubric decided. Judge the football only.")

    done = 0
    for screen, (qid, todo) in enumerate(pending, 1):
        q = queries[qid]
        rows = [all_rows[u] for u in todo if u in all_rows]
        if not rows:
            continue

        print("\n" + "=" * 90)
        print(f"SCREEN {screen}/{len(pending)}   ({done}/{total_items} judged)")
        print(f'QUERY: "{q["text"]}"')
        print("RUBRIC: " + "\n        ".join(
            textwrap.wrap(" ".join(q["rubric"].split()), width=82)))
        print("=" * 90)
        for i, row in enumerate(rows, 1):
            show_item(i, row)
        print(PROMPT)

        while True:
            got = parse(input("  which ones match? "), len(rows))
            if got is None:
                print("  didn't understand that. Numbers, or blank, or a / s / q.")
                continue
            break

        if got == "q":
            save(labels)
            report(labels, all_rows, queries)
            print(f"\nsaved to {AUDIT} — rerun any time to continue")
            return
        if got == "s":
            continue

        for i, row in enumerate(rows, 1):
            labels[f"{qid}|{row['possession_uid']}"] = 1 if i in got else 0
        done += len(rows)
        save(labels)

    save(labels)
    report(labels, all_rows, queries)
    print(f"\nsaved to {AUDIT}. Now run:  python eval\\score_retrieval.py")


if __name__ == "__main__":
    main()
