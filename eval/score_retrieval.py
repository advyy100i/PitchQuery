"""Score the three rankers against the programmatic rubrics -> docs/retrieval_eval.md.

Reports P@5, P@10, MRR and latency for sparse, dense and fused separately.
Scoring them separately is the point: plan §12 claims the sparse n-gram ranker
beats the dense one because the tokens are a controlled vocabulary rather than
natural language, and that dense earns its place only in fusion. That is a claim
about numbers, so produce the numbers.

Relevance comes from eval/judge.py, applied to every retrieved possession.
Because the rubric is a function rather than a pooled sample, there are no
unjudged results — every returned item gets a label, and recall against the
full corpus is computable too.

Run:
  python eval/score_retrieval.py
  python eval/score_retrieval.py --repeats 5
"""
import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from core import db  # noqa: E402
from core.config import DOCS_DIR  # noqa: E402
from core.retrieval import Filters, Retriever, hydrate  # noqa: E402
from eval.judge import FILTER_DOMINATED, judge  # noqa: E402

HERE = Path(__file__).resolve().parent
RANKERS = ("sparse", "dense", "fused")


def pct(values, p):
    if not values:
        return 0.0
    vs = sorted(values)
    return vs[min(int(round(p / 100 * (len(vs) - 1))), len(vs) - 1)]


def corpus_relevant(conn, qid: str, f: Filters) -> int:
    """How many possessions in the whole corpus satisfy the rubric — the
    denominator for recall, and a sanity check that the query is answerable."""
    sql, params = f.where()
    with conn.cursor() as cur:
        cur.execute(f"SELECT possession_uid, token_string, zone_path, n_events, "
                    f"ended_in_shot, ended_in_goal, xg_sum, play_pattern "
                    f"FROM possessions WHERE {sql}", params)
        cols = [d.name for d in cur.description]
        return sum(judge(qid, dict(zip(cols, r))) for r in cur.fetchall())


def audit_agreement(conn) -> str:
    """How far the rubrics were verified against a human, with the direction of
    the disagreements — which matters more than the rate. A rubric that is
    stricter than a human understates precision; one that is looser inflates it.
    """
    path = HERE / "audit_labels.yaml"
    if not path.exists():
        return ("**Not yet audited.** The rubrics have not been checked against human "
                "judgement — run `python eval/audit.py` before quoting these numbers.")
    labels = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not labels:
        return "**Not yet audited.** Run `python eval/audit.py`."

    rows = {r["possession_uid"]: r
            for r in hydrate(conn, list({k.split("|", 1)[1] for k in labels}))}
    agree = strict = loose = 0
    for key, human in labels.items():
        qid, uid = key.split("|", 1)
        row = rows.get(uid)
        if row is None:
            continue
        auto = judge(qid, row)
        if auto == human:
            agree += 1
        elif human == 1:
            strict += 1        # human said relevant, rubric said no
        else:
            loose += 1         # rubric said relevant, human said no
    total = agree + strict + loose
    if not total:
        return "**Not yet audited.**"

    bias = ("The rubrics are net **stricter** than the human, so precision here is a "
            "floor rather than an inflated figure"
            if strict > loose else
            "The rubrics are net **looser** than the human, so precision here is "
            "optimistic and should be read as an upper bound"
            if loose > strict else
            "Disagreements are balanced in both directions, so there is no systematic "
            "bias in either direction")
    return (
        f"**Human agreement: {agree}/{total} = {agree / total * 100:.0f}%** on a blind "
        f"{total}-item audit (`eval/audit.py`) — the auditor was never shown what the "
        f"predicate decided. Of the {strict + loose} disagreements, {strict} were the "
        f"rubric refusing something the human accepted and {loose} the reverse. {bias}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    queries = yaml.safe_load((HERE / "queries.yaml").read_text(encoding="utf-8"))["queries"]
    conn = db.connect()
    r = Retriever()
    r.model  # load MiniLM before timing anything, or query 1 pays 12 s of import

    scores = {k: {"p5": [], "p10": [], "mrr": []} for k in RANKERS}
    disc = {k: {"p5": [], "p10": [], "mrr": []} for k in RANKERS}
    latency = {k: [] for k in RANKERS}
    per_query = []

    for q in queries:
        qid = q["id"]
        f = Filters(**(q.get("filters") or {}))

        for _ in range(args.repeats):
            t0 = time.perf_counter()
            r.search(conn, sequence_hint=q["sequence_hint"], filters=f,
                     limit=args.limit, use_dense=False)
            latency["sparse"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            out = r.search(conn, sequence_hint=q["sequence_hint"], filters=f,
                           limit=args.limit)
            latency["fused"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            r.dense_rank(conn, q["sequence_hint"], filters=f, limit=50)
            latency["dense"].append((time.perf_counter() - t0) * 1000)

        ranked = {"sparse": out["sparse"], "dense": out["dense"], "fused": out["results"]}
        rows = {row["possession_uid"]: row
                for row in hydrate(conn, list({u for lst in ranked.values() for u in lst}))}

        row = {"id": qid, "text": q["text"],
               "n_relevant": corpus_relevant(conn, qid, f),
               "n_candidates": out["n_candidates"]}
        for k in RANKERS:
            labels = [judge(qid, rows[u]) for u in ranked[k] if u in rows]
            p5 = sum(labels[:5]) / 5 if labels else 0.0
            p10 = sum(labels[:10]) / 10 if labels else 0.0
            mrr = next((1 / i for i, v in enumerate(labels, 1) if v), 0.0)
            scores[k]["p5"].append(p5)
            scores[k]["p10"].append(p10)
            scores[k]["mrr"].append(mrr)
            if qid not in FILTER_DOMINATED:
                disc[k]["p5"].append(p5)
                disc[k]["p10"].append(p10)
                disc[k]["mrr"].append(mrr)
            row[f"{k}_p5"] = p5
        per_query.append(row)
        print(f"  {qid} sparse {row['sparse_p5']:.1f}  dense {row['dense_p5']:.1f}  "
              f"fused {row['fused_p5']:.1f}   {q['text'][:46]}")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM possessions")
        corpus_size = cur.fetchone()[0]

    n_disc = len(queries) - len(FILTER_DOMINATED)
    lines = [
        "# Retrieval evaluation",
        "",
        f"{len(queries)} queries over a {corpus_size:,}-possession corpus. "
        "Relevance is decided by the programmatic rubrics in `eval/judge.py`, one "
        "predicate per query, translated from the written rubric in "
        "`eval/queries.yaml` before any results were seen.",
        "",
        audit_agreement(conn),
        "",
        "Data source: StatsBomb.",
        "",
        "## Headline",
        "",
        f"Excluding the {len(FILTER_DOMINATED)} queries whose SQL filter alone already "
        f"satisfies the rubric for >=90% of rows it returns "
        f"({', '.join(sorted(FILTER_DOMINATED))}) — ranking cannot affect those, so "
        f"including them only inflates the mean. **{n_disc} discriminating queries:**",
        "",
        "| ranker | P@5 | P@10 | MRR | p50 latency | p95 latency |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for k in RANKERS:
        lines.append(
            f"| {k} | **{st.mean(disc[k]['p5']):.3f}** | {st.mean(disc[k]['p10']):.3f} | "
            f"{st.mean(disc[k]['mrr']):.3f} | {pct(latency[k], 50):.0f} ms | "
            f"{pct(latency[k], 95):.0f} ms |")

    lines += ["", f"All {len(queries)} queries, for completeness:", "",
              "| ranker | P@5 | P@10 | MRR |", "|---|--:|--:|--:|"]
    for k in RANKERS:
        lines.append(f"| {k} | {st.mean(scores[k]['p5']):.3f} | "
                     f"{st.mean(scores[k]['p10']):.3f} | {st.mean(scores[k]['mrr']):.3f} |")

    lines += ["", "## Per query (P@5)", "",
              "| id | query | relevant in corpus | sparse | dense | fused |",
              "|---|---|--:|--:|--:|--:|"]
    for p in per_query:
        star = " *" if p["id"] in FILTER_DOMINATED else ""
        lines.append(f"| {p['id']}{star} | {p['text']} | {p['n_relevant']:,} | "
                     f"{p['sparse_p5']:.1f} | {p['dense_p5']:.1f} | {p['fused_p5']:.1f} |")
    lines += ["", "`*` = filter-dominated, excluded from the headline mean."]

    out_path = DOCS_DIR / "retrieval_eval.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[10:16]))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
