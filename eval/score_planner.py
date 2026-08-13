"""Score the rule-based planner against the hand-written structured queries.

This is the Phase 6 number the plan asks for: "structured query" vs "natural
language query" retrieval quality. Here the natural-language side is parsed by
`core/planner.py` rather than an LLM, so the comparison is:

    hand    the filters and sequence_hint written by a human in queries.yaml
    auto    what the parser produces from the English `text` field alone

Both are then retrieved and judged identically, by the same programmatic
rubrics used in eval/score_retrieval.py. If `auto` matches `hand`, the English
front-end costs nothing in quality — which is the entire claim.

Run:
  python eval/score_planner.py
  python eval/score_planner.py --aggressive   # also emit must_include filters
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
from core.planner import Vocabulary, plan  # noqa: E402
from core.retrieval import Filters, Retriever, hydrate  # noqa: E402
from eval.judge import FILTER_DOMINATED, judge  # noqa: E402

HERE = Path(__file__).resolve().parent


def score_one(r, conn, qid, filters, hint, limit=10):
    out = r.search(conn, sequence_hint=hint, filters=filters, limit=limit)
    rows = {row["possession_uid"]: row for row in hydrate(conn, out["results"])}
    labels = [judge(qid, rows[u]) for u in out["results"] if u in rows]
    return {
        "p5": sum(labels[:5]) / 5 if labels else 0.0,
        "p10": sum(labels[:10]) / 10 if labels else 0.0,
        "mrr": next((1 / i for i, v in enumerate(labels, 1) if v), 0.0),
        "n": out["n_candidates"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggressive", action="store_true",
                    help="let the planner emit must_include token filters")
    args = ap.parse_args()

    queries = yaml.safe_load((HERE / "queries.yaml").read_text(encoding="utf-8"))["queries"]
    conn = db.connect()
    vocab = Vocabulary.from_db(conn)
    r = Retriever()
    r.model  # load MiniLM before timing

    rows, parse_ms = [], []
    agg = {"hand": {"p5": [], "p10": [], "mrr": []},
           "auto": {"p5": [], "p10": [], "mrr": []}}
    disc = {"hand": {"p5": [], "p10": [], "mrr": []},
            "auto": {"p5": [], "p10": [], "mrr": []}}
    # Queries where both sides produced identical filters — the only truly
    # apples-to-apples ranking comparison (see the caveat in the report).
    same_only = {"hand": {"p5": [], "p10": [], "mrr": []},
                 "auto": {"p5": [], "p10": [], "mrr": []}}
    exact_filters = 0

    for q in queries:
        qid = q["id"]
        hand_f = Filters(**(q.get("filters") or {}))

        t0 = time.perf_counter()
        p = plan(q["text"], vocab, aggressive=args.aggressive)
        parse_ms.append((time.perf_counter() - t0) * 1000)

        hand_dict = {k: v for k, v in hand_f.__dict__.items() if v not in (None, [], {})}
        auto_dict = {k: v for k, v in p.filters.__dict__.items() if v not in (None, [], {})}
        same = hand_dict == auto_dict
        exact_filters += same

        s_hand = score_one(r, conn, qid, hand_f, q["sequence_hint"])
        s_auto = score_one(r, conn, qid, p.filters, p.sequence_hint)

        for k in ("p5", "p10", "mrr"):
            agg["hand"][k].append(s_hand[k])
            agg["auto"][k].append(s_auto[k])
            if qid not in FILTER_DOMINATED:
                disc["hand"][k].append(s_hand[k])
                disc["auto"][k].append(s_auto[k])
                if same:
                    same_only["hand"][k].append(s_hand[k])
                    same_only["auto"][k].append(s_auto[k])

        rows.append({"id": qid, "text": q["text"], "same": same,
                     "hand": s_hand, "auto": s_auto,
                     "hint": p.sequence_hint, "filters": auto_dict,
                     "unmatched": p.unmatched})
        flag = "" if same else "  (filters differ)"
        print(f"  {qid}  hand P@5 {s_hand['p5']:.1f}   auto P@5 {s_auto['p5']:.1f}"
              f"   {q['text'][:40]}{flag}")

    n_disc = len(queries) - len(FILTER_DOMINATED)
    lines = [
        "# Planner evaluation — English to structured query, no LLM",
        "",
        "`core/planner.py` translates an English description into the same "
        "`Filters` + `sequence_hint` a human would write by hand. It is a rule "
        "parser: deterministic, free, no API key, and every filter is traceable "
        "to the phrase that produced it.",
        "",
        f"**Parse cost: {st.mean(parse_ms):.2f} ms mean, "
        f"{max(parse_ms):.2f} ms worst case.** No network, no tokens, no key.",
        "",
        f"**Filter agreement with the hand-written queries: "
        f"{exact_filters}/{len(queries)} exact.**",
        "",
        "Data source: StatsBomb.",
        "",
        "## Retrieval quality",
        "",
        f"Same rubrics and the same {n_disc} discriminating queries as "
        "`docs/retrieval_eval.md` (the 5 filter-dominated queries are excluded "
        "from the headline for the reason given there).",
        "",
        "| query source | P@5 | P@10 | MRR |",
        "|---|--:|--:|--:|",
        f"| hand-written structured | {st.mean(disc['hand']['p5']):.3f} | "
        f"{st.mean(disc['hand']['p10']):.3f} | {st.mean(disc['hand']['mrr']):.3f} |",
        f"| **parsed from English** | **{st.mean(disc['auto']['p5']):.3f}** | "
        f"{st.mean(disc['auto']['p10']):.3f} | {st.mean(disc['auto']['mrr']):.3f} |",
        "",
        f"All {len(queries)} queries: hand P@5 {st.mean(agg['hand']['p5']):.3f}, "
        f"parsed P@5 {st.mean(agg['auto']['p5']):.3f}.",
        "",
        "### One number in that table is not comparable",
        "",
        "Where the two sides produce *different* filters, they are not ranking "
        "the same candidate set, and the rubric cannot always tell which choice "
        "was right. `q07` is the clear case: the parser reads \"from the "
        "goalkeeper\" as `From Keeper` where the hand-written query chose "
        "`From Goal Kick`, and q07's rubric only tests whether the possession "
        "reaches the final third — it never checks the play pattern, because "
        "the filter was supposed to guarantee it. The parser scores 1.0 against "
        "0.2 on a distinction the rubric is blind to.",
        "",
        f"Restricted to the {len(same_only['auto']['p5'])} discriminating "
        f"queries where **both sides produced identical filters**, so only the "
        f"ranking differs: hand P@5 "
        f"{st.mean(same_only['hand']['p5']):.3f}, parsed P@5 "
        f"{st.mean(same_only['auto']['p5']):.3f}. That is the honest "
        "like-for-like figure, and it is the one to quote.",
        "",
        "## Per query",
        "",
        "| id | query | filters agree | hand P@5 | parsed P@5 |",
        "|---|---|:-:|--:|--:|",
    ]
    for row in rows:
        lines.append(f"| {row['id']} | {row['text']} | {'yes' if row['same'] else 'no'} | "
                     f"{row['hand']['p5']:.1f} | {row['auto']['p5']:.1f} |")

    # -- held-out paraphrases --------------------------------------------------
    para_path = HERE / "paraphrases.yaml"
    if para_path.exists():
        paras = yaml.safe_load(para_path.read_text(encoding="utf-8"))["paraphrases"]
        by_id = {r["id"]: r for r in rows}
        pr, expected_fail = [], []
        for item in paras:
            qid = item["id"]
            if qid not in by_id:
                continue
            pp = plan(item["text"], vocab, aggressive=args.aggressive)
            s = score_one(r, conn, qid, pp.filters, pp.sequence_hint)
            rec = {"id": qid, "text": item["text"], "p5": s["p5"],
                   "orig_p5": by_id[qid]["auto"]["p5"],
                   "hint": pp.sequence_hint, "unmatched": pp.unmatched}
            (expected_fail if item.get("expect_fail") else pr).append(rec)

        if pr:
            mean_para = st.mean(x["p5"] for x in pr)
            mean_orig = st.mean(x["orig_p5"] for x in pr)
            lines += [
                "", "## Held-out paraphrases — does it generalise?", "",
                "The rules above were written while looking at failures on the "
                "30 `text` fields, so that headline is an **in-sample** number. "
                "These paraphrases restate the same intents in deliberately "
                "different words and are judged by the original rubrics.",
                "",
                f"**{len(pr)} paraphrases: P@5 {mean_para:.3f}, versus "
                f"{mean_orig:.3f} for the original wording of the same queries "
                f"({mean_para - mean_orig:+.3f}).**",
                "",
                "| id | paraphrase | original P@5 | paraphrase P@5 |",
                "|---|---|--:|--:|",
            ]
            for x in sorted(pr, key=lambda z: z["p5"] - z["orig_p5"]):
                lines.append(f"| {x['id']} | {x['text']} | {x['orig_p5']:.1f} | "
                             f"{x['p5']:.1f} |")
            worst = [x for x in pr if x["p5"] < x["orig_p5"] - 0.19]
            if worst:
                lines += ["", "Where the paraphrase lost ground, the words the "
                          "parser could not place:", ""]
                for x in worst:
                    lines.append(f"- `{x['id']}` — ignored: *{x['unmatched'] or 'nothing'}*")
        if expected_fail:
            lines += ["", "### Known limits (included deliberately)", ""]
            for x in expected_fail:
                lines.append(f"- `{x['id']}` — \"{x['text']}\" scored "
                             f"{x['p5']:.1f}. Ignored: *{x['unmatched']}*")
            lines.append("")
            lines.append("A rule parser has no alias table, so a nickname the "
                         "database has never seen is simply not a team. An LLM "
                         "would resolve it. This is the concrete cost of the "
                         "trade, kept in the report rather than dropped from it.")

    lines += [
        "", "## Known limitations", "",
        "**Short-possession bias.** A thin hint retrieves thin possessions. "
        "Ask for \"Barcelona working the ball into the left half-space\" and the "
        "top hits are three-token, one-second fragments — which the rubric "
        "scores as relevant (they do end in F-LI) while a human would call them "
        "worthless. The cause is TF-IDF cosine similarity, not the parser: a "
        "3-token possession matching the hint exactly outscores a 20-token one "
        "containing the same tokens, because the longer vector is diluted. "
        "Padding the hint differently was tried and measured worse "
        "(P@5 0.680 -> 0.592), so it was reverted. The real fix is length "
        "normalisation in `core/retrieval.py`, which is a ranker change, not a "
        "planner change.",
        "",
        "**Vocabulary coverage is the whole ceiling.** The parser understands "
        "the terms in `core/planner.py` and nothing else. It has no alias table "
        "and no paraphrase ability beyond the synonym sets — see the PSG case "
        "above. An LLM would generalise where this does not; that is the "
        "trade, and it is why the `ignored` field is surfaced in the UI rather "
        "than hidden, so a user can see when their words were not understood.",
        "",
        "**The rubric cannot judge every disagreement.** Where the two filter "
        "sets differ, the programmatic rubric sometimes has no way to tell "
        "which was right (see q07 above). The like-for-like figure exists "
        "because of this.",
        "",
        "## What the parser produced", ""]
    for row in rows:
        lines += [f"**{row['id']}** — {row['text']}", "",
                  f"- filters: `{row['filters'] or 'none'}`",
                  f"- hint: `{row['hint']}`"]
        if row["unmatched"]:
            lines.append(f"- words ignored: *{row['unmatched']}*")
        lines.append("")

    out = DOCS_DIR / "planner_eval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[12:22]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
