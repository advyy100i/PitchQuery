"""Phase 6: compare this run's metrics against the committed baseline.

Prints a markdown table and exits 1 if anything broke a threshold, so a pull
request that makes retrieval worse cannot be merged on the strength of the diff
looking reasonable.

Three decisions worth stating, because each is the difference between a gate
that works and a gate that gets disabled:

  * Thresholds are one-sided. Only the direction that means "worse" fails. An
    improvement is reported and never blocks.

  * A metric missing from the baseline is reported as new, not as a failure.
    Adding a measurement should not require a two-step merge.

  * Different corpus fingerprints do not fail the build. eval/report.py stamps
    every metrics file with a row count and a hash of the possession uids; when
    they differ the two numbers were measured on different data and the
    comparison is meaningless, so the table says so instead of pretending a
    threshold was crossed. The one thing that WOULD be a silent disaster —
    comparing a fused P@5 against a sparse one — is caught separately, because
    `ranker` is part of the payload.

Run:
  python ci/compare_metrics.py                 # both reports
  python ci/compare_metrics.py retrieval       # one
  python ci/compare_metrics.py --out $GITHUB_STEP_SUMMARY
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.report import BASELINE_DIR, OUT_DIR  # noqa: E402

# metric -> (direction, tolerance, label)
#
# "direction" is the sign of a change that means the project got worse. The
# tolerances come from the plan and are deliberately loose enough to absorb
# tie-breaking wobble on a 25-query set and tight enough to catch a real loss:
# P@5 moves by 0.04 when a single result changes rank on a single query, so
# 0.02 catches anything systematic without firing on one.
THRESHOLDS = {
    "retrieval": {
        "p_at_5":         ("down", 0.02,  "P@5"),
        "p_at_10":        ("down", 0.02,  "P@10"),
        "mrr":            ("down", 0.03,  "MRR"),
        # Latency is the noisiest thing here — a shared CI runner can stall for
        # tens of milliseconds — so this is a smoke alarm for an algorithmic
        # change, not a benchmark.
        "p95_latency_ms": ("up",   50.0,  "p95 latency (ms)"),
    },
    "xg": {
        "logloss": ("up",   0.005, "log-loss"),
        "brier":   ("up",   0.002, "Brier"),
        "ece":     ("up",   0.005, "expected calibration error"),
        "auc":     ("down", 0.010, "ROC-AUC"),
    },
}

# Fields that describe the measurement rather than the result. A change in any
# of them invalidates the comparison instead of failing it.
CONTEXT = ("ranker", "scope", "n_queries", "n_discriminating", "n_shots")


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def compare(name: str) -> tuple:
    """Returns (markdown lines, failures, comparable)."""
    new_path, base_path = OUT_DIR / f"{name}.json", BASELINE_DIR / f"{name}.json"
    if not new_path.exists():
        return ([f"### {name}", "", f"No `{new_path.relative_to(Path.cwd()) if new_path.is_relative_to(Path.cwd()) else new_path}` "
                 f"— the eval step did not run."], [f"{name}: no metrics produced"], False)
    new = json.loads(new_path.read_text(encoding="utf-8"))
    if not base_path.exists():
        return ([f"### {name}", "",
                 f"No committed baseline yet. Measured this run: "
                 + ", ".join(f"`{k}` {fmt(new.get(k))}" for k in THRESHOLDS[name]),
                 "", "Commit it with `python eval/report.py " + name + "`."], [], False)
    base = json.loads(base_path.read_text(encoding="utf-8"))

    lines = [f"### {name}", ""]
    failures, notes = [], []

    for field in CONTEXT:
        if field in base and field in new and base[field] != new[field]:
            notes.append(f"`{field}` changed: {base[field]} → {new[field]}")
    same_corpus = base.get("corpus", {}).get("hash") == new.get("corpus", {}).get("hash")
    if not same_corpus:
        notes.append(
            f"corpus fingerprint differs "
            f"({base.get('corpus', {}).get('hash')} → {new.get('corpus', {}).get('hash')}), "
            f"so these were measured on different data — thresholds are reported "
            f"but not enforced")

    lines += ["| metric | baseline | this run | delta | threshold | |",
              "|---|--:|--:|--:|--:|:--|"]
    for key, (direction, tol, label) in THRESHOLDS[name].items():
        old, cur = base.get(key), new.get(key)
        if cur is None:
            lines.append(f"| {label} | {fmt(old)} | — | — | | not measured |")
            continue
        if old is None:
            lines.append(f"| {label} | — | {fmt(cur)} | — | | new |")
            continue
        delta = cur - old
        worse = delta > tol if direction == "up" else delta < -tol
        if worse and same_corpus:
            failures.append(f"{name}.{key}: {label} {fmt(old)} → {fmt(cur)} "
                            f"({delta:+.4f}, limit {tol})")
            mark = "**FAIL**"
        elif worse:
            mark = "over limit (not enforced)"
        elif (delta < 0) is (direction == "up") and abs(delta) > 1e-9:
            mark = "improved"
        else:
            mark = "ok"
        sign = "up to" if direction == "up" else "down to"
        lines.append(f"| {label} | {fmt(old)} | {fmt(cur)} | {delta:+.4f} | "
                     f"{sign} {tol} | {mark} |")

    if notes:
        lines += [""] + [f"> {n}" for n in notes]
    return lines, failures, same_corpus


def main():
    # A Windows console is cp1252 and this table is UTF-8, so printing it raised
    # UnicodeEncodeError and the gate exited 1 for the wrong reason — a metric
    # gate that fails on its own output is worse than no gate. Widen the stream
    # rather than downgrade the characters: the same text goes to the PR comment.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=None,
                    choices=list(THRESHOLDS) + [[]],
                    help="which reports to compare (default: all of them)")
    ap.add_argument("--out", default=None,
                    help="also append the table here (e.g. $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()
    names = args.names or list(THRESHOLDS)

    body, failures = ["## Metric gate", ""], []
    comparable = True
    for name in names:
        lines, fails, same = compare(name)
        body += lines + [""]
        failures += fails
        comparable = comparable and same

    if failures:
        body += ["**Blocked.** " + str(len(failures)) + " metric(s) outside tolerance:", ""]
        body += [f"- {f}" for f in failures]
    elif not comparable:
        # Not "all within tolerance". Some of them may not be, and the run said
        # so a few lines up; claiming a pass here would contradict the table.
        body += ["**Not compared.** This run and the baseline were measured on "
                 "different data, so the thresholds above are reported and not "
                 "enforced. Nothing is blocked and nothing is verified."]
    else:
        body += ["All metrics within tolerance."]

    # Describes the BASELINE, which is always the fixture, rather than whatever
    # this run happened to measure. Reading the size off the current run made
    # the footer claim a local full-corpus run was a 67k "sample" of itself.
    try:
        base = json.loads((BASELINE_DIR / "retrieval.json").read_text(encoding="utf-8"))
        size = f"{base['corpus']['rows']:,}-possession "
    except Exception:
        size = ""
    body += ["",
             f"_Baselines are measured against `eval/fixtures/corpus.sql.gz`, a "
             f"{size}sample whose relevant rows are chosen by the rubrics in "
             "`eval/judge.py` — not the full 67k corpus, and never chosen by what "
             "the ranker returns. See `eval/fixtures/make_fixture.py`._"]

    text = "\n".join(body)
    print(text)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
