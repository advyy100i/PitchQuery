"""Score the xG models on held-out competitions -> docs/benchmark.md + calibration.png.

The headline is the gap-closed figure from plan §8:

    gap_closed = (ll_base - ll_mine) / (ll_base - ll_sb)

Read it as: of the distance the distance+angle baseline would have to travel to
reach StatsBomb's production model, how much does mine cover. It is a ratio of
log-loss improvements, so it is only meaningful when StatsBomb actually beats
the baseline on this split — which the script checks rather than assumes.

Phrase it as "closes X% of the gap to StatsBomb's production model". Never
"beats StatsBomb". If a slice comes out ahead, name the slice and expect the
interviewer to suspect leakage — then show them the competition-level split.

Run:
  python models/evaluate_xg.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from core.config import DOCS_DIR, INDEX_DIR  # noqa: E402
from models.metrics import metrics  # noqa: E402

MODELS = [("distance+angle", "p_base"), ("mine (+context)", "p_mine"),
          ("StatsBomb (reference)", "p_sb")]


def calibration_plot(df: pd.DataFrame, path: Path, bins: int = 10) -> None:
    """Reliability curve. Discrimination without calibration is not an xG model:
    AUC only cares about ordering, and a model can rank perfectly while being
    systematically wrong about how many goals to expect."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect calibration")

    colours = {"p_base": "#e4572e", "p_mine": "#2e86ab", "p_sb": "#6a8f3c"}
    for label, col in MODELS:
        p = df[col].values
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() >= 20:
                xs.append(p[m].mean())
                ys.append(df["is_goal"].values[m].mean())
        ax.plot(xs, ys, "o-", color=colours[col], label=label, lw=1.8, ms=5)

    # Bin means top out around 0.4 — a 0-1 axis would squash every point into the
    # bottom-left corner and hide the differences that matter.
    hi = max(max(xs), max(ys)) * 1.15
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("predicted xG (bin mean)")
    ax.set_ylabel("observed goal rate")
    ax.set_title("Calibration on held-out competitions\n(equal-count bins, min 20 shots)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    for label, col in MODELS:
        ax2.hist(df[col], bins=40, histtype="step", lw=1.6,
                 color=colours[col], label=label)
    ax2.set_yscale("log")
    ax2.set_xlabel("predicted xG")
    ax2.set_ylabel("shots (log scale)")
    ax2.set_title("Prediction distribution")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.text(0.5, 0.01, "Data source: StatsBomb", ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    pred_path = INDEX_DIR / "xg_test_predictions.csv"
    if not pred_path.exists():
        print(f"no {pred_path} — run models/train_xg.py first")
        return
    df = pd.read_csv(pred_path)
    split = json.loads((INDEX_DIR / "xg_split.json").read_text(encoding="utf-8"))
    y = df["is_goal"].values

    res = {col: metrics(y, df[col].values) for _, col in MODELS}
    ll_base, ll_mine, ll_sb = (res["p_base"]["log_loss"], res["p_mine"]["log_loss"],
                               res["p_sb"]["log_loss"])

    denom = ll_base - ll_sb
    if denom <= 0:
        gap_note = (f"StatsBomb's model does **not** beat the distance+angle baseline "
                    f"on this split (log-loss {ll_sb:.4f} vs {ll_base:.4f}), so the "
                    f"gap-closed ratio is undefined and is not reported. That is a "
                    f"property of the split, not a result.")
        gap = None
    else:
        gap = (ll_base - ll_mine) / denom
        gap_note = (f"**My model closes {gap * 100:.0f}% of the log-loss gap** between a "
                    f"distance+angle baseline and StatsBomb's production model, on "
                    f"competitions neither model was trained on here.")

    calibration_plot(df, DOCS_DIR / "calibration.png")

    lines = [
        "# xG benchmark",
        "",
        gap_note,
        "",
        f"Held out: **{', '.join(split['test_comps'])}** "
        f"({split['n_test']:,} shots). Trained on {split['n_train']:,} shots across "
        f"{len(split['train_groups'])} other competition/seasons: "
        f"{', '.join(split['train_groups'])}.",
        "",
        "Split by competition/season, never by shot — shots inside a match share a "
        "game state and a shooter, so a random shot split leaks between train and "
        "test and inflates every number below. Penalties are excluded "
        "(fixed geometry, ~78% conversion, and they flatter AUC badly). "
        "`statsbomb_xg` is a comparison column only and is never a feature.",
        "",
        "Data source: StatsBomb.",
        "",
        "| model | log-loss | Brier | ECE | ROC-AUC | expected goals | observed | O/E | gap closed |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for label, col in MODELS:
        m = res[col]
        if col == "p_base":
            g = "0%"
        elif col == "p_sb":
            g = "100%"
        else:
            g = f"**{gap * 100:.0f}%**" if gap is not None else "n/a"
        lines.append(
            f"| {label} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} | "
            f"{m['auc']:.4f} | "
            f"{m['expected']:.1f} | {m['observed']} | "
            f"{m['observed'] / m['expected']:.3f} | {g} |")

    lines += ["", "![calibration](calibration.png)", "",
              "## By competition", "",
              "| competition/season | shots | goals | baseline | mine | StatsBomb |",
              "|---|--:|--:|--:|--:|--:|"]
    for g, sub in df.groupby("group"):
        lines.append(
            f"| {g} | {len(sub):,} | {int(sub['is_goal'].sum())} | "
            f"{log_loss(sub['is_goal'], np.clip(sub['p_base'], 1e-6, 1 - 1e-6), labels=[0, 1]):.4f} | "
            f"{log_loss(sub['is_goal'], np.clip(sub['p_mine'], 1e-6, 1 - 1e-6), labels=[0, 1]):.4f} | "
            f"{log_loss(sub['is_goal'], np.clip(sub['p_sb'], 1e-6, 1 - 1e-6), labels=[0, 1]):.4f} |")

    out = DOCS_DIR / "benchmark.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[8:16]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
