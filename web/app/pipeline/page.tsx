"use client";

/**
 * The operational view: did the pipeline run, is the data there, which model is
 * champion, has the data moved, and what are people searching for.
 *
 * Five sections in the order you would actually check them, reading one cached
 * /ops call. This replaced a Streamlit app — same five panels, same degradation
 * rules, now in the app it reports on, with the app's type system and the app's
 * design, deployed by the same push.
 *
 * The point of the panel-level hints is that a hosted deployment cannot answer
 * three of the five, and that is the true state rather than a failure: the
 * watermark and the dbt schemas never ship to Neon, and MLflow is a SQLite file
 * on a laptop. Each panel says which and why. A page that goes blank because
 * one dependency is absent gets closed and never opened again.
 */

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { DivergingBars, TimeSeries } from "../../components/Charts";
import { ops as fetchOps, type Ops } from "../../lib/ops";

const S1 = "var(--s1)";
const S2 = "var(--s2)";

const int = (n: number | null | undefined) =>
  n === null || n === undefined ? "—" : n.toLocaleString();
const dec = (n: number | null | undefined, d = 4) =>
  n === null || n === undefined ? "—" : n.toFixed(d);
const pct = (n: number | null | undefined) =>
  n === null || n === undefined ? "—" : `${Math.round(n * 100)}%`;
const stamp = (s: string | null | undefined) =>
  !s ? "—" : s.replace("T", " ").slice(0, 16);
// models/tracking.py tags a run trained on an uncommitted tree as "<sha>-dirty",
// and the first eight characters of that end in a hyphen presented as hash.
const sha = (s: string) => s.slice(0, 8).replace(/[^0-9a-f]+$/, "");

function Panel({ title, note, children }: {
  title: string; note?: string; children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {note && <p className="lede panel-note">{note}</p>}
      {children}
    </section>
  );
}

/** Why a panel is empty, in the panel, where the answer is needed. */
function Hint({ text, kind = "info" }: { text?: string | null; kind?: "info" | "warn" }) {
  if (!text) return null;
  return <p className={`hint ${kind}`}>{text}</p>;
}

function Tiles({ items }: { items: [string, string][] }) {
  return (
    <dl className="tiles">
      {items.map(([k, v]) => (
        <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}

export default function Pipeline() {
  const [data, setData] = useState<Ops | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState(0);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      setData(await fetchOps());
      setError(null);
    } catch (e) {
      setError(
        "Cannot reach the API. Start it with:  uvicorn api.main:app --port 8000" +
        `  (${e})`);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const runs = data?.pipeline.runs ?? [];
  const layers = data?.layers.tables ?? [];
  const champ = data?.champion;
  const drift = data?.drift.reports ?? [];
  const q = data?.queries;
  const shown = drift[Math.min(report, drift.length - 1)];

  const layerTotal = (layer: string) => {
    const part = layers.filter((t) => t.layer === layer && t.rows !== null);
    return part.length ? part.reduce((a, t) => a + (t.rows ?? 0), 0) : null;
  };

  const metrics = champ?.metrics ?? {};
  const champTiles: [string, string][] = [
    ["Log-loss", dec(metrics.log_loss)],
    ["Brier", dec(metrics.brier)],
    ["ECE", dec(metrics.ece)],
    ["ROC-AUC", dec(metrics.auc)],
  ];
  if (metrics.gap_closed !== undefined && metrics.gap_closed !== null)
    champTiles.push(["Gap closed to StatsBomb", pct(metrics.gap_closed)]);
  else if (metrics.statsbomb_log_loss !== undefined && metrics.statsbomb_log_loss !== null)
    champTiles.push(["StatsBomb log-loss", dec(metrics.statsbomb_log_loss)]);

  return (
    <main className={busy && data ? "stale" : undefined}>
      <header>
        <div>
          <h1>Pipeline</h1>
          <p className="lede">
            The operational view of PitchQuery: ingest, the warehouse layers, the
            champion model, feature drift and the query log.
          </p>
        </div>
        <div className="header-side">
          <nav className="modes">
            <Link href="/">Search</Link>
            <button type="button" className="on" aria-current="page" disabled>
              Pipeline
            </button>
          </nav>
          <p className="small muted">
            {data ? <>Read {stamp(data.generated_at)} UTC, cached {data.cache_ttl_s}s. </> : null}
            <button type="button" className="linky" onClick={load} disabled={busy}>
              {busy ? "Reading…" : "Refresh"}
            </button>
          </p>
        </div>
      </header>

      {error && <p className="error">{error}</p>}
      {!data && !error && <p className="muted">Reading the database… the hosted API sleeps after 15 minutes idle and takes ~50 s to wake.</p>}

      {data && (
        <>
          <Panel
            title="Ingest"
            note="From ingest_watermark, which the loader advances inside the same transaction as its inserts. It is the record of what committed, not of what was attempted — a flow that failed halfway leaves this pointing at the last match that landed."
          >
            <Hint text={data.pipeline.error ? data.pipeline.hint : null} />
            <Hint text={!data.pipeline.error ? data.pipeline.hint : null} />
            {runs.length > 0 && (
              <>
                <Tiles items={[
                  ["Competitions loaded", int(runs.length)],
                  ["Rows loaded (cumulative)",
                    int(runs.reduce((a, r) => a + r.rows_loaded, 0))],
                  ["Last run", stamp(
                    runs.map((r) => r.last_run_at).filter(Boolean).sort().pop())],
                ]} />
                <table className="grid-table">
                  <thead>
                    <tr><th>Competition</th><th>Season</th><th>Last match</th>
                      <th>Last run</th><th className="num">Rows loaded</th></tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => (
                      <tr key={`${r.competition_id}:${r.season_id}`}>
                        <td className="mono">{r.competition_id}</td>
                        <td className="mono">{r.season_id}</td>
                        <td className="mono">{r.last_match_id ?? "—"}</td>
                        <td>{stamp(r.last_run_at)}</td>
                        <td className="num">{int(r.rows_loaded)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </Panel>

          <Panel
            title="Rows per layer"
            note="Bronze is written by Python, silver and gold by dbt. A gold table that has not kept up with bronze means dbt build has not run since the last ingest."
          >
            <div className="layer-grid">
              {(["bronze", "silver", "gold"] as const).map((layer) => (
                <div className="layer" key={layer}>
                  <dl className="tiles one">
                    <div>
                      <dt>{layer} rows</dt>
                      <dd>{int(layerTotal(layer))}</dd>
                    </div>
                  </dl>
                  <table className="grid-table">
                    <tbody>
                      {layers.filter((t) => t.layer === layer).map((t) => (
                        <tr key={t.table}>
                          <td className="mono">{t.table}</td>
                          <td className="num">
                            {t.rows === null
                              ? <span className="muted">not built</span>
                              : <>{t.estimated && <span className="muted" title="count(*) passed the 3 s ceiling; this is the planner's estimate">≈ </span>}{int(t.rows)}</>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
            <Hint text={data.layers.hint} />
          </Panel>

          <Panel
            title="Champion xG model"
            note="The champion alias only moves when a run's held-out log-loss beats the incumbent. That rule is code in models/tracking.py, not a habit."
          >
            <Tiles items={champTiles} />
            <p className="small muted">
              {champ?.source === "mlflow" ? (
                <>
                  MLflow registry, version <strong>v{champ.version}</strong>
                  {champ.commit && <> at commit <code>{sha(champ.commit)}</code></>}
                  {champ.params?.n_train_shots && <> · {String(champ.params.n_train_shots)} training shots</>}
                  {champ.params?.test_comps && <> · held out {String(champ.params.test_comps)}</>}
                  {champ.params?.calibration && <> · calibration {String(champ.params.calibration)}</>}
                </>
              ) : (
                <>
                  {champ?.params?.n_shots && <>{String(champ.params.n_shots)} held-out shots</>}
                  {champ?.params?.test_comps && <> from {String(champ.params.test_comps)}</>}
                  {champ?.params?.measured_at && <> · measured {stamp(String(champ.params.measured_at))}</>}
                </>
              )}
            </p>
            <Hint text={champ?.note} />
          </Panel>

          <Panel
            title="Feature drift"
            note="Reported as effect size, not as a drift verdict. Over thousands of shots a statistical test calls nearly every column drifted; Cohen's d says how far apart the distributions actually are."
          >
            <Hint text={data.drift.hint} />
            {shown && (
              <>
                <div className="filters">
                  <label>
                    <span>Report</span>
                    <select value={report}
                            onChange={(e) => setReport(Number(e.target.value))}>
                      {drift.map((r, i) => (
                        <option key={r.name} value={i}>{r.name}</option>
                      ))}
                    </select>
                  </label>
                  <p className="small muted">
                    {int(shown.n_current)} shots from the {shown.current} against{" "}
                    {int(shown.n_reference)} from the {shown.reference}.
                  </p>
                </div>

                <div className="split">
                  <DivergingBars
                    positiveLabel={`higher in the ${shown.current}`}
                    negativeLabel={`higher in the ${shown.reference}`}
                    rows={[...shown.shifts]
                      .sort((a, b) => Math.abs(b.cohens_d) - Math.abs(a.cohens_d))
                      .map((s) => ({
                        label: s.feature,
                        value: s.cohens_d,
                        detail: `${s.feature}: ${s.reference_mean.toFixed(3)} → ${s.current_mean.toFixed(3)}, Cohen's d ${s.cohens_d.toFixed(3)}`,
                      }))}
                  />
                  <table className="grid-table">
                    <thead>
                      <tr><th>Feature</th><th className="num">{shown.reference}</th>
                        <th className="num">{shown.current}</th><th className="num">d</th></tr>
                    </thead>
                    <tbody>
                      {[...shown.shifts]
                        .sort((a, b) => Math.abs(b.cohens_d) - Math.abs(a.cohens_d))
                        .map((s) => (
                          <tr key={s.feature}>
                            <td className="mono">{s.feature}</td>
                            <td className="num">{s.reference_mean.toFixed(3)}</td>
                            <td className="num">{s.current_mean.toFixed(3)}</td>
                            <td className="num">{s.cohens_d >= 0 ? "+" : "−"}
                              {Math.abs(s.cohens_d).toFixed(3)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
                <p className="small muted">
                  Anything under about |0.2| is a difference you would struggle to
                  see. Reports are written by{" "}
                  <code>python monitoring/drift_report.py</code> and committed to{" "}
                  <code>docs/drift/</code>.
                </p>
              </>
            )}
          </Panel>

          <Panel
            title="Queries"
            note="From search_log and click_log. The vocabulary table is the useful artefact: it is the list of words the parser does not know, written by the people using it."
          >
            <Hint text={q?.error ? q.hint : null} kind="warn" />
            <Hint text={!q?.error ? q?.hint : null} />
            {q && !q.error && (
              <>
                <Tiles items={[
                  ["Searches", int(q.totals.searches)],
                  ["Results opened", int(q.totals.clicks)],
                  ["Opened at rank ≥ 5", int(q.totals.deep_clicks)],
                  ["Words not understood", int(q.totals.unknown_words)],
                ]} />

                {q.daily.length > 0 && (
                  <div className="split">
                    <figure className="chart-card">
                      <figcaption>Searches and results opened, per day</figcaption>
                      <TimeSeries
                        rows={q.daily as never[]}
                        series={[
                          { key: "searches", name: "Searches", color: S1 },
                          { key: "clicks", name: "Opened", color: S2 },
                        ]}
                      />
                    </figure>
                    <figure className="chart-card">
                      <figcaption>p95 search latency, milliseconds</figcaption>
                      <TimeSeries
                        rows={q.daily as never[]}
                        series={[{ key: "p95_ms", name: "p95", color: S1 }]}
                        unit=" ms"
                        empty="No latency recorded yet."
                      />
                    </figure>
                  </div>
                )}

                <div className="split">
                  <div>
                    <h3>Words the parser could not place</h3>
                    <p className="small muted">
                      A to-do list for <code>core/planner.py</code> written by
                      users rather than guessed at.
                    </p>
                    {q.unparsed.length ? (
                      <table className="grid-table">
                        <thead><tr><th>Word</th><th className="num">Searches</th></tr></thead>
                        <tbody>
                          {q.unparsed.map((u) => (
                            <tr key={u.word}>
                              <td className="mono">{u.word}</td>
                              <td className="num">{int(u.n)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : <p className="muted small">None yet.</p>}
                  </div>

                  <div>
                    <h3>Results opened at rank 5 or below</h3>
                    <p className="small muted">
                      Rankings that were wrong — the answer was there and the
                      ranker put it where nobody looks.
                    </p>
                    {q.deep_clicks.length ? (
                      <table className="grid-table">
                        <thead>
                          <tr><th>Query</th><th className="num">Rank</th><th>Ranker</th></tr>
                        </thead>
                        <tbody>
                          {q.deep_clicks.map((c, i) => (
                            <tr key={`${c.possession_uid}-${i}`}>
                              <td title={c.possession_uid}>{c.query_text ?? "—"}</td>
                              <td className="num">{c.rank}</td>
                              <td className="mono">{c.ranker ?? "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p className="muted small">
                        None yet. <code>python -m pipeline.telemetry --write</code>{" "}
                        collects these into <code>eval/candidates.json</code> for
                        hand-grading. Nothing reaches the eval set automatically.
                      </p>
                    )}
                  </div>
                </div>
              </>
            )}
          </Panel>
        </>
      )}

      <footer className="muted small">
        Orchestration (Prefect), the warehouse (dbt), tracking (MLflow),
        monitoring (Prometheus/Grafana) and the match replay (Redpanda) run
        locally via Docker profiles. Only the API and this app are hosted,
        because the free tier gives 512 MB and the API already uses 252 MB.
        {" "}Data source:{" "}
        <a href="https://github.com/statsbomb/open-data">StatsBomb Open Data</a>.
      </footer>
    </main>
  );
}
