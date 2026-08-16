"use client";

import { useCallback, useEffect, useState } from "react";
import Pitch from "../components/Pitch";
import QueryBar from "../components/QueryBar";
import ResultCard from "../components/ResultCard";
import ScoutingNote from "../components/ScoutingNote";
import ShapePicker from "../components/ShapePicker";
import ShotPanel from "../components/ShotPanel";
import {
  type Filters, type Meta, type PossessionDetail, type SearchResponse,
  ask, byShape, meta as fetchMeta, possession, search, similar,
} from "../lib/api";

export default function Page() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [detail, setDetail] = useState<PossessionDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"words" | "shape">("words");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMeta().then(setMeta).catch(() => setError(
      "Cannot reach the API. Start it with:  uvicorn api.main:app --port 8000"));
  }, []);

  const select = useCallback(async (uid: string) => {
    setDetail(null);
    try {
      setDetail(await possession(uid));
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const run = useCallback(async (hint: string, filters: Filters) => {
    setBusy(true);
    setError(null);
    try {
      const r = await search(hint, filters, 20);
      setRes(r);
      if (r.results.length) select(r.results[0].possession_uid);
      else setDetail(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [select]);

  const askEnglish = useCallback(async (question: string) => {
    setBusy(true);
    setError(null);
    try {
      const r = await ask(question, {}, 20);
      setRes(r);
      if (r.results.length) select(r.results[0].possession_uid);
      else setDetail(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [select]);

  const drawShape = useCallback(async (zones: string[]) => {
    setBusy(true);
    setError(null);
    try {
      const r = await byShape(zones, {}, 20);
      setRes(r);
      if (r.results.length) select(r.results[0].possession_uid);
      else setDetail(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [select]);

  const more = useCallback(async (uid: string) => {
    setBusy(true);
    try {
      const r = await similar(uid, 20);
      setRes(r);
      if (r.results.length) select(r.results[0].possession_uid);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [select]);

  return (
    <main>
      <header>
        <div>
          <h1>PitchQuery</h1>
          <p className="lede">
            Tactical possession search over open football data.  (Initial load may take 20–30 seconds)
          </p>
        </div>
        <dl className="stats">
          <div><dt>Possessions</dt>
            <dd>{meta ? meta.possessions.toLocaleString() : "—"}</dd></div>
          <div><dt>Matches</dt>
            <dd>{meta ? meta.matches.toLocaleString() : "—"}</dd></div>
          <div><dt>Teams</dt><dd>{meta ? meta.teams.length : "—"}</dd></div>
        </dl>
      </header>

      <div className="modes">
        <button className={mode === "words" ? "on" : ""} onClick={() => setMode("words")}>
          Describe it
        </button>
        <button className={mode === "shape" ? "on" : ""} onClick={() => setMode("shape")}>
          Draw it
        </button>
      </div>

      {mode === "words" ? (
        <QueryBar meta={meta} onSearch={run} onAsk={askEnglish} busy={busy}
                  lastFilters={res?.filters} tookMs={res?.took_ms}
                  nCandidates={res?.n_candidates} plan={res?.plan} />
      ) : (
        <section className="querybar">
          <ShapePicker onSearch={drawShape} busy={busy} />
          {res && mode === "shape" && (
            <div className="readout small muted">
              {res.n_candidates.toLocaleString()} possessions have this shape,
              ranked in {res.took_ms.toFixed(0)} ms
            </div>
          )}
        </section>
      )}

      {error && <p className="error">{error}</p>}

      <div className="layout">
        <section className="results">
          {res?.note?.length ? (
            <ScoutingNote note={res.note} onSelect={select}
                          selected={detail?.summary.possession_uid} />
          ) : null}
          <h2>
            Results{" "}
            {res && <span className="muted small">({res.results.length})</span>}
          </h2>
          <div className="result-list">
            {res?.results.map((row) => (
              <ResultCard
                key={row.possession_uid}
                row={row}
                selected={detail?.summary.possession_uid === row.possession_uid}
                onSelect={() => select(row.possession_uid)}
                onSimilar={() => more(row.possession_uid)}
              />
            ))}
            {res && res.results.length === 0 && (
              <p className="muted">Nothing matched those filters.</p>
            )}
          </div>
        </section>

        <section className="viewer">
          <h2>Clip</h2>
          {detail ? (
            <>
              <div className="clip-head">
                <strong>{detail.summary.team}</strong>{" "}
                <span className="muted">v {detail.summary.opponent}</span>
                <div className="muted small">
                  {detail.summary.competition} {detail.summary.season} ·{" "}
                  {detail.summary.play_pattern} ·{" "}
                  <code>{detail.summary.possession_uid}</code>
                </div>
              </div>
              <Pitch events={detail.events} freezeFrame={detail.freeze_frame}
                     endedInGoal={detail.summary.ended_in_goal} />
              <code className="tokens block">{detail.summary.token_string}</code>
              {/* Optional chaining is load-bearing: an API deployed a few
                  minutes behind this page does not send `shots` yet. */}
              {detail.shots?.length ? <ShotPanel shots={detail.shots} /> : null}
            </>
          ) : (
            <p className="muted">Select a result to watch it.</p>
          )}
        </section>
      </div>

      <footer className="muted small">
        Data source: <a href="https://github.com/statsbomb/open-data">StatsBomb Open Data</a>.
        Used under the StatsBomb open data licence.
      </footer>
    </main>
  );
}
