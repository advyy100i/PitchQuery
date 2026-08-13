"use client";

import { useState } from "react";
import type { Filters, Meta, PlanResponse } from "../lib/api";

/**
 * English is the primary input. It is translated by `core/planner.py` — a rule
 * parser, not a model — and the translation is shown in full underneath.
 *
 * Showing it is the point. Every filter is labelled with the phrase that
 * produced it, and anything the parser could not place is listed as "not
 * understood" rather than silently dropped. A user can see exactly what was
 * searched for and correct it, which is not something an opaque translator
 * can offer.
 *
 * The token query is still there under "Token query" for anyone who wants to
 * write the grammar directly.
 */

const ENGLISH_EXAMPLES = [
  "right-wing cross into the box that ends in a shot",
  "fast counter-attack ending in a shot",
  "Barcelona working the ball into the left half-space",
  "playing out from the back under heavy pressing",
  "interception in midfield turned into an immediate attack",
];

const EXAMPLES: { label: string; hint: string; filters: Filters }[] = [
  {
    label: "Right-wing cross → shot",
    hint: "PASS@M-R+ RECV@F-R CARRY@F-R CROSS@F-R> RECV@F-C SHOT@F-C",
    filters: { ended_in_shot: true },
  },
  {
    label: "Counter-attack goal",
    hint: "RECOV@M-C CARRY@M-C+ PASS@M-C+ RECV@F-C CARRY@F-C SHOT@F-C",
    filters: { play_pattern: "From Counter", ended_in_shot: true },
  },
  {
    label: "Corner → shot",
    hint: "SETP@F-R> RECV@F-C SHOT@F-C",
    filters: { play_pattern: "From Corner" },
  },
  {
    label: "Switch of play",
    hint: "CARRY@M-L PASS@M-L RECV@M-L SWITCH@M-L+ RECV@M-R CARRY@M-R+",
    filters: {},
  },
];

type Props = {
  meta: Meta | null;
  onSearch: (hint: string, filters: Filters) => void;
  onAsk: (question: string) => void;
  busy: boolean;
  lastFilters?: Record<string, unknown>;
  tookMs?: number;
  nCandidates?: number;
  plan?: PlanResponse | null;
};

export default function QueryBar({ meta, onSearch, onAsk, busy, lastFilters, tookMs,
                                   nCandidates, plan }: Props) {
  const [hint, setHint] = useState(EXAMPLES[0].hint);
  const [filters, setFilters] = useState<Filters>(EXAMPLES[0].filters);
  const [question, setQuestion] = useState(ENGLISH_EXAMPLES[0]);
  const [advanced, setAdvanced] = useState(false);

  const set = (k: keyof Filters, v: unknown) =>
    setFilters((f) => ({ ...f, [k]: v === "" ? undefined : v }));

  return (
    <section className="querybar">
      <label className="field">
        <span>Describe a passage of play</span>
        <input
          className="ask"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") onAsk(question); }}
          placeholder="right-wing cross into the box that ends in a shot"
        />
      </label>

      <div className="examples">
        {ENGLISH_EXAMPLES.map((ex) => (
          <button key={ex} className="chip"
                  onClick={() => { setQuestion(ex); onAsk(ex); }}>
            {ex}
          </button>
        ))}
      </div>

      <div className="filters">
        <button className="go" onClick={() => onAsk(question)} disabled={busy}>
          {busy ? "Searching…" : "Search"}
        </button>
        <button onClick={() => setAdvanced((a) => !a)}>
          {advanced ? "Hide token query" : "Token query"}
        </button>
      </div>

      {plan && (
        <div className="readout">
          <div>
            <strong>Translated to a structured query</strong>{" "}
            <span className="muted small">
              in {plan.parse_ms.toFixed(1)} ms by rules — no model, no API key
            </span>
          </div>
          <ul className="plan-terms">
            {plan.terms.map((t, i) => (
              <li key={i}>
                <code>{t.phrase}</code> <span className="muted">→ {t.effect}</span>
              </li>
            ))}
          </ul>
          <div className="small">
            <span className="muted">matched against: </span>
            <code>{plan.sequence_hint}</code>
          </div>
          {plan.ignored && (
            <div className="small muted">
              not understood: <em>{plan.ignored}</em>
            </div>
          )}
          {tookMs !== undefined && (
            <div className="small muted">
              {nCandidates?.toLocaleString()} possessions passed the filters,
              ranked in {tookMs.toFixed(0)} ms
            </div>
          )}
        </div>
      )}

      {advanced && (
        <div className="advanced">
          <label className="field">
            <span>Token sequence</span>
            <input
              value={hint}
              onChange={(e) => setHint(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") onSearch(hint, filters); }}
              spellCheck={false}
              placeholder="CROSS@F-R> RECV@F-C SHOT@F-C"
            />
          </label>

          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button key={ex.label} className="chip"
                      onClick={() => { setHint(ex.hint); setFilters(ex.filters);
                                       onSearch(ex.hint, ex.filters); }}>
                {ex.label}
              </button>
            ))}
          </div>

          <div className="filters">
            <label>
              <span>Team</span>
              <select value={filters.team ?? ""} onChange={(e) => set("team", e.target.value)}>
                <option value="">any</option>
                {meta?.teams.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </label>
            <label>
              <span>Competition</span>
              <select value={filters.competition ?? ""}
                      onChange={(e) => set("competition", e.target.value)}>
                <option value="">any</option>
                {meta?.competitions.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label>
              <span>Play pattern</span>
              <select value={filters.play_pattern ?? ""}
                      onChange={(e) => set("play_pattern", e.target.value)}>
                <option value="">any</option>
                {meta?.play_patterns.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
            <label className="check">
              <input type="checkbox" checked={!!filters.ended_in_shot}
                     onChange={(e) => set("ended_in_shot", e.target.checked || undefined)} />
              <span>ended in a shot</span>
            </label>
            <button className="go" onClick={() => onSearch(hint, filters)} disabled={busy}>
              Run token query
            </button>
          </div>
        </div>
      )}

      {lastFilters && !plan && (
        <div className="readout">
          <span className="muted small">Ran with:</span>{" "}
          <code>{Object.keys(lastFilters).length ? JSON.stringify(lastFilters) : "no filters"}</code>
          {tookMs !== undefined && (
            <span className="muted small">
              {" "}· {nCandidates?.toLocaleString()} possessions passed the filters,
              ranked in {tookMs.toFixed(0)} ms
            </span>
          )}
        </div>
      )}
    </section>
  );
}
