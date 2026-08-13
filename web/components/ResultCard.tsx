"use client";

import type { PossessionSummary } from "../lib/api";

/**
 * A result row. The zone-path sparkline is the point: it lets you scan the
 * shape of twenty possessions without playing twenty clips.
 */

const BANDS: Record<string, number> = { D: 0, M: 1, F: 2 };
const CHANNELS: Record<string, number> = { L: 0, LI: 1, C: 2, RI: 3, R: 4 };

function Sparkline({ zonePath }: { zonePath: string }) {
  const zones = zonePath.split(" ").filter(Boolean);
  if (zones.length < 2) return null;

  const w = 132;
  const h = 34;
  const pts = zones.map((z, i) => {
    const [b, c] = z.split("-");
    const x = (i / (zones.length - 1)) * (w - 6) + 3;
    // y is the channel (left at top), x-position within the sparkline is time.
    const y = ((CHANNELS[c] ?? 2) / 4) * (h - 8) + 4;
    return { x, y, band: BANDS[b] ?? 1 };
  });

  return (
    <svg width={w} height={h} className="spark" aria-label={`zone path ${zonePath}`}>
      {/* band shading: darker = closer to the opponent goal */}
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={p.band === 2 ? 2.6 : 1.8}
                className={`spark-dot band-${p.band}`} />
      ))}
      <polyline
        points={pts.map((p) => `${p.x},${p.y}`).join(" ")}
        className="spark-line"
      />
    </svg>
  );
}

type Props = {
  row: PossessionSummary;
  selected: boolean;
  onSelect: () => void;
  onSimilar: () => void;
};

export default function ResultCard({ row, selected, onSelect, onSimilar }: Props) {
  return (
    <div className={`card ${selected ? "selected" : ""}`} onClick={onSelect}
         role="button" tabIndex={0}
         onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onSelect(); }}>
      <div className="card-head">
        <strong>{row.team}</strong>
        <span className="muted"> v {row.opponent}</span>
        {row.ended_in_goal && <span className="badge goal">GOAL</span>}
        {!row.ended_in_goal && row.ended_in_shot && <span className="badge shot">shot</span>}
        {row.xg_sum > 0 && <span className="badge xg">xG {row.xg_sum.toFixed(2)}</span>}
      </div>

      <div className="card-meta muted small">
        {row.competition} {row.season} · {row.play_pattern} · {row.n_events} tokens ·{" "}
        {row.duration_s.toFixed(0)}s
      </div>

      <Sparkline zonePath={row.zone_path} />

      <code className="tokens">{row.token_string}</code>

      <div className="card-actions">
        <button onClick={(e) => { e.stopPropagation(); onSimilar(); }}>More like this</button>
        <span className="uid muted small">{row.possession_uid}</span>
      </div>
    </div>
  );
}
