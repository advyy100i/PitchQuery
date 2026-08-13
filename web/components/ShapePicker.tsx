"use client";

import { useState } from "react";

/**
 * Draw a possession by clicking zones on the pitch.
 *
 * This is retrieval with no text and no model at all: the query is the path
 * you click, matched against the `zone_path` every possession already carries.
 * The number of zones you draw sets the length of move you get back — three
 * zones asks for a direct transition, eight asks for a long build-up.
 */

const BANDS = ["D", "M", "F"] as const;
const CHANNELS = ["L", "LI", "C", "RI", "R"] as const;

const W = 120;
const H = 80;
const BAND_W = W / 3;
const CH_H = H / 5;

// Centre point of a zone, for drawing the path.
function centre(zone: string): { x: number; y: number } {
  const [b, c] = zone.split("-");
  const bi = BANDS.indexOf(b as (typeof BANDS)[number]);
  const ci = CHANNELS.indexOf(c as (typeof CHANNELS)[number]);
  return { x: bi * BAND_W + BAND_W / 2, y: ci * CH_H + CH_H / 2 };
}

type Props = {
  onSearch: (zones: string[]) => void;
  busy: boolean;
};

export default function ShapePicker({ onSearch, busy }: Props) {
  const [path, setPath] = useState<string[]>([]);

  const add = (zone: string) =>
    // Clicking the same zone twice in a row is a no-op: dwelling in a zone and
    // passing through it are the same journey, which is how the matcher reads
    // it too (consecutive repeats are collapsed server-side).
    setPath((p) => (p[p.length - 1] === zone ? p : [...p, zone]));

  const points = path.map(centre);

  return (
    <div className="shape">
      <svg viewBox={`-2 -2 ${W + 4} ${H + 4}`} className="pitch shape-pitch"
           role="group" aria-label="Click zones to draw a possession shape">
        <rect x={0} y={0} width={W} height={H} className="turf" />
        {/* Mown stripes. Purely atmospheric, but it is the difference between
            a diagram of a pitch and something that reads as one. */}
        {[1, 3, 5].map((i) => (
          <rect key={i} x={i * 20} y={0} width={20} height={H} className="mow" />
        ))}

        {/* clickable zones */}
        {BANDS.map((b, bi) =>
          CHANNELS.map((c, ci) => {
            const zone = `${b}-${c}`;
            const n = path.filter((z) => z === zone).length;
            return (
              <g key={zone}>
                <rect
                  x={bi * BAND_W} y={ci * CH_H} width={BAND_W} height={CH_H}
                  className={`zone-cell ${n ? "picked" : ""}`}
                  onClick={() => add(zone)}
                  role="button" tabIndex={0}
                  aria-label={`zone ${zone}`}
                  onKeyDown={(e) => { if (e.key === "Enter") add(zone); }}
                />
                <text x={bi * BAND_W + BAND_W / 2} y={ci * CH_H + CH_H / 2 + 1.6}
                      className="zone-label" pointerEvents="none">
                  {zone}
                </text>
              </g>
            );
          })
        )}

        {/* pitch markings, drawn over the cells but never intercepting clicks */}
        <g className="lines" pointerEvents="none">
          <rect x={0} y={0} width={W} height={H} />
          <line x1={60} y1={0} x2={60} y2={H} />
          <circle cx={60} cy={40} r={10} />
          <rect x={0} y={18} width={18} height={44} />
          <rect x={102} y={18} width={18} height={44} />
          <rect x={-2} y={36} width={2} height={8} className="goal" />
          <rect x={120} y={36} width={2} height={8} className="goal" />
        </g>

        {/* the drawn path */}
        {points.length > 1 && (
          <polyline points={points.map((p) => `${p.x},${p.y}`).join(" ")}
                    className="shape-line" pointerEvents="none" />
        )}
        {points.map((p, i) => (
          <g key={i} pointerEvents="none">
            <circle cx={p.x} cy={p.y} r={3.2} className="shape-dot" />
            <text x={p.x} y={p.y + 1.2} className="shape-dot-label">{i + 1}</text>
          </g>
        ))}

        <g className="arrow" pointerEvents="none">
          <line x1={52} y1={-1} x2={68} y2={-1} />
          <polygon points="68,-1 65,-2.2 65,0.2" />
        </g>
      </svg>

      <div className="controls">
        <button className="go" disabled={busy || path.length === 0}
                onClick={() => onSearch(path)}>
          {busy ? "Searching…" : `Find this shape${path.length ? ` (${path.length})` : ""}`}
        </button>
        <button disabled={!path.length} onClick={() => setPath((p) => p.slice(0, -1))}>
          Undo
        </button>
        <button disabled={!path.length} onClick={() => setPath([])}>Clear</button>
        <code className="muted small">{path.join(" → ") || "click zones to draw a move"}</code>
      </div>
    </div>
  );
}
