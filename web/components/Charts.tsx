"use client";

/**
 * Two chart forms, hand-rolled, for the operational page.
 *
 * No charting library. The pitch in Pitch.tsx is already hand-drawn SVG against
 * the same tokens, and a dependency that ships its own palette, its own type
 * scale and its own tooltip would be the one thing on the page that does not
 * look like the page.
 *
 * The series colours are NOT --accent and --warn. Those are UI tokens: the
 * accent is a deliberately low-chroma teal (OKLCH C 0.063) chosen to sit under
 * text without shouting, and a 2px line drawn in it reads as grey beside a
 * second series. --s1/--s2 are the same two hues stepped up to the chroma a
 * data mark needs, and they are checked rather than eyeballed — deuteranope and
 * protanope separation ΔE 16.0, normal-vision 25.9, both over 3:1 against the
 * panel in light and dark. See the block that defines them in globals.css.
 *
 * Identity is never carried by colour alone: every series has a legend entry
 * and a direct label at the end of its line, and every chart has a table under
 * it with the same numbers in it.
 */

import { useEffect, useRef, useState } from "react";

/** Container width in CSS pixels, so SVG text is sized in pixels and not scaled
 *  by a viewBox. A viewBox that stretches to fit also stretches its labels, and
 *  the same chart then has 8px axis text on a phone and 15px on a monitor. */
function useWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(el);
    setW(el.getBoundingClientRect().width);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/**
 * The next round number above the data, so ticks land on numbers people read.
 *
 * Two ladders. An axis that is halved for its middle tick can only use 1/2/5,
 * or the tick reads 1.25. A diverging bar chart has no middle tick and a
 * three-rung ladder wastes half its width on any value just over a rung —
 * Cohen's d of 0.21 against a 0.5 domain is a chart of stubs.
 */
function niceCeil(v: number, steps: number[] = [1, 2, 5, 10]): number {
  if (!isFinite(v) || v <= 0) return steps[0];
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const r = v / mag;
  return (steps.find((s) => r <= s) ?? 10) * mag;
}

/** Rungs that stay exact at two decimals, which the axis labels print at. */
const BAR_STEPS = [1, 1.5, 2, 3, 5, 10];

const day = (iso: string) =>
  new Date(iso + "T00:00:00Z").toLocaleDateString(undefined,
    { month: "short", day: "numeric", timeZone: "UTC" });

export type Series = { key: string; name: string; color: string };
type Row = Record<string, string | number | null>;

/**
 * A time series, one line per series on ONE y-axis.
 *
 * Deliberately not able to draw two scales. Searches per day and p95 latency in
 * milliseconds are two of these side by side rather than two axes on one plot:
 * where the two scales meet is an arbitrary choice, and the reader takes the
 * crossing point for a finding.
 */
export function TimeSeries({
  rows, series, height = 180, unit = "", digits = 0, empty = "No data yet.",
}: {
  rows: Row[];
  series: Series[];
  height?: number;
  unit?: string;
  digits?: number;
  empty?: string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const fmt = (v: number) =>
    v.toLocaleString(undefined, { maximumFractionDigits: digits }) + unit;

  if (!rows.length) return <p className="muted small chart-empty">{empty}</p>;

  // Right padding holds the direct labels; without it the last point's name is
  // clipped by the SVG edge, which is the label-overflow failure exactly.
  const padL = 44, padR = series.length > 1 ? 78 : 62, padT = 12, padB = 24;
  const width = Math.max(w, 260);
  const plotW = Math.max(width - padL - padR, 10);
  const plotH = height - padT - padB;

  const at = (r: Row, k: string) => {
    const v = r[k];
    return typeof v === "number" ? v : null;
  };
  const top = niceCeil(Math.max(
    ...series.flatMap((s) => rows.map((r) => at(r, s.key) ?? 0)), 0));
  const n = rows.length;
  const sx = (i: number) => padL + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const sy = (v: number) => padT + plotH - (v / top) * plotH;

  // Segments, not one path: a day with no measurement (p95 over zero searches)
  // is a gap in the line, not a straight edge drawn across it.
  const pathFor = (s: Series) => {
    let d = "", open = false;
    rows.forEach((r, i) => {
      const v = at(r, s.key);
      if (v === null) { open = false; return; }
      d += `${open ? "L" : "M"}${sx(i).toFixed(1)} ${sy(v).toFixed(1)} `;
      open = true;
    });
    return d.trim();
  };

  const ticks = [0, top / 2, top];
  const h = hover !== null && hover < n ? hover : null;

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img"
           aria-label={`${series.map((s) => s.name).join(" and ")} by day`}
           onMouseLeave={() => setHover(null)}>
        {/* Solid hairlines, one shade off the surface. Never dashed: a dashed
            grid reads as a threshold or a projection when it is just a grid. */}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={padL + plotW} y1={sy(t)} y2={sy(t)}
                  className="grid" />
            <text x={padL - 8} y={sy(t) + 3.5} className="tick" textAnchor="end">
              {t.toLocaleString(undefined, { maximumFractionDigits: digits })}
            </text>
          </g>
        ))}

        <text x={padL} y={height - 7} className="tick">{day(String(rows[0].day))}</text>
        {n > 1 && (
          <text x={padL + plotW} y={height - 7} className="tick" textAnchor="end">
            {day(String(rows[n - 1].day))}
          </text>
        )}

        {h !== null && (
          <line x1={sx(h)} x2={sx(h)} y1={padT} y2={padT + plotH} className="crosshair" />
        )}

        {series.map((s) => {
          const last = [...rows].reverse().find((r) => at(r, s.key) !== null);
          const lastI = last ? rows.indexOf(last) : -1;
          return (
            <g key={s.key}>
              <path d={pathFor(s)} fill="none" stroke={s.color} strokeWidth={2}
                    strokeLinecap="round" strokeLinejoin="round" />
              {/* One marker per point only when they are far enough apart to be
                  hit; otherwise the endpoint carries the direct label and the
                  crosshair carries the rest. */}
              {n <= 12 && rows.map((r, i) => {
                const v = at(r, s.key);
                return v === null ? null : (
                  <circle key={i} cx={sx(i)} cy={sy(v)} r={4} fill={s.color}
                          stroke="var(--panel)" strokeWidth={2} />
                );
              })}
              {lastI >= 0 && (
                <>
                  <circle cx={sx(lastI)} cy={sy(at(last!, s.key)!)} r={4}
                          fill={s.color} stroke="var(--panel)" strokeWidth={2} />
                  <text x={sx(lastI) + 9} y={sy(at(last!, s.key)!) + 3.5}
                        className="direct">
                    {fmt(at(last!, s.key)!)}
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* Full-height bands: the hit target is the column, not the 8px dot. */}
        {rows.map((_, i) => (
          <rect key={i} x={padL + (i - 0.5) * (plotW / Math.max(n - 1, 1))}
                y={padT} width={plotW / Math.max(n - 1, 1)} height={plotH}
                fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
      </svg>

      {h !== null && (
        <div className="tip" style={{
          left: Math.min(Math.max(sx(h) - 60, 0), Math.max(width - 132, 0)),
        }}>
          <strong>{day(String(rows[h].day))}</strong>
          {series.map((s) => (
            <span key={s.key}>
              <i className="swatch" style={{ background: s.color }} />
              {s.name}
              <b>{at(rows[h], s.key) === null ? "—" : fmt(at(rows[h], s.key)!)}</b>
            </span>
          ))}
        </div>
      )}

      {series.length > 1 && (
        <ul className="legend">
          {series.map((s) => (
            <li key={s.key}>
              <i className="swatch" style={{ background: s.color }} />{s.name}
            </li>
          ))}
        </ul>
      )}

      <details className="table-twin">
        <summary className="small muted">Table</summary>
        <table className="grid-table">
          <thead>
            <tr><th>Day</th>{series.map((s) => <th key={s.key}>{s.name}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{day(String(r.day))}</td>
                {series.map((s) => (
                  <td key={s.key} className="num">
                    {at(r, s.key) === null ? "—" : fmt(at(r, s.key)!)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

/**
 * A diverging horizontal bar chart: two hues for the two directions, a neutral
 * rule at zero.
 *
 * Built from HTML boxes rather than SVG because half of it is text — feature
 * names and signed values — and text in a scaled viewBox is text at the wrong
 * size. Values sit outside the bar end always, so a bar too short to hold its
 * own label is not a bar with a clipped label.
 */
export function DivergingBars({
  rows, domain, digits = 2, positiveLabel, negativeLabel,
}: {
  rows: { label: string; value: number; detail?: string }[];
  domain?: number;
  digits?: number;
  positiveLabel: string;
  negativeLabel: string;
}) {
  if (!rows.length) return null;
  const max = domain ?? niceCeil(
    Math.max(...rows.map((r) => Math.abs(r.value))), BAR_STEPS);
  const pct = (v: number) => (Math.abs(v) / max) * 50;

  return (
    <div className="dbars">
      <ul className="legend">
        <li><i className="swatch" style={{ background: "var(--s2)" }} />{positiveLabel}</li>
        <li><i className="swatch" style={{ background: "var(--s1)" }} />{negativeLabel}</li>
      </ul>
      {rows.map((r) => (
        <div className="dbar-row" key={r.label} title={r.detail ?? r.label}>
          <span className="dbar-label mono">{r.label}</span>
          <span className="dbar-track">
            <span className="dbar-zero" />
            <span
              className={`dbar-fill ${r.value >= 0 ? "pos" : "neg"}`}
              style={r.value >= 0
                ? { left: "50%", width: `${pct(r.value)}%` }
                : { right: "50%", width: `${pct(r.value)}%` }}
            />
          </span>
          <span className="dbar-value tnum">
            {r.value >= 0 ? "+" : "−"}{Math.abs(r.value).toFixed(digits)}
          </span>
        </div>
      ))}
      <div className="dbar-axis small muted">
        <span />
        <span className="dbar-scale">
          <span>−{max.toFixed(digits)}</span>
          <span>0</span>
          <span>+{max.toFixed(digits)}</span>
        </span>
        <span />
      </div>
    </div>
  );
}
