"use client";

import type { ShotXG } from "../lib/api";

/**
 * The two xG models on the same shot, with the context that separates them.
 *
 * Three things this panel is careful about:
 *
 * 1. The bars share one scale. Showing each against its own maximum would make
 *    a 0.04 chance and a 0.4 chance look alike, and the whole point is the gap
 *    between the two numbers.
 *
 * 2. It says whether the shot was held out of training. Most shots a visitor
 *    clicks come from competitions the model learned from, where agreeing with
 *    StatsBomb proves nothing. Presenting those identically to genuinely
 *    out-of-sample shots would flatter the model, so the badge distinguishes
 *    them.
 *
 * 3. It never invents a number. Penalties are excluded from training, so the
 *    model declines and the panel says why rather than showing a blank.
 *
 * Deliberately absent: any claim about which model is "right" on a single shot.
 * One shot cannot settle that — a 0.15 chance that goes in is not evidence
 * against 0.15. The aggregate answer lives in docs/benchmark.md.
 */

type Props = { shot: ShotXG };

function Bar({ label, value, scale, tone }:
             { label: string; value: number; scale: number; tone: string }) {
  return (
    <div className="xg-row">
      <span className="xg-label">{label}</span>
      <span className="xg-track">
        <span className={`xg-fill ${tone}`}
              style={{ width: `${Math.max(1.5, (value / scale) * 100)}%` }} />
      </span>
      <span className="xg-value">{value.toFixed(3)}</span>
    </div>
  );
}

export default function ShotPanel({ shot }: Props) {
  const mine = shot.my_xg;
  const sb = shot.statsbomb_xg;

  // A shared ceiling, with a floor so ordinary chances are not hairlines.
  const scale = Math.max(0.3, mine ?? 0, sb ?? 0);

  const context: string[] = [];
  if (shot.distance != null) context.push(`${shot.distance.toFixed(1)} m out`);
  if (shot.body_part) context.push(shot.body_part.toLowerCase());
  if (shot.n_def_in_cone != null) {
    context.push(shot.n_def_in_cone === 0
      ? "no defender in the shooting cone"
      : `${shot.n_def_in_cone} defender${shot.n_def_in_cone > 1 ? "s" : ""} in the cone`);
  }
  if (shot.gk_off_line != null) {
    context.push(`keeper ${shot.gk_off_line.toFixed(1)} m off the line`);
  }

  return (
    <section className="xg-panel">
      <div className="xg-head">
        <h3>Expected goals</h3>
        {shot.in_holdout === true && (
          <span className="badge holdout" title="This competition was excluded from training entirely">
            held out of training
          </span>
        )}
        {shot.in_holdout === false && (
          <span className="badge muted-badge" title="The model trained on this competition, so agreement here is not evidence">
            in training data
          </span>
        )}
      </div>

      {mine != null ? (
        <Bar label="This model" value={mine} scale={scale} tone="mine" />
      ) : (
        <div className="xg-row">
          <span className="xg-label">This model</span>
          <span className="xg-declined muted small">{shot.my_xg_note ?? "no prediction"}</span>
        </div>
      )}

      {sb != null && <Bar label="StatsBomb" value={sb} scale={scale} tone="theirs" />}

      <p className="xg-context muted small">
        {shot.player && <><strong>{shot.player}</strong>{shot.minute != null && `, ${shot.minute}'`}. </>}
        {shot.is_goal ? "Scored. " : "Did not score. "}
        {context.length > 0 && `${context.join(", ")}.`}
      </p>

      <p className="muted small">
        LightGBM on shot geometry plus freeze-frame context, calibrated and held
        out by whole tournaments. One shot settles nothing either way — the
        aggregate comparison is in the benchmark.
      </p>
    </section>
  );
}
