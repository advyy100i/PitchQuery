"use client";

import type { ShotXG } from "../lib/api";

/**
 * This project's xG model on the shot the possession ends in.
 *
 * Only our number is shown. StatsBomb's is in the API response and drives the
 * benchmark in docs/benchmark.md, but a second figure here would turn every
 * clip into a scoreboard — and a single shot cannot settle which model is
 * better. A 0.15 chance that goes in is not evidence against 0.15. That
 * question needs thousands of shots and a log-loss, not a side-by-side.
 *
 * The bar runs 0 to 1, because that is what a probability is. Rescaling it to
 * make small chances look bigger would be flattering the number rather than
 * reporting it.
 *
 * Two things it will not do:
 *
 * - Invent a value. Penalties are excluded from training, so the model declines
 *   and the panel says why rather than showing a blank.
 *
 * - Imply a prediction is out of sample when it is not. Most shots a visitor
 *   clicks come from competitions the model learned from; the badge separates
 *   those from the two tournaments it never saw.
 */

type Props = { shot: ShotXG };

export default function ShotPanel({ shot }: Props) {
  const mine = shot.my_xg;

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
          <span className="badge holdout"
                title="This competition was excluded from training entirely">
            held out of training
          </span>
        )}
        {shot.in_holdout === false && (
          <span className="badge muted-badge"
                title="The model trained on this competition, so this is a fit, not a test">
            in training data
          </span>
        )}
      </div>

      {mine != null ? (
        <div className="xg-single">
          <span className="xg-big">{mine.toFixed(2)}</span>
          <span className="xg-track">
            <span className="xg-fill" style={{ width: `${Math.max(1.5, mine * 100)}%` }} />
          </span>
        </div>
      ) : (
        <p className="xg-declined muted small">{shot.my_xg_note ?? "no prediction"}</p>
      )}

      <p className="xg-context muted small">
        {shot.player && (
          <><strong>{shot.player}</strong>{shot.minute != null && `, ${shot.minute}'`}. </>
        )}
        {shot.is_goal ? "Scored. " : "Did not score. "}
        {context.length > 0 && `${context.join(", ")}.`}
      </p>

      <p className="muted small">
        LightGBM on shot geometry plus freeze-frame context — defenders in the
        shooting cone, where the keeper is — calibrated, and held out by whole
        tournaments rather than by shot.
      </p>
    </section>
  );
}
