"use client";

import type { ShotXG } from "../lib/api";

/**
 * This project's xG model on the shot the possession ends in.
 *
 * One bar, ours. StatsBomb's value appears only as a difference in prose,
 * because a second bar invites the reader to declare a winner from one shot,
 * and one shot cannot settle that — a 0.15 chance that goes in is not evidence
 * against 0.15. Stated as a gap it reads as what it is: how far this model sits
 * from the reference on this particular chance. The verdict over thousands of
 * shots is in docs/benchmark.md.
 *
 * The bar runs 0 to 1, because that is what a probability is. Rescaling it to
 * make small chances look bigger would be flattering the number rather than
 * reporting it.
 *
 * It will not invent a value: penalties are excluded from training, so the
 * model declines and the panel says why rather than showing a blank.
 */

type Props = { shot: ShotXG };

/** How far this model sits from StatsBomb on this shot, in words. */
function difference(mine: number, theirs: number): string {
  const gap = mine - theirs;
  if (Math.abs(gap) < 0.005) {
    return `Within a hundredth of StatsBomb's ${theirs.toFixed(2)}.`;
  }
  return `${Math.abs(gap).toFixed(2)} ${gap > 0 ? "higher" : "lower"} than ` +
         `StatsBomb's ${theirs.toFixed(2)}.`;
}

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
      </div>

      {mine != null ? (
        <>
          <div className="xg-single">
            <span className="xg-big">{mine.toFixed(2)}</span>
            <span className="xg-track">
              <span className="xg-fill" style={{ width: `${Math.max(1.5, mine * 100)}%` }} />
            </span>
          </div>
          {shot.statsbomb_xg != null && (
            <p className="xg-diff small">{difference(mine, shot.statsbomb_xg)}</p>
          )}
        </>
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
