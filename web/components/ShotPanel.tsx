"use client";

import type { ShotXG } from "../lib/api";

/**
 * This project's xG model on the shots in a possession.
 *
 * The headline number is the LAST shot — the one the animation stops on, and
 * the one that decides whether the move ended in a goal. About one shot-ending
 * possession in ten contains a rebound or a second attempt, and there the
 * possession total on the result card is nobody's chance in particular. So the
 * final attempt gets named and the earlier ones are listed under it, with the
 * total shown so the two views reconcile instead of quietly disagreeing.
 *
 * StatsBomb's value appears only as a difference in prose. A second bar invites
 * the reader to declare a winner from one shot, and one shot cannot settle
 * that — a 0.15 chance that goes in is not evidence against 0.15. Stated as a
 * gap it reads as what it is. The verdict over thousands of shots is in
 * docs/benchmark.md.
 *
 * The bar runs 0 to 1, because that is what a probability is. Rescaling it to
 * make small chances look bigger would be flattering the number rather than
 * reporting it.
 *
 * It will not invent a value: penalties are excluded from training, so the
 * model declines and the panel says why rather than showing a blank.
 */

type Props = { shots: ShotXG[] };

/** How far this model sits from StatsBomb on this shot, in words. */
function difference(mine: number, theirs: number): string {
  const gap = mine - theirs;
  if (Math.abs(gap) < 0.005) {
    return `Within a hundredth of StatsBomb's ${theirs.toFixed(2)}.`;
  }
  return `${Math.abs(gap).toFixed(2)} ${gap > 0 ? "higher" : "lower"} than ` +
         `StatsBomb's ${theirs.toFixed(2)}.`;
}

function describe(shot: ShotXG): string {
  const bits: string[] = [];
  if (shot.distance != null) bits.push(`${shot.distance.toFixed(1)} m out`);
  if (shot.body_part) bits.push(shot.body_part.toLowerCase());
  if (shot.n_def_in_cone != null) {
    bits.push(shot.n_def_in_cone === 0
      ? "no defender in the shooting cone"
      : `${shot.n_def_in_cone} defender${shot.n_def_in_cone > 1 ? "s" : ""} in the cone`);
  }
  if (shot.gk_off_line != null) {
    bits.push(`keeper ${shot.gk_off_line.toFixed(1)} m off the line`);
  }
  return bits.join(", ");
}

export default function ShotPanel({ shots }: Props) {
  if (!shots.length) return null;

  const last = shots[shots.length - 1];
  const earlier = shots.slice(0, -1);
  const mine = last.my_xg;

  // Only the shots the model actually scored. Summing a declined penalty as
  // zero would understate the move rather than describe it.
  const scored = shots.filter((s) => s.my_xg != null);
  const total = scored.reduce((sum, s) => sum + (s.my_xg as number), 0);

  return (
    <section className="xg-panel">
      <div className="xg-head">
        <h3>Expected goals</h3>
        <span className="xg-which muted small">
          {last.is_goal ? "the goal" : "final shot"}
        </span>
      </div>

      {mine != null ? (
        <>
          <div className="xg-single">
            <span className="xg-big">{mine.toFixed(2)}</span>
            <span className="xg-track">
              <span className="xg-fill" style={{ width: `${Math.max(1.5, mine * 100)}%` }} />
            </span>
          </div>
          {last.statsbomb_xg != null && (
            <p className="xg-diff small">{difference(mine, last.statsbomb_xg)}</p>
          )}
        </>
      ) : (
        <p className="xg-declined muted small">{last.my_xg_note ?? "no prediction"}</p>
      )}

      <p className="xg-context muted small">
        {last.player && (
          <><strong>{last.player}</strong>{last.minute != null && `, ${last.minute}'`}. </>
        )}
        {last.is_goal ? "Scored. " : "Did not score. "}
        {describe(last) && `${describe(last)}.`}
      </p>

      {earlier.length > 0 && (
        <div className="xg-earlier">
          <h4 className="muted small">
            {earlier.length === 1 ? "Earlier attempt" : "Earlier attempts"}
          </h4>
          <ul>
            {earlier.map((s) => (
              <li key={s.event_id}>
                <span className="xg-when muted">{s.minute != null ? `${s.minute}'` : "—"}</span>
                <span className="xg-who">{s.player ?? "unknown"}</span>
                {/* A dash with no explanation reads as missing data. The
                    model declined, and hovering says why. */}
                <span className="xg-each" title={s.my_xg == null ? s.my_xg_note ?? "" : ""}>
                  {s.my_xg != null ? s.my_xg.toFixed(2) : "—"}
                </span>
              </li>
            ))}
          </ul>
          {scored.length > 1 && (
            <p className="xg-total small">
              {scored.length} shots in the move, {total.toFixed(2)} xG in total.
            </p>
          )}
        </div>
      )}

      <p className="muted small">
        LightGBM on shot geometry plus freeze-frame context — defenders in the
        shooting cone, where the keeper is — calibrated, and held out by whole
        tournaments rather than by shot.
      </p>
    </section>
  );
}
