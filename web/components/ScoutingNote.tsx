"use client";

import type { NoteClaim } from "../lib/api";

/**
 * The scouting note. Every sentence is followed by the clips it was computed
 * from — click one to watch it.
 *
 * The citations are not annotations added to prose after the fact. Each
 * sentence was generated *from* exactly those possessions (see core/notes.py),
 * so a claim and its evidence cannot disagree. There is nothing to fact-check.
 */

type Props = {
  note: NoteClaim[];
  onSelect: (uid: string) => void;
  selected?: string;
};

export default function ScoutingNote({ note, onSelect, selected }: Props) {
  if (!note.length) return null;

  return (
    <section className="note">
      <h2>Scouting note</h2>
      <ul>
        {note.map((c, i) => (
          <li key={i}>
            <span>{c.text}</span>{" "}
            <span className="cites">
              {c.uids.slice(0, 4).map((uid) => (
                <button key={uid}
                        className={`cite ${selected === uid ? "on" : ""}`}
                        title={`watch ${uid}`}
                        onClick={() => onSelect(uid)}>
                  {uid}
                </button>
              ))}
              {c.uids.length > 4 && (
                <span className="muted small"> +{c.uids.length - 4}</span>
              )}
            </span>
          </li>
        ))}
      </ul>
      <p className="muted small">
        Each sentence is computed from the clips it cites, so the citations
        cannot be wrong. Generated without a language model.
      </p>
    </section>
  );
}
