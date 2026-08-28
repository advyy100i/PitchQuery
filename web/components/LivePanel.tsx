"use client";

import { useEffect, useRef, useState } from "react";
import { type LiveMessage, liveUrl } from "../lib/api";

/**
 * The match replay, building one token at a time.
 *
 * Labelled a replay everywhere it appears, in the heading and in the status
 * line, because that is what it is: StatsBomb publish open data long after the
 * match, `stream/producer.py` reads a cached file and sleeps the real gaps, and
 * the consumer rebuilds possessions from it incrementally. Every message from
 * the API carries `source: "replay"` and this component renders that field
 * rather than a hard-coded word, so the label cannot drift from the truth.
 *
 * Renders nothing at all when no producer is running. A panel that says
 * "waiting for events" on a page nobody has started a replay for is noise; the
 * feature is off until there is something to show.
 */
export default function LivePanel() {
  const [current, setCurrent] = useState<LiveMessage | null>(null);
  const [closed, setClosed] = useState<LiveMessage[]>([]);
  const [state, setState] = useState<"idle" | "open" | "gone">("idle");
  const socket = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket;
    try {
      ws = new WebSocket(liveUrl());
    } catch {
      // No API, or a browser refusing the ws:// upgrade. Nothing to report:
      // the replay is a development feature and the page works without it.
      return;
    }
    socket.current = ws;

    ws.onopen = () => !cancelled && setState("open");
    ws.onclose = () => !cancelled && setState("gone");
    ws.onerror = () => !cancelled && setState("gone");
    ws.onmessage = (e) => {
      if (cancelled) return;
      const msg: LiveMessage = JSON.parse(e.data);
      if (msg.type === "token") setCurrent(msg);
      else if (msg.type === "possession_closed") {
        setCurrent(null);
        // Only the possessions the batch pipeline would have kept. Showing the
        // two-token throw-in fragments it drops would make the panel disagree
        // with the index it is meant to illustrate.
        if (msg.kept) setClosed((prev) => [msg, ...prev].slice(0, 6));
      } else if (msg.type === "replay_end") {
        setCurrent(null);
        setState("gone");
      }
    };

    return () => {
      cancelled = true;
      ws.close();
    };
  }, []);

  if (state === "idle" || (state === "gone" && !closed.length)) return null;

  const source = current?.source ?? closed[0]?.source ?? "replay";

  return (
    <section className="live">
      <h2>
        Match {source}{" "}
        <span className="muted small">
          {state === "open" ? "· receiving" : "· finished"}
        </span>
      </h2>
      <p className="muted small">
        Recorded events replayed through Kafka and regrouped into possessions as
        they arrive — not a live feed. xG is scored the moment a shot lands.
      </p>

      {current ? (
        <div className="live-current">
          <div className="live-head">
            <strong>{current.team}</strong>{" "}
            <span className="muted small">
              {current.minute}&rsquo;{String(current.second ?? 0).padStart(2, "0")} ·{" "}
              {current.n_tokens} tokens
              {current.my_xg_sum != null && ` · xG ${current.my_xg_sum.toFixed(3)}`}
            </span>
          </div>
          <code className="tokens block">{current.token_string}</code>
        </div>
      ) : (
        <p className="muted small">Between possessions…</p>
      )}

      {closed.length > 0 && (
        <ol className="live-closed">
          {closed.map((p, i) => (
            <li key={`${p.possession}-${i}`}>
              <span className="muted small">
                {p.team} · {p.n_tokens} tokens · {p.duration_s?.toFixed(0)}s
                {p.my_xg_sum != null && ` · xG ${p.my_xg_sum.toFixed(3)}`}
              </span>
              <code className="tokens block small">{p.token_string}</code>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
