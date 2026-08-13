"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { EventPoint, FreezeFramePlayer } from "../lib/api";

/**
 * SVG pitch with a ball animated along consecutive event locations.
 *
 * Coordinates are StatsBomb's 120x80 with the origin top-left and the attack
 * always running left to right, so the viewBox is the pitch and no transform is
 * needed. Defensive events arrive already mirrored into this frame by the API.
 *
 * Player dots are only drawn where real positions exist — the freeze frame at
 * the moment of the shot. Everywhere else the ball moves alone. Inventing
 * player positions would make the animation look better and mean nothing.
 */

const W = 120;
const H = 80;
const DEFAULT_STEP_MS = 400;

type Props = {
  events: EventPoint[];
  freezeFrame?: FreezeFramePlayer[];
  endedInGoal?: boolean;
};

type Step = {
  x: number;
  y: number;
  endX: number;
  endY: number;
  ms: number;
  label: string;
  attacking: boolean;
  pressure: boolean;
  isShot: boolean;
};

function buildSteps(events: EventPoint[]): Step[] {
  const located = events.filter((e) => e.x !== null && e.y !== null);
  return located.map((e, i) => {
    const next = located[i + 1];
    // A pass or carry knows where it finished; anything else moves the ball to
    // wherever the next event happens, which is what actually occurred.
    const endX = e.end_x ?? next?.x ?? e.x!;
    const endY = e.end_y ?? next?.y ?? e.y!;
    return {
      x: e.x!,
      y: e.y!,
      endX,
      endY,
      ms: Math.max(120, Math.round((e.duration ?? DEFAULT_STEP_MS / 1000) * 1000)),
      label: e.token ?? e.type,
      attacking: e.is_attacking,
      pressure: e.under_pressure,
      isShot: e.type === "Shot",
    };
  });
}

function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function Pitch({ events, freezeFrame = [], endedInGoal }: Props) {
  const steps = useMemo(() => buildSteps(events), [events]);
  const [i, setI] = useState(0);
  const [t, setT] = useState(0); // 0..1 within the current step
  const [playing, setPlaying] = useState(false);
  const raf = useRef<number | null>(null);
  const last = useRef<number>(0);
  // The rAF callback closes over `i` from the render that armed it, which can
  // already be stale by the time a frame lands — that let the index run past
  // the last step and render "5/4". The ref is the single source of truth for
  // the loop; `i` exists only to drive re-renders.
  const iRef = useRef(0);

  const reduced = useMemo(prefersReducedMotion, []);

  // Respect prefers-reduced-motion: never auto-play, and jump rather than tween.
  useEffect(() => {
    iRef.current = 0;
    setI(0);
    setT(0);
    setPlaying(!reduced && steps.length > 0);
  }, [steps, reduced]);

  useEffect(() => {
    if (!playing || steps.length === 0) return;
    last.current = performance.now();

    const tick = (now: number) => {
      const dt = now - last.current;
      last.current = now;
      setT((prev) => {
        const step = steps[iRef.current];
        if (!step) return prev;
        const next = prev + dt / step.ms;
        if (next < 1) return next;
        if (iRef.current >= steps.length - 1) {
          setPlaying(false);
          return 1;
        }
        iRef.current += 1;
        setI(iRef.current);
        return 0;
      });
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, steps]);

  if (steps.length === 0) {
    return <p className="muted">No located events in this possession.</p>;
  }

  const step = steps[Math.min(i, steps.length - 1)];
  const ease = reduced ? 1 : t;
  const bx = step.x + (step.endX - step.x) * ease;
  const by = step.y + (step.endY - step.y) * ease;

  // The path travelled so far, for a trail behind the ball.
  const trail = steps.slice(0, i + 1).map((s, k) => (k === i ? `${bx},${by}` : `${s.endX},${s.endY}`));
  const trailPoints = [`${steps[0].x},${steps[0].y}`, ...trail].join(" ");

  const showFreeze = freezeFrame.length > 0 && i >= steps.length - 1;

  return (
    <div>
      <svg viewBox={`-2 -2 ${W + 4} ${H + 4}`} className="pitch" role="img"
           aria-label="Animated pitch showing the possession">
        {/* markings */}
        <rect x={0} y={0} width={W} height={H} className="turf" />
        {/* Mown stripes. Purely atmospheric, but it is the difference between
            a diagram of a pitch and something that reads as one. */}
        {[1, 3, 5].map((band) => (
          <rect key={band} x={band * 20} y={0} width={20} height={H} className="mow" />
        ))}
        <g className="lines">
          <rect x={0} y={0} width={W} height={H} />
          <line x1={60} y1={0} x2={60} y2={H} />
          <circle cx={60} cy={40} r={10} />
          <circle cx={60} cy={40} r={0.7} className="dot" />
          {/* penalty areas */}
          <rect x={0} y={18} width={18} height={44} />
          <rect x={102} y={18} width={18} height={44} />
          {/* six-yard boxes */}
          <rect x={0} y={30} width={6} height={20} />
          <rect x={114} y={30} width={6} height={20} />
          {/* goals */}
          <rect x={-2} y={36} width={2} height={8} className="goal" />
          <rect x={120} y={36} width={2} height={8} className="goal" />
          <circle cx={12} cy={40} r={0.7} className="dot" />
          <circle cx={108} cy={40} r={0.7} className="dot" />
        </g>

        {/* attack direction */}
        <g className="arrow">
          <line x1={52} y1={-1} x2={68} y2={-1} />
          <polygon points="68,-1 65,-2.2 65,0.2" />
        </g>

        <polyline points={trailPoints} className="trail" />

        {showFreeze &&
          freezeFrame.map((p, k) => (
            <circle key={k} cx={p.x} cy={p.y} r={1.6}
                    className={p.keeper ? "gk" : p.teammate ? "mate" : "opp"} />
          ))}

        <circle cx={bx} cy={by} r={step.isShot ? 2.2 : 1.7}
                className={`ball ${step.isShot ? "shot" : ""}`} />
        {step.pressure && <circle cx={bx} cy={by} r={3.4} className="pressure" />}
      </svg>

      <div className="controls">
        <button onClick={() => {
          if (i >= steps.length - 1) { iRef.current = 0; setI(0); setT(0); setPlaying(true); }
          else setPlaying((p) => !p);
        }}>
          {playing ? "Pause" : i >= steps.length - 1 ? "Replay" : "Play"}
        </button>
        <input type="range" min={0} max={steps.length - 1} value={Math.min(i, steps.length - 1)}
               aria-label="Scrub through the possession"
               onChange={(e) => {
                 const v = Number(e.target.value);
                 setPlaying(false); iRef.current = v; setI(v); setT(0);
               }} />
        <span className="step-label">
          {i + 1}/{steps.length} <code>{step.label}</code>
          {step.attacking ? "" : " (defence)"}
        </span>
      </div>

      {showFreeze && (
        <p className="muted small">
          Real freeze frame at the moment of the shot{endedInGoal ? " (goal)" : ""} —{" "}
          <span className="key key-mate">attacking</span>,{" "}
          <span className="key key-opp">defending</span>,{" "}
          <span className="key key-gk">keeper</span>. No positions are shown at other
          moments, because the data does not contain them.
        </p>
      )}
    </div>
  );
}
