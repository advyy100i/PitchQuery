// Types mirror api/schemas.py. Keep them in step — the API is the contract.

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type PossessionSummary = {
  possession_uid: string;
  match_id: number;
  team: string;
  opponent: string | null;
  competition: string | null;
  season: string | null;
  play_pattern: string | null;
  zone_path: string;
  token_string: string;
  n_events: number;
  duration_s: number;
  ended_in_shot: boolean;
  ended_in_goal: boolean;
  /** StatsBomb's xG, summed over the possession's shots. Not displayed. */
  xg_sum: number;
  /** This project's model, summed the same way. Null when it declines them all. */
  my_xg_sum: number | null;
};

export type PlanTerm = { phrase: string; effect: string };

export type PlanResponse = {
  text: string;
  filters: Record<string, unknown>;
  sequence_hint: string;
  terms: PlanTerm[];
  ignored: string;
  parse_ms: number;
};

export type NoteClaim = { text: string; uids: string[] };

export type SearchResponse = {
  results: PossessionSummary[];
  n_candidates: number;
  took_ms: number;
  sequence_hint: string;
  filters: Record<string, unknown>;
  ranker_uids: { sparse?: string[]; dense?: string[] };
  plan?: PlanResponse | null;
  note: NoteClaim[];
};

export type EventPoint = {
  idx: number;
  period: number;
  minute: number;
  second: number;
  type: string;
  player: string | null;
  team: string | null;
  is_attacking: boolean;
  x: number | null;
  y: number | null;
  end_x: number | null;
  end_y: number | null;
  duration: number | null;
  under_pressure: boolean;
  token: string | null;
};

export type FreezeFramePlayer = { x: number; y: number; teammate: boolean; keeper: boolean };

/** The shot a possession ends in, scored by both models. */
export type ShotXG = {
  event_id: string;
  player: string | null;
  minute: number | null;
  distance: number;
  angle: number;
  body_part: string | null;
  shot_type: string | null;
  is_goal: boolean;
  statsbomb_xg: number | null;
  my_xg: number | null;
  /** Why my_xg is null, when it is. Penalties are excluded from training. */
  my_xg_note: string | null;
  /** True when this competition was held out of training entirely. */
  in_holdout: boolean | null;
  n_def_in_cone: number | null;
  dist_nearest_def: number | null;
  gk_dist_to_goal: number | null;
  gk_off_line: number | null;
};

export type PossessionDetail = {
  summary: PossessionSummary;
  events: EventPoint[];
  freeze_frame: FreezeFramePlayer[];
  /**
   * Every shot in the possession, in order. The last is the one the clip ends on.
   *
   * Optional on purpose, even though the API always sends it. The frontend
   * deploys to Vercel in seconds and the API to Render's free tier in minutes,
   * so after every push there is a window where this page is talking to an
   * older API that does not have this field yet. Typing it as required let
   * `detail.shots.length` compile, and reading `.length` of undefined does not
   * fail quietly — it takes down the whole React tree with "Application error:
   * a client-side exception has occurred". Optional makes the compiler insist
   * on the guard at every use.
   */
  shots?: ShotXG[];
};

export type Filters = {
  team?: string;
  competition?: string;
  play_pattern?: string;
  ended_in_shot?: boolean;
  ended_in_goal?: boolean;
  min_xg?: number;
  start_band?: string;
  end_band?: string;
};

export type Meta = {
  competitions: string[];
  teams: string[];
  play_patterns: string[];
  possessions: number;
  matches: number;
};

function qs(params: Record<string, unknown>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    u.set(k, String(v));
  }
  return u.toString();
}

export async function search(
  sequenceHint: string,
  filters: Filters,
  limit = 20
): Promise<SearchResponse> {
  const r = await fetch(`${API}/search?${qs({ sequence_hint: sequenceHint, ...filters, limit })}`);
  if (!r.ok) throw new Error(`search failed: ${r.status}`);
  return r.json();
}

/** Search from plain English. The API parses it with core/planner.py — no LLM. */
export async function ask(q: string, filters: Filters = {}, limit = 20): Promise<SearchResponse> {
  const r = await fetch(`${API}/search?${qs({ q, ...filters, limit })}`);
  if (!r.ok) throw new Error(`search failed: ${r.status}`);
  return r.json();
}

/** Retrieve by a shape drawn on the pitch. No text, no vectors, no model. */
export async function byShape(zones: string[], filters: Filters = {},
                              limit = 20): Promise<SearchResponse> {
  const r = await fetch(`${API}/shape?${qs({ zones: zones.join(","), ...filters, limit })}`);
  if (!r.ok) throw new Error(`shape search failed: ${r.status}`);
  return r.json();
}

export async function similar(uid: string, limit = 20): Promise<SearchResponse> {
  const r = await fetch(`${API}/similar/${encodeURIComponent(uid)}?limit=${limit}`);
  if (!r.ok) throw new Error(`similar failed: ${r.status}`);
  return r.json();
}

export async function possession(uid: string): Promise<PossessionDetail> {
  const r = await fetch(`${API}/possession/${encodeURIComponent(uid)}`);
  if (!r.ok) throw new Error(`possession failed: ${r.status}`);
  return r.json();
}

export async function meta(): Promise<Meta> {
  const r = await fetch(`${API}/meta`);
  if (!r.ok) throw new Error(`meta failed: ${r.status}`);
  return r.json();
}
