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
  ranker_uids: { sparse?: string[]; dense?: string[]; fused?: string[] };
  /** Which ranker ordered these: "sparse", "fused", "learned" or "shape". */
  ranker?: string;
  /** Time the learned reranker spent, separate from took_ms. */
  rerank_ms?: number | null;
  /**
   * Row id in search_log. POST it back to /click when a result is opened.
   *
   * Optional because search logging can be switched off server-side, and
   * because the API deploys minutes behind this page — a build that assumed it
   * was always present would send `undefined` to /click after every push.
   */
  search_id?: number | null;
  plan?: PlanResponse | null;
  note: NoteClaim[];
};

/** One possession rebuilt from the replayed event stream. See stream/producer.py. */
export type LiveMessage = {
  type: "token" | "possession_opened" | "possession_closed" | "replay_end";
  /** Always "replay". These are recorded events, never a live feed. */
  source: string;
  possession?: number;
  team?: string;
  opponent?: string | null;
  token?: string;
  tokens?: string[];
  token_string?: string;
  zone_path?: string;
  n_tokens?: number;
  duration_s?: number;
  minute?: number | null;
  second?: number | null;
  player?: string | null;
  my_xg_sum?: number | null;
  kept?: boolean;
  shots?: { player: string | null; minute: number | null; my_xg: number | null;
            note: string | null; outcome: string | null }[];
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

/**
 * Report that a result was opened, and where it was ranked.
 *
 * Fire-and-forget by design. A click is a side effect of the user reading
 * something: there is nothing to retry, nothing to show them if it fails, and
 * nothing about opening a clip should depend on a logging table being up. The
 * catch is therefore empty on purpose rather than by omission.
 *
 * `rank` is 1-based, matching what the user sees.
 */
export function reportClick(searchId: number | null | undefined,
                            uid: string, rank: number): void {
  if (!searchId) return;
  fetch(`${API}/click?${qs({ search_id: searchId, possession_uid: uid, rank })}`,
        { method: "POST", keepalive: true }).catch(() => {});
}

/** WebSocket URL for the replay feed, derived from the API origin. */
export function liveUrl(): string {
  return API.replace(/^http/, "ws") + "/live";
}

export async function meta(): Promise<Meta> {
  const r = await fetch(`${API}/meta`);
  if (!r.ok) throw new Error(`meta failed: ${r.status}`);
  return r.json();
}
