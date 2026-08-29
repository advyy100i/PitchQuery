// Types mirror api/ops.py, the same way lib/api.ts mirrors api/schemas.py.
//
// Every field that a hosted deployment cannot fill is optional or nullable here
// rather than assumed: the Neon copy has no `ingest_watermark` and no dbt
// schemas, and Render has no MLflow. Typing those as required would compile and
// then blank the page on the one deployment anyone else looks at.

import { API } from "./api";

export type PipelineRun = {
  competition_id: number;
  season_id: number;
  last_match_id: number | null;
  last_run_at: string | null;
  rows_loaded: number;
};

export type LayerRow = {
  layer: "bronze" | "silver" | "gold";
  table: string;
  qualified: string;
  rows: number | null;
  /** True when count(*) hit the 3 s ceiling and this is the planner's estimate. */
  estimated: boolean;
  /** "ok", "missing", or the database's own message. */
  state: string;
};

export type Champion = {
  /** "mlflow" when the registry answered, "baseline" for the committed JSON. */
  source: "mlflow" | "baseline" | null;
  version: string | null;
  metrics: Record<string, number | null>;
  params: Record<string, string | number | null>;
  commit?: string | null;
  note?: string;
  error: string | null;
};

export type DriftShift = {
  feature: string;
  reference_mean: number;
  current_mean: number;
  cohens_d: number;
};

export type DriftReport = {
  name: string;
  split: string;
  date: string;
  reference: string;
  current: string;
  n_reference: number;
  n_current: number;
  shifts: DriftShift[];
};

export type DailyRow = {
  day: string;
  searches: number;
  clicks: number;
  p95_ms: number | null;
};

export type DeepClick = {
  query_text: string | null;
  rank: number;
  possession_uid: string;
  ranker: string | null;
  ts: string | null;
};

export type Ops = {
  generated_at: string;
  cache_ttl_s: number;
  pipeline: { runs: PipelineRun[]; error: string | null; hint: string | null };
  layers: { tables: LayerRow[]; hint: string | null };
  champion: Champion;
  drift: { reports: DriftReport[]; hint: string | null };
  queries: {
    daily: DailyRow[];
    unparsed: { word: string; n: number }[];
    deep_clicks: DeepClick[];
    totals: {
      searches?: number;
      clicks?: number;
      deep_clicks?: number;
      unknown_words?: number;
    };
    error: string | null;
    hint: string | null;
  };
};

export async function ops(): Promise<Ops> {
  const r = await fetch(`${API}/ops`);
  if (!r.ok) throw new Error(`ops failed: ${r.status}`);
  return r.json();
}
