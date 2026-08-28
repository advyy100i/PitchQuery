-- Silver: the shots an xG model is allowed to see.
--
-- Three exclusions, all of them the same three rules models/train_xg.py already
-- enforces in Python. Encoding them here as well is not duplication — it is what
-- lets the training script select from one table instead of carrying a WHERE
-- clause that has to be kept in step with a mart it does not own:
--
--   * penalties, which have fixed geometry and a ~78% conversion rate, and
--     flatter every metric that includes them;
--   * shots with no recorded geometry, which the model cannot score;
--   * `statsbomb_xg` stays a column and never becomes a feature. It rides along
--     for comparison only, and mart_xg_features passes it through under a name
--     that says so.

select
    s.event_id      as shot_id,
    s.match_id,
    s.competition_id,
    s.season_id,
    -- The grouping key the training split holds out. Built once, here, so the
    -- trainer cannot compose it differently from the drift report.
    s.competition_id || ':' || s.season_id as comp_season,
    s.team,
    s.player,
    s.x,
    s.y,
    s.distance      as distance_m,
    s.angle         as angle_rad,
    s.body_part,
    s.technique,
    s.shot_type,
    s.first_time,
    s.under_pressure,
    s.play_pattern,
    s.is_goal,
    s.statsbomb_xg,
    s.n_def_in_cone,
    s.dist_nearest_def,
    s.gk_dist_to_goal,
    s.gk_off_line,
    s.freeze_frame is not null as has_freeze_frame

from {{ source('pitchquery', 'shots') }} s
where s.shot_type <> 'Penalty'
  and s.distance is not null
  and s.angle is not null
