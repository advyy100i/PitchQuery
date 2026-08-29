-- Gold: the exact table models/train_xg.py trains on.
--
-- "Exact" is the point. The trainer used to carry its own SELECT with its own
-- exclusions, so the feature definition lived in a Python string and nothing
-- tested it. Here it is a table with a schema, a uniqueness test and a range
-- test on every column, and `dbt build` fails before a training run can read a
-- shot 200 yards from goal.
--
-- The geometry columns come from core/features.py, not from SQL. That is
-- deliberate: the API scores a shot with the same function at serve time, and
-- one implementation that both sides share is the whole no-skew argument. What
-- SQL adds is an independent count of who was in the frame, which is what the
-- `n_def_in_cone <= n_opponents_in_frame` test compares the Python answer
-- against — a mirroring bug or a bad parse breaks that inequality immediately.

-- Tagged `hosted`: this model reads only columns that survive the trip to the
-- deployed database, so `dbt build --select tag:hosted` can build it there. See
-- deploy/export_to_neon.py --dbt.
{{ config(tags=['hosted']) }}

with frame as (

    select
        shot_id,
        count(*) filter (where not teammate and not is_keeper) as n_opponents_in_frame,
        count(*) filter (where teammate)                       as n_teammates_in_frame,
        count(*) filter (where not teammate and is_keeper)     as n_keepers_in_frame,
        min(dist_to_shot) filter (where not teammate and not is_keeper)
                                                               as sql_dist_nearest_def
    from {{ ref('stg_freeze_frames') }}
    group by 1

)

select
    s.shot_id,
    s.match_id,
    s.competition_id,
    s.season_id,
    s.comp_season,
    s.team,
    s.player,

    -- label
    s.is_goal,

    -- geometry
    s.distance_m,
    s.angle_rad,
    s.x,
    s.y,

    -- situation
    s.body_part,
    s.technique,
    s.shot_type,
    s.first_time,
    s.under_pressure,
    s.play_pattern,

    -- freeze-frame context. Null, never zero, when there is no frame: LightGBM
    -- routes missing values natively and a zero here would invent an empty
    -- penalty area that nobody observed.
    s.n_def_in_cone,
    s.dist_nearest_def,
    s.gk_dist_to_goal,
    s.gk_off_line,
    s.has_freeze_frame,

    -- the SQL-side counts, for the cross-check tests
    f.n_opponents_in_frame,
    f.n_teammates_in_frame,
    f.n_keepers_in_frame,
    f.sql_dist_nearest_def,

    -- Comparison only. Named so that a `select *` into a feature frame is an
    -- obvious mistake rather than a silent leak.
    s.statsbomb_xg as reference_statsbomb_xg

from {{ ref('stg_shots') }} s
left join frame f on f.shot_id = s.shot_id
