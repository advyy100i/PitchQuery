-- Silver: the shot-moment picture, one row per player per shot.
--
-- The plan asked for one row per defender. This keeps the teammates too, behind
-- a flag, because staging that drops rows is a staging layer you cannot audit:
-- with only the opponents in the table there is no way to tell an eight-player
-- frame from a nineteen-player frame with eleven attackers, and those two mean
-- very different things about how well the moment was captured. Every consumer
-- filters `where not teammate`.
--
-- Materialised as a table (~200k rows): unnesting 11k JSONB arrays on every
-- select is the one JSONB cost in this warehouse worth paying once.

-- Tagged `hosted`: it reads shots.freeze_frame, which does ship. See
-- deploy/export_to_neon.py --dbt.
{{ config(materialized='table', tags=['hosted']) }}

select
    s.event_id                                  as shot_id,
    s.match_id,
    s.competition_id,
    s.season_id,
    ff.ordinality::int                          as player_idx,
    (ff.value -> 'location' ->> 0)::real        as x,
    (ff.value -> 'location' ->> 1)::real        as y,
    coalesce((ff.value ->> 'teammate')::boolean, false) as teammate,
    ff.value -> 'position' ->> 'name'           as position,
    ff.value -> 'position' ->> 'name' = 'Goalkeeper' as is_keeper,
    -- Distance from this player to the shot, in the shooter's frame. The
    -- geometric cone test stays in core/features.py — one implementation shared
    -- by training and serving is worth more than a second one in SQL that could
    -- disagree with it. This column is what mart_xg_features cross-checks the
    -- Python answer against.
    sqrt(power((ff.value -> 'location' ->> 0)::real - s.x, 2)
       + power((ff.value -> 'location' ->> 1)::real - s.y, 2))::real as dist_to_shot

from {{ source('pitchquery', 'shots') }} s
-- Joined to stg_shots rather than filtered independently, so this model covers
-- exactly the shots that are modelled and the referential test between them
-- holds by construction. Penalties carry freeze frames too; nothing downstream
-- scores them, and leaving them in made the relationship test fail on 44 rows
-- that were never wrong, only out of scope.
join {{ ref('stg_shots') }} t on t.shot_id = s.event_id
cross join lateral jsonb_array_elements(s.freeze_frame) with ordinality ff
where s.freeze_frame is not null
  and jsonb_array_length(coalesce(ff.value -> 'location', '[]'::jsonb)) >= 2
