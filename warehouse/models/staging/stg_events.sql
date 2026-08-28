-- Silver: typed event rows.
--
-- The loader already promotes the columns the retrieval path needs, because
-- serving must never parse 1.3 GB of JSONB on the hot path. This model unpacks
-- the rest — the qualifiers that describe *how* an action came out, which
-- nothing in the API reads but every analytical question asks about.
--
-- A view, not a table. It is typing and unpacking, not computation, and
-- materialising 1.6M rows twice to save a JSONB parse the marts do once is a
-- poor trade.

select
    e.event_id,
    e.match_id,
    e.idx,
    e.period,
    e.minute,
    e.second,
    e.type            as type_name,
    e.play_pattern,
    e.possession,
    e.possession_team,
    e.team,
    e.player,
    e.position,
    e.x,
    e.y,
    e.end_x,
    e.end_y,
    e.under_pressure,
    e.duration,
    e.token,

    -- Pass qualifiers. StatsBomb omits `pass.outcome` entirely on a completed
    -- pass, so absence is the success case and coalescing to 'Complete' here is
    -- what stops every downstream count of completions being silently null.
    (e.raw -> 'pass' ->> 'length')::real                     as pass_length,
    (e.raw -> 'pass' ->> 'angle')::real                      as pass_angle,
    e.raw -> 'pass' -> 'height' ->> 'name'                   as pass_height,
    coalesce(e.raw -> 'pass' -> 'outcome' ->> 'name',
             case when e.type = 'Pass' then 'Complete' end)  as pass_outcome,
    e.raw -> 'pass' -> 'technique' ->> 'name'                as pass_technique,
    coalesce((e.raw -> 'pass' ->> 'cross')::boolean, false)  as is_cross,
    coalesce((e.raw -> 'pass' ->> 'switch')::boolean, false) as is_switch,
    e.raw -> 'pass' -> 'type' ->> 'name'                     as pass_set_piece,

    e.raw -> 'shot' -> 'outcome' ->> 'name'                  as shot_outcome,
    e.raw -> 'duel' -> 'outcome' ->> 'name'                  as duel_outcome,
    e.raw -> 'dribble' -> 'outcome' ->> 'name'               as dribble_outcome,

    -- Denormalised from `matches` so every downstream model can group by
    -- competition without repeating the join.
    m.competition_id,
    m.season_id,
    m.competition,
    m.season,
    m.match_date

from {{ source('pitchquery', 'events') }} e
join {{ source('pitchquery', 'matches') }} m on m.match_id = e.match_id
