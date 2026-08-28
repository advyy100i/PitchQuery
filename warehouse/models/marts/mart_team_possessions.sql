-- Gold: one row per team per competition/season, for the dashboard.
--
-- Two aggregates joined rather than one big group-by: possessions are the unit
-- the product retrieves, events are the unit that says what a team actually
-- did, and folding them together in a single pass would multiply the possession
-- counts by the number of events in each one.

with possession_side as (

    select
        p.competition,
        p.season,
        p.team,
        count(*)                                              as n_possessions,
        avg(p.n_events)::real                                 as mean_tokens,
        avg(p.duration_s)::real                               as mean_duration_s,
        count(*) filter (where p.ended_in_shot)               as n_ending_in_shot,
        count(*) filter (where p.ended_in_goal)               as n_ending_in_goal,
        sum(p.xg_sum)::real                                   as statsbomb_xg_total,
        -- Where moves start. A team that begins most possessions in the final
        -- third is pressing high; one that begins them in its own is not. This
        -- is the cheapest honest description of a style the corpus can give.
        count(*) filter (where p.start_zone like 'F-%')       as n_starting_final_third,
        count(*) filter (where p.start_zone like 'D-%')       as n_starting_own_third
    from {{ source('pitchquery', 'possessions') }} p
    group by 1, 2, 3

),

event_side as (

    select
        e.competition,
        e.season,
        e.team,
        count(*)                                                  as n_events,
        count(*) filter (where e.type_name = 'Pass')              as n_passes,
        count(*) filter (where e.type_name = 'Pass'
                           and e.pass_outcome = 'Complete')       as n_passes_completed,
        count(*) filter (where e.is_cross)                        as n_crosses,
        count(*) filter (where e.pass_technique = 'Through Ball') as n_through_balls,
        count(*) filter (where e.type_name = 'Shot')              as n_shots,
        count(*) filter (where e.type_name = 'Shot'
                           and e.shot_outcome = 'Goal')           as n_goals
    from {{ ref('stg_events') }} e
    group by 1, 2, 3

)

select
    p.competition,
    p.season,
    p.team,
    p.n_possessions,
    p.mean_tokens,
    p.mean_duration_s,
    p.n_ending_in_shot,
    p.n_ending_in_goal,
    p.statsbomb_xg_total,
    p.n_starting_final_third,
    p.n_starting_own_third,
    (p.n_ending_in_shot::real / nullif(p.n_possessions, 0))       as shot_rate,
    (p.n_starting_final_third::real / nullif(p.n_possessions, 0)) as high_start_rate,
    e.n_events,
    e.n_passes,
    e.n_passes_completed,
    (e.n_passes_completed::real / nullif(e.n_passes, 0))          as pass_completion,
    e.n_crosses,
    e.n_through_balls,
    e.n_shots,
    e.n_goals

from possession_side p
-- A left join, because `possessions` drops passages shorter than three tokens
-- while `events` keeps everything. A team can appear on the event side with no
-- possession row, but never the other way round; if it ever did, the not_null
-- test on n_events below is what says so.
left join event_side e
       on e.competition = p.competition
      and e.season = p.season
      and e.team = p.team
