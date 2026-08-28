-- A defender inside the shooting cone is by definition one of the opponents
-- visible in the freeze frame, so the Python count can never exceed the SQL
-- count of the same frame.
--
-- This is the only test in the warehouse that checks Python against SQL rather
-- than data against a range, and it is the one worth having: the coordinates in
-- a freeze frame are recorded in the shooter's attacking direction, and every
-- past bug in this area has been a frame that was not mirrored with the event
-- it belongs to. Such a frame still passes every range test — the players are
-- all on the pitch — and fails this one immediately, because the cone would
-- pick up defenders the aggregate says are not there.

select
    shot_id,
    n_def_in_cone,
    n_opponents_in_frame
from {{ ref('mart_xg_features') }}
where has_freeze_frame
  and n_def_in_cone > n_opponents_in_frame
