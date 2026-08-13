"""Response models. Keeping these explicit means the frontend has a contract,
and it stops raw database rows leaking out of the API by accident."""
from typing import Optional

from pydantic import BaseModel, Field


class PossessionSummary(BaseModel):
    """One row in a result list. Enough to render a card without a second call."""
    possession_uid: str
    match_id: int
    team: str
    opponent: Optional[str] = None
    competition: Optional[str] = None
    season: Optional[str] = None
    play_pattern: Optional[str] = None
    zone_path: str
    token_string: str
    n_events: int
    duration_s: float
    ended_in_shot: bool
    ended_in_goal: bool
    xg_sum: float


class PlanTerm(BaseModel):
    """One phrase the parser recognised and what it produced."""
    phrase: str
    effect: str


class PlanResponse(BaseModel):
    """The English -> structured query translation, shown to the user verbatim.

    Every filter traces to the phrase that caused it, and `ignored` lists the
    words the parser could not place — so the readout is a real account of what
    was understood, not a decorative summary.
    """
    text: str
    filters: dict
    sequence_hint: str
    terms: list[PlanTerm]
    ignored: str = ""
    parse_ms: float = 0.0


class NoteClaim(BaseModel):
    """One sentence of the scouting note, with the evidence behind it.

    `uids` is not an annotation added after the fact — the sentence was
    computed from exactly these possessions, so it cannot cite anything else.
    """
    text: str
    uids: list[str]


class SearchResponse(BaseModel):
    results: list[PossessionSummary]
    n_candidates: int = Field(description="possessions passing the hard filters, before ranking")
    took_ms: float
    sequence_hint: str = Field(description="the token string that was actually matched")
    filters: dict = Field(description="the structured filters applied — shown in the UI for trust")
    ranker_uids: dict = Field(default_factory=dict,
                              description="per-ranker uid lists, for the sparse/dense comparison")
    plan: Optional[PlanResponse] = Field(
        default=None, description="set when the search came from an English query")
    note: list[NoteClaim] = Field(
        default_factory=list,
        description="scouting note; every claim cites the possessions it was computed from")


class EventPoint(BaseModel):
    """One step of the animation."""
    idx: int
    period: int
    minute: int
    second: int
    type: str
    player: Optional[str] = None
    team: Optional[str] = None
    is_attacking: bool
    x: Optional[float] = None
    y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    duration: Optional[float] = None
    under_pressure: bool = False
    token: Optional[str] = None


class FreezeFramePlayer(BaseModel):
    x: float
    y: float
    teammate: bool
    keeper: bool = False


class PossessionDetail(BaseModel):
    summary: PossessionSummary
    events: list[EventPoint]
    freeze_frame: list[FreezeFramePlayer] = Field(
        default_factory=list,
        description="player positions at the shot, when the possession ends in one")


class ShotComparison(BaseModel):
    """Both xG values side by side — the point of the whole benchmark."""
    event_id: str
    match_id: int
    team: Optional[str] = None
    player: Optional[str] = None
    x: float
    y: float
    distance: float
    angle: float
    body_part: Optional[str] = None
    shot_type: Optional[str] = None
    play_pattern: Optional[str] = None
    is_goal: bool
    statsbomb_xg: Optional[float] = None
    my_xg: Optional[float] = None
    n_def_in_cone: Optional[int] = None
    dist_nearest_def: Optional[float] = None
    gk_dist_to_goal: Optional[float] = None
    gk_off_line: Optional[float] = None
    freeze_frame: list[FreezeFramePlayer] = Field(default_factory=list)
