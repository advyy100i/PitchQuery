"""Deterministic English -> structured query translation.

Plan §7 specified an LLM for this job. A rule parser replaces it, for reasons
that hold up better than the cost saving:

  * It is free and needs no API key, so the whole system stays self-contained.
  * It is deterministic — the same sentence always produces the same query, so
    a retrieval regression is always the retrieval's fault and never the
    planner's mood.
  * Every filter is traceable to the exact phrase that produced it, which makes
    the "show the user the translated query" feature honest rather than
    decorative (see `Plan.explain`).
  * It is testable. `eval/score_planner.py` scores it against the 30
    hand-written structured queries in eval/queries.yaml, so its quality is a
    measured number, not a vibe.

The cost is coverage: it understands the vocabulary below and nothing else.
An LLM would generalise to phrasings this misses. That trade is stated in the
README rather than hidden.

Design rule, and the reason precision holds up:

    HARD FILTERS come only from unambiguous phrases — team, competition,
    play pattern, outcome, band, xG, length. A wrong filter returns an empty
    result set.

    CHANNELS AND ACTIONS go into the sequence_hint, which only affects
    ranking. A wrong hint costs a few places, not the whole answer.

So "left half-space" steers the hint towards F-LI; it does not add an
`end_zone = 'F-LI'` filter that would throw away every near-miss.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from core.retrieval import Filters
from core.zones import BANDS, CHANNELS

# --- vocabulary ---------------------------------------------------------------
# Each entry maps a canonical value to its synonym set. Order within a set does
# not matter; the matcher always prefers the LONGEST phrase that matches at a
# position, so "left half-space" wins over "left" and "goal kick" over "goal".

CHANNEL_TERMS = {
    "L": ("left wing", "left-wing", "left flank", "left touchline", "down the left",
          "wide left", "left side", "left byline"),
    "LI": ("left half-space", "left halfspace", "left half space", "inside-left",
           "inside left", "left channel", "left inside channel", "inside-left channel"),
    "C": ("centrally", "central", "centre", "center", "through the middle",
          "middle of the pitch", "down the middle"),
    "RI": ("right half-space", "right halfspace", "right half space", "inside-right",
           "inside right", "right channel", "right inside channel", "inside-right channel"),
    "R": ("right wing", "right-wing", "right flank", "right touchline", "down the right",
          "wide right", "right side", "right byline"),
}

BAND_TERMS = {
    "D": ("own third", "defensive third", "own half", "from defence", "from defense",
          "from the back", "back line", "deep in their own half", "defensive area"),
    "M": ("middle third", "midfield", "centre of the pitch", "middle of the park"),
    "F": ("final third", "attacking third", "opposition half", "opponents half",
          "opponent's half", "attacking half", "forward third"),
}

# Actions the grammar can emit. PASS is the fallback and is not matched directly.
ACTION_TERMS = {
    "CROSS": ("cross", "crossed", "crosses", "crossing", "whipped in", "delivery"),
    "THROUGH": ("through ball", "through-ball", "played in behind", "in behind",
                "threaded", "slipped in", "split the defence"),
    "SWITCH": ("switch of play", "switched", "switch", "cross-field", "crossfield",
               "diagonal ball", "changed flanks", "other flank"),
    "DRIB": ("dribble", "dribbled", "dribbling", "beat his man", "beat their man",
             "took on", "take-on", "run at"),
    "CARRY": ("carry", "carried", "drove forward", "driving run", "ran with the ball",
              "overlapping run", "overlap"),
    "SHOT": ("shot", "shoots", "shoot", "effort", "attempt", "struck", "strike",
             "header", "headed goalward", "goalward", "finish"),
    "RECOV": ("turnover", "recovery", "recovered", "won the ball", "wins the ball",
              "regain", "won possession", "pressing high", "high press", "press high",
              "wins it back", "counter-press"),
    "INT": ("interception", "intercepted", "cut out"),
    "DUEL": ("duel", "tackle", "aerial duel"),
    "CLR": ("clearance", "cleared"),
    "LOSS": ("lost the ball", "dispossessed", "miscontrol", "gave it away"),
}

# A cutback is its own beat: a wide final-third ball into the box, then a
# central shot. It is not tagged CROSS in the grammar, so it gets its own rule.
CUTBACK_TERMS = ("cutback", "cut-back", "cut back", "squared it", "pulled it back",
                 "laid it back")

# Two passes exchanged in place — renders as PASS/RECV/PASS/RECV.
COMBO_TERMS = ("one-two", "one two", "give and go", "give-and-go", "combination",
               "lay-off", "layoff", "lay off", "knock-down", "wall pass")

PATTERN_TERMS = {
    "From Corner": ("corner", "corner kick"),
    "From Free Kick": ("free kick", "free-kick", "freekick", "set piece", "set-piece"),
    "From Throw In": ("throw-in", "throw in", "throwin"),
    "From Counter": ("counter-attack", "counter attack", "counterattack", "counter",
                     "on the break", "fast break", "transition"),
    "From Goal Kick": ("goal kick", "goal-kick", "goalkick"),
    "From Keeper": ("from the goalkeeper", "from the keeper", "from their keeper"),
    "From Kick Off": ("kick-off", "kick off", "kickoff"),
    "Regular Play": ("open play", "regular play"),
}

SHOT_OUTCOME = ("ends in a shot", "ending in a shot", "ends with a shot",
                "ending with a shot", "leads to a shot", "leading to a shot",
                "creates a shot", "results in a shot", "produces a shot",
                "that creates a shot", "shot at the end")
GOAL_OUTCOME = ("goal", "scored", "scoring", "scores", "ends in a goal", "finished off")

BOX_TERMS = ("into the box", "in the box", "inside the box", "penalty area",
             "penalty box", "six-yard box", "six yard box", "18-yard box")
OUTSIDE_BOX_TERMS = ("outside the box", "from distance", "long range", "long-range",
                     "edge of the box", "edge of the area")
PRESSURE_TERMS = ("under pressure", "under heavy pressing", "pressure-resistant",
                  "heavy pressing", "under press", "pressed", "under the press")
DIRECT_TERMS = ("long ball", "direct", "route one", "long pass")
QUICK_TERMS = ("quick", "fast", "rapid", "immediate", "immediately", "swift")
BACKWARD_TERMS = ("recycled backwards", "recycled", "backwards", "back to the defence",
                  "played back", "reset")

# Length and xG thresholds. Stated here rather than inline so the numbers are
# reviewable in one place — they are judgement calls, not measurements.
LONG_TERMS = ("long possession", "extended possession", "sustained possession",
              "sustained spell", "long spell")
PATIENT_TERMS = ("patient", "patiently", "keep-ball", "controlled possession")
HIGH_XG_TERMS = ("high-xg", "high xg", "big chance", "clear chance", "clear-cut chance",
                 "golden chance", "high quality chance")
LONG_MIN_EVENTS = 25
PATIENT_MIN_EVENTS = 20
HIGH_XG_THRESHOLD = 0.3

# Actions that mark where a possession BEGAN rather than where it was going.
# Used to disambiguate a bare "in the <band>" — see the zone rules in plan().
BALL_WINNING = frozenset({"RECOV", "INT", "DUEL", "CLR"})

# --- archetypes ---------------------------------------------------------------
# Phrases that describe a whole passage rather than a single action. Each
# expands into several beats, because a one-token hint retrieves badly: the
# corpus averages 19 tokens per possession, so TF-IDF over 1-3 grams has almost
# nothing to match against a lone 'INT@M-C'.

BUILDUP_TERMS = ("build-up", "build up", "building out", "playing out from the back",
                 "played out from the back", "build out")
REBOUND_TERMS = ("rebound", "blocked", "parried", "second ball", "spilled",
                 "loose ball", "follow-up")
ATTACK_TERMS = ("attack", "attacking", "break forward", "broke forward",
                "immediate attack", "counter-attack", "forward")
# Shortest hint worth issuing. Below this the ranker has too little to match.
MIN_HINT_BEATS = 3

# Where each restart is taken from, and whether it is played into the box.
# (band, default channel, delivered into the box)
SETPIECE_START = {
    "From Corner": ("F", "R", True),
    "From Free Kick": ("F", "LI", True),
    "From Throw In": ("F", "R", False),
    "From Goal Kick": ("D", "C", False),
    "From Kick Off": ("M", "C", False),
}


@dataclass
class Vocabulary:
    """Team/competition names, loaded from the database rather than hardcoded.

    Matching against real values is what lets 'Barcelona' become a filter while
    'Bayer Leverkusen' does too, without a curated list going stale.
    """
    teams: tuple = ()
    competitions: tuple = ()

    @classmethod
    def from_db(cls, conn) -> "Vocabulary":
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT team FROM possessions WHERE team IS NOT NULL")
            teams = tuple(r[0] for r in cur.fetchall())
            cur.execute("SELECT DISTINCT competition FROM possessions "
                        "WHERE competition IS NOT NULL")
            comps = tuple(r[0] for r in cur.fetchall())
        return cls(teams=teams, competitions=comps)


@dataclass
class Beat:
    """One action in the intended passage, with an optional zone."""
    action: str
    band: Optional[str] = None
    channel: Optional[str] = None
    box: bool = False
    pressure: bool = False


@dataclass
class Match:
    """One phrase the parser recognised, and what it produced. Drives the UI
    readout — the user sees which words became which constraint."""
    phrase: str
    start: int
    effect: str


@dataclass
class Plan:
    text: str
    filters: Filters
    sequence_hint: str
    matches: list = field(default_factory=list)
    beats: list = field(default_factory=list)

    def explain(self) -> list:
        return [{"phrase": m.phrase, "effect": m.effect} for m in self.matches]

    @property
    def unmatched(self) -> str:
        """Words the parser ignored — the honest measure of its coverage."""
        spans = sorted((m.start, m.start + len(m.phrase)) for m in self.matches)
        out, cursor = [], 0
        for s, e in spans:
            if s > cursor:
                out.append(self.text[cursor:s])
            cursor = max(cursor, e)
        out.append(self.text[cursor:])
        return " ".join(" ".join(out).split())


# --- matching engine ----------------------------------------------------------

def _phrase_pattern(phrases) -> re.Pattern:
    """One alternation, longest phrase first, so 'left half-space' beats 'left'.

    The final word takes an optional inflection so 'finishing' matches 'finish'
    and 'crosses' matches 'cross'. Without it the parser is brittle in exactly
    the way a rule system is accused of being — 'finishing from the inside
    right' silently produced no shot at all.
    """
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(p) + r"(?:s|es|ed|ing|d)?"
                                        for p in ordered) + r")\b",
                      re.IGNORECASE)


def _spans(*hit_lists) -> list:
    """(start, end) spans of already-consumed phrases."""
    out = []
    for hits in hit_lists:
        for item in hits:
            if item is None:
                continue
            phrase, pos = (item[1], item[2]) if len(item) == 3 else (item[0], item[1])
            out.append((pos, pos + len(phrase)))
    return out


def _find_outside(text: str, phrases, spans) -> Optional[tuple]:
    """First match that does not sit inside an already-claimed phrase.

    Stops 'attacking third' (a band) from also firing the 'attack' archetype.
    """
    for m in _phrase_pattern(phrases).finditer(text):
        s, e = m.start(), m.end()
        if any(s < span_e and e > span_s for span_s, span_e in spans):
            continue
        return m.group(0), s
    return None


def _scan(text: str, table: dict) -> list:
    """Find every (canonical_value, matched_phrase, position) in the text."""
    hits = []
    for value, phrases in table.items():
        for m in _phrase_pattern(phrases).finditer(text):
            hits.append((value, m.group(0), m.start()))
    return hits


def _find(text: str, phrases) -> Optional[tuple]:
    m = _phrase_pattern(phrases).search(text)
    return (m.group(0), m.start()) if m else None


def _match_name(text: str, names) -> Optional[tuple]:
    """Longest real team/competition name appearing in the text.

    Longest-wins matters: the corpus has both 'England' and "England Women's",
    and a query naming one must not silently select the other.
    """
    best = None
    low = text.lower()
    for name in names:
        i = low.find(name.lower())
        if i < 0:
            continue
        # Require a word boundary so 'Spain' doesn't match inside another word.
        before_ok = i == 0 or not low[i - 1].isalnum()
        after = i + len(name)
        after_ok = after >= len(low) or not low[after].isalnum()
        if before_ok and after_ok and (best is None or len(name) > len(best[0])):
            best = (name, i)
    return best


# --- hint rendering -----------------------------------------------------------

# Where each action tends to happen when the sentence does not say.
DEFAULT_BAND = {
    "CROSS": "F", "SHOT": "F", "DRIB": "F", "THROUGH": "M", "SWITCH": "M",
    "CARRY": "M", "PASS": "M", "RECOV": "M", "INT": "M", "DUEL": "M",
    "CLR": "D", "LOSS": "M", "SETP": "F",
}


def _zone(band: str, channel: str) -> str:
    return f"{band}-{channel}"


# Only these actions can carry '+' or '>'. In core/zones.py the modifiers are
# derived from an event's end_location, and only passes, carries and shots have
# one — so a receipt or a dribble physically cannot be tagged progressive or
# into-the-box. Emitting 'DRIB@F-C>' produces a token that appears NOWHERE in
# the corpus, so the whole n-gram around it matches nothing and the query goes
# dead. Shots are excluded separately: there the modifiers would leak outcome.
CAN_TAKE_MODS = frozenset({"PASS", "CROSS", "THROUGH", "SWITCH", "CARRY", "SETP"})

# Balls that are progressive by nature — a through ball or a switch that gains
# no ground is not what the words mean.
INHERENTLY_PROGRESSIVE = frozenset({"THROUGH", "SWITCH"})


def _tok(action: str, band: str, channel: str, *, prog=False, box=False, pressure=False) -> str:
    mods = ""
    if action in CAN_TAKE_MODS:
        if prog or action in INHERENTLY_PROGRESSIVE:
            mods += "+"
        if box:
            mods += ">"
    if pressure:
        mods += "^"
    return f"{action}@{_zone(band, channel)}{mods}"


def _render_hint(beats: list, *, start_band: Optional[str], end_channel: str,
                 pressure: bool) -> str:
    """Turn ordered beats into a synthetic possession in the token grammar.

    The connective RECV/CARRY tokens matter: the corpus is full of them (a
    completed pass is always followed by a receipt), so a hint without them
    matches poorly against real possessions.
    """
    toks: list = []
    if not beats:
        return ""

    first = beats[0]
    cur_band = first.band or DEFAULT_BAND.get(first.action, "M")
    cur_ch = first.channel or "C"

    # Lead-in: if the passage opens in the final third, show it arriving there.
    if start_band and start_band != cur_band:
        toks += [_tok("PASS", start_band, cur_ch, prog=True),
                 _tok("RECV", cur_band, cur_ch)]
    elif cur_band == "F" and first.action in ("CROSS", "DRIB"):
        toks += [_tok("PASS", "M", cur_ch, prog=True),
                 _tok("RECV", "F", cur_ch),
                 _tok("CARRY", "F", cur_ch)]

    for i, beat in enumerate(beats):
        band = beat.band or DEFAULT_BAND.get(beat.action, cur_band)
        ch = beat.channel or cur_ch
        moved = (band, ch) != (cur_band, cur_ch)
        prog = BANDS.index(band) > BANDS.index(cur_band) if moved else False

        # Travelling between zones costs a pass and a receipt.
        if moved and i > 0:
            toks.append(_tok("PASS", cur_band, cur_ch, prog=prog,
                             pressure=pressure))
            toks.append(_tok("RECV", band, ch, pressure=pressure))
            if beat.action == "RECV":
                # The link's receipt IS this beat — don't emit it twice.
                cur_band, cur_ch = band, ch
                continue

        toks.append(_tok(beat.action, band, ch, box=beat.box,
                         pressure=beat.pressure or pressure))
        cur_band, cur_ch = band, ch

        # A ball into the box is received centrally by whoever attacks it.
        if beat.box and beat.action != "SHOT":
            toks.append(_tok("RECV", "F", end_channel))
            cur_band, cur_ch = "F", end_channel

    return " ".join(toks)


# --- the parser ---------------------------------------------------------------

def plan(text: str, vocab: Optional[Vocabulary] = None, *,
         aggressive: bool = False) -> Plan:
    """Translate an English description into filters plus a token-grammar hint.

    `aggressive=True` additionally emits `must_include` token constraints for
    explicitly-named actions ('a switch of play' -> every result must contain a
    SWITCH token). That raises precision sharply but is a hard filter, so it is
    off by default and reported as a separate number.
    """
    vocab = vocab or Vocabulary()
    f = Filters()
    matches: list = []

    def note(phrase, start, effect):
        matches.append(Match(phrase=phrase, start=start, effect=effect))

    # -- named entities --------------------------------------------------------
    hit = _match_name(text, vocab.teams)
    if hit:
        f.team = hit[0]
        note(hit[0], hit[1], f"team = {hit[0]}")
    hit = _match_name(text, vocab.competitions)
    if hit:
        f.competition = hit[0]
        note(hit[0], hit[1], f"competition = {hit[0]}")

    # -- play pattern ----------------------------------------------------------
    # Longest match across all patterns, so 'goal kick' beats the 'goal' in
    # GOAL_OUTCOME and 'counter-attack' beats 'counter'.
    pattern_hits = _scan(text, PATTERN_TERMS)
    if pattern_hits:
        value, phrase, pos = max(pattern_hits, key=lambda h: len(h[1]))
        f.play_pattern = value
        note(phrase, pos, f"play_pattern = {value}")

    # -- action beats (scanned early: the zone rules below depend on them) -----
    action_hits = _dedupe_overlaps(sorted(_scan(text, ACTION_TERMS), key=lambda h: h[2]))
    cutback = _find(text, CUTBACK_TERMS)
    combo = _find(text, COMBO_TERMS)

    # -- outcome ---------------------------------------------------------------
    goal_hit = _find(text, GOAL_OUTCOME)
    shot_hit = _find(text, SHOT_OUTCOME)
    # Any mention of a shot at all means the passage should end in one — the
    # earlier version only matched full phrases like "ends in a shot", so
    # "shot from a rebound" and "headed goalward" silently produced no filter.
    if shot_hit is None:
        bare = next(((p, i) for v, p, i in action_hits if v == "SHOT"), None)
        shot_hit = bare
    # A cutback is played for someone to finish; the shot is implied.
    if shot_hit is None and cutback:
        shot_hit = cutback

    # A goal is also a shot; only the stronger constraint is applied.
    if goal_hit and not (f.play_pattern == "From Goal Kick"
                         and goal_hit[0].lower() == "goal"):
        f.ended_in_goal = True
        note(goal_hit[0], goal_hit[1], "ended_in_goal = true")
    elif shot_hit:
        f.ended_in_shot = True
        note(shot_hit[0], shot_hit[1], "ended_in_shot = true")

    # -- xG and length ---------------------------------------------------------
    hit = _find(text, HIGH_XG_TERMS)
    if hit:
        f.min_xg = HIGH_XG_THRESHOLD
        note(hit[0], hit[1], f"min_xg = {HIGH_XG_THRESHOLD}")
    hit = _find(text, LONG_TERMS)
    if hit:
        f.min_events = LONG_MIN_EVENTS
        note(hit[0], hit[1], f"min_events = {LONG_MIN_EVENTS}")
    else:
        hit = _find(text, PATIENT_TERMS)
        if hit:
            f.min_events = PATIENT_MIN_EVENTS
            note(hit[0], hit[1], f"min_events = {PATIENT_MIN_EVENTS}")

    # -- modifiers -------------------------------------------------------------
    box_hit = _find(text, BOX_TERMS)
    outside_hit = _find(text, OUTSIDE_BOX_TERMS)
    pressure_hit = _find(text, PRESSURE_TERMS)
    direct_hit = _find(text, DIRECT_TERMS)
    if box_hit and not outside_hit:
        note(box_hit[0], box_hit[1], "hint: ends in the box (>)")
    if outside_hit:
        note(outside_hit[0], outside_hit[1], "hint: shot from outside the box")
    if pressure_hit:
        note(pressure_hit[0], pressure_hit[1], "hint: under pressure (^)")
    if direct_hit:
        note(direct_hit[0], direct_hit[1], "hint: direct/progressive (+)")

    # -- zones -----------------------------------------------------------------
    band_hits = _dedupe_overlaps(sorted(_scan(text, BAND_TERMS), key=lambda h: h[2]))
    channel_hits = _dedupe_overlaps(sorted(_scan(text, CHANNEL_TERMS),
                                           key=lambda h: h[2]))
    # Channels steer the hint rather than adding a filter, but they must still
    # be recorded — otherwise `unmatched` claims the parser ignored words it
    # actually used, and the coverage report lies in the flattering direction.
    for value, phrase, pos in channel_hits:
        note(phrase, pos, f"hint: {value} channel")

    # A band phrase becomes a hard filter only when the sentence says where the
    # move STARTS or ENDS. The hard case is a bare "in the ...", which can mean
    # either — compare:
    #     "high turnover IN THE FINAL THIRD leading to a shot"   (where it began)
    #     "England attacking the left channel IN THE FINAL THIRD" (where it ended)
    # The disambiguator is the nearest preceding action: a ball-winning verb
    # marks where possession started, anything else marks where it arrived.
    for value, phrase, pos in band_hits:
        lead = text[max(0, pos - 22):pos].lower()
        explicit_start = phrase.lower() in (
            "own third", "own half", "from defence", "from defense", "from the back")
        motion_to = any(w in lead for w in ("into the", "reaches the", "to the",
                                            "arrives in", "reaching the"))

        if explicit_start or (value == "D" and any(w in lead for w in
                                                   ("from", "out of", "in their"))):
            f.start_band = "D" if value == "D" else value
            note(phrase, pos, f"start_band = {f.start_band}")
        elif motion_to and value == "F":
            f.end_band = "F"
            note(phrase, pos, "end_band = F")
        elif "in the" in lead:
            prev = _preceding_action(action_hits, pos)
            if prev in BALL_WINNING:
                f.start_band = value
                note(phrase, pos, f"start_band = {value} (where the {prev} happened)")
            elif value == "F":
                f.end_band = "F"
                note(phrase, pos, "end_band = F")
            else:
                note(phrase, pos, f"hint: {value} band")
        else:
            note(phrase, pos, f"hint: {value} band")

    # "...ends with a shot from the right half-space" names the finishing zone
    # explicitly, which is specific enough to be a hard filter.
    shot_zone = _shot_from_channel(text)
    if shot_zone:
        channel, phrase, pos = shot_zone
        f.end_zone = f"F-{channel}"
        note(phrase, pos, f"end_zone = F-{channel}")

    # -- action beats ----------------------------------------------------------
    beats: list = []
    for value, phrase, pos in action_hits:
        band = _nearest(band_hits, pos)
        channel = _nearest(channel_hits, pos)
        if value == "SHOT":
            # A shot does not inherit a channel just because a wing was named
            # earlier in the sentence: "down the left wing ending in a shot"
            # means the ball came from the left and was struck in the middle.
            # Only an explicit "shot from the <channel>" moves it off centre.
            channel = f.end_zone.split("-")[1] if f.end_zone else "C"
            band = "F"
        beats.append(Beat(action=value, band=band, channel=channel))
        note(phrase, pos, f"action {value}")

    # A set-piece possession always opens with the restart itself. Without this
    # the hint for "free kick played into the box" contains no SETP token at
    # all, and misses the one token every such possession actually starts with.
    if f.play_pattern in SETPIECE_START and not any(b.action == "SETP" for b in beats):
        band, default_ch, into_box = SETPIECE_START[f.play_pattern]
        ch = (channel_hits[0][0] if channel_hits else default_ch)
        beats.insert(0, Beat("SETP", band, ch,
                             box=into_box and bool(box_hit or f.play_pattern
                                                   == "From Corner")))

    # "pressing high" / "high turnover" means high up the pitch, and the word
    # carrying that is an adjective the band table cannot see.
    for b in beats:
        if b.action in BALL_WINNING and b.band is None:
            spot = next((p for v, _ph, p in action_hits if v == b.action), None)
            if spot is not None and "high" in text[max(0, spot - 12):spot + 24].lower():
                b.band = "F"

    # -- archetypes ------------------------------------------------------------
    buildup = _find(text, BUILDUP_TERMS)
    if buildup:
        # Playing out from the back: a few touches in the defensive third, then
        # progression. Also pins start_band, which the phrase states outright.
        f.start_band = "D"
        beats = [Beat("RECV", "D", "C"), Beat("CARRY", "D", "C"),
                 Beat("PASS", "D", "LI"), Beat("RECV", "M", "LI")] + beats
        note(buildup[0], buildup[1], "start_band = D, build-up sequence")

    rebound = _find(text, REBOUND_TERMS)
    if rebound:
        # A rebound needs the ball won back between two attempts — that gap is
        # what the rubric actually tests for.
        shots = [i for i, b in enumerate(beats) if b.action == "SHOT"]
        insert_at = shots[1] if len(shots) >= 2 else (shots[0] if shots else len(beats))
        beats.insert(insert_at, Beat("RECOV", "F", "C"))
        note(rebound[0], rebound[1], "action RECOV (rebound between attempts)")

    if cutback:
        ch = _nearest(channel_hits, cutback[1]) or "R"
        beats.append(Beat(action="PASS", band="F", channel=ch, box=True))
        note(cutback[0], cutback[1], "action cutback (PASS into the box)")

    if combo:
        ch = _nearest(channel_hits, combo[1]) or "LI"
        beats.append(Beat(action="PASS", band="F", channel=ch))
        beats.append(Beat(action="PASS", band="F", channel=ch))
        note(combo[0], combo[1], "action one-two (PASS/PASS)")

    # Apply box/pressure modifiers to the last beat that can carry them.
    if box_hit and not outside_hit and beats:
        carrier = next((b for b in reversed(beats) if b.action in CAN_TAKE_MODS), None)
        if carrier is not None:
            carrier.box = True
        else:
            # "dribble into the box" — a DRIB cannot carry '>', so the entry is
            # made by the carry that follows it, which is what the data shows.
            last = beats[-1]
            beats.append(Beat("CARRY", "F", last.channel or "C", box=True))

    # Terminal shot: ensure the hint ends the way the filters say it does.
    end_channel = channel_hits[-1][0] if channel_hits else "C"
    wants_shot = f.ended_in_shot or f.ended_in_goal
    has_shot = any(b.action == "SHOT" for b in beats)
    if wants_shot and not has_shot:
        beats.append(Beat(action="SHOT", band="F",
                          channel="C" if not outside_hit else "C"))
    if outside_hit:
        for b in beats:
            if b.action == "SHOT":
                b.band, b.channel, b.box = "M", "C", False

    # "turned into an immediate attack" — the sentence names no further action,
    # but it does say the ball ended up in the final third. Say so in tokens.
    attack = _find_outside(text, ATTACK_TERMS,
                           _spans(band_hits, channel_hits, pattern_hits))
    if attack:
        if not beats:
            ch = channel_hits[-1][0] if channel_hits else "C"
            beats += [Beat("RECOV", "M", ch), Beat("CARRY", "M", ch),
                      Beat("RECV", "F", ch)]
            note(attack[0], attack[1], "hint: attacking sequence into the final third")
        else:
            last = beats[-1]
            last_band = last.band or DEFAULT_BAND.get(last.action, "M")
            if last_band != "F" and last.action != "SHOT":
                ch = last.channel or "C"
                beats += [Beat("CARRY", last_band, ch), Beat("RECV", "F", ch)]
                note(attack[0], attack[1], "hint: progression into the final third")

    # A shot with no other content still needs somewhere to be taken from.
    if not beats:
        ch = channel_hits[-1][0] if channel_hits else "C"
        band = band_hits[-1][0] if band_hits else "F"
        beats.append(Beat(action="CARRY", band=band, channel=ch))

    # Pad a thin hint into something possession-shaped. A single token has
    # almost no n-gram overlap with a real 19-token possession, so a sparse
    # query returns close to noise however good the ranker is.
    # Never pad in front of a restart: a corner or free kick IS the start of the
    # possession, so touches before it describe something that cannot happen.
    if len(beats) < MIN_HINT_BEATS and beats[0].action != "SETP":
        first = beats[0]
        band = first.band or DEFAULT_BAND.get(first.action, "M")
        ch = first.channel or "C"
        beats = [Beat("RECV", band, ch), Beat("CARRY", band, ch)] + beats
        # NOTE: padding with an arriving pass instead (PASS@M-LI RECV@F-LI ...)
        # was tried and measured WORSE — P@5 0.680 -> 0.592 — and did not fix
        # the underlying problem, so it was reverted rather than kept on
        # intuition. The real cause is that TF-IDF cosine favours short
        # documents: a 3-token possession matching the hint exactly outscores a
        # 20-token one containing the same tokens, because the longer vector is
        # diluted. A thin hint therefore retrieves thin possessions whatever it
        # is padded with. Fixing that means length normalisation in the ranker,
        # not more guessing in the parser. Recorded in docs/planner_eval.md.

    hint = _render_hint(beats, start_band=f.start_band,
                        end_channel="C" if end_channel in ("L", "R") else end_channel,
                        pressure=bool(pressure_hit))

    if aggressive:
        must = sorted({b.action for b in beats
                       if b.action in ("CROSS", "SWITCH", "THROUGH", "DRIB", "INT")})
        if must:
            f.must_include = must

    return Plan(text=text, filters=f, sequence_hint=hint, matches=matches, beats=beats)


def _dedupe_overlaps(hits: list) -> list:
    """Keep the longest match at each position; drop anything it covers."""
    kept: list = []
    for value, phrase, pos in sorted(hits, key=lambda h: (h[2], -len(h[1]))):
        end = pos + len(phrase)
        if any(pos < k_end and end > k_pos for _, k_pos, k_end in kept):
            continue
        kept.append((value, pos, end))
    by_pos = {pos: (value, pos, end) for value, pos, end in kept}
    out = []
    for value, phrase, pos in hits:
        if pos in by_pos and by_pos[pos][0] == value:
            out.append((value, phrase, pos))
    return sorted(out, key=lambda h: h[2])


def _preceding_action(action_hits: list, pos: int) -> Optional[str]:
    """The action mentioned most recently before `pos`, if any."""
    prev = None
    for value, _phrase, hpos in action_hits:
        if hpos < pos:
            prev = value
        else:
            break
    return prev


# "shot from the right half-space" and friends — a finishing zone stated outright.
_SHOT_WORDS = r"shots?|efforts?|attempts?|finish(?:ing|ed|es)?|headers?|strikes?"
_CHANNEL_ALT = "|".join(
    re.escape(p) for p in sorted(
        (p for ps in CHANNEL_TERMS.values() for p in ps), key=len, reverse=True))
_SHOT_FROM_RE = re.compile(
    rf"\b(?:{_SHOT_WORDS})\b[^.]{{0,20}}?\bfrom the\s+({_CHANNEL_ALT})\b", re.IGNORECASE)


def _shot_from_channel(text: str) -> Optional[tuple]:
    m = _SHOT_FROM_RE.search(text)
    if not m:
        return None
    phrase = m.group(1).lower()
    for channel, terms in CHANNEL_TERMS.items():
        if phrase in [t.lower() for t in terms]:
            return channel, m.group(0), m.start()
    return None


def _nearest(hits: list, pos: int, window: int = 40) -> Optional[str]:
    """The zone term closest to an action phrase, within a small window.

    'interception in midfield' binds M to the interception; a band mentioned at
    the far end of the sentence does not.
    """
    best, best_d = None, window + 1
    for value, phrase, hpos in hits:
        d = abs(hpos - pos)
        if d < best_d:
            best, best_d = value, d
    return best if best_d <= window else None
