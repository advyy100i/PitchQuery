# Planner evaluation — English to structured query, no LLM

`core/planner.py` translates an English description into the same `Filters` + `sequence_hint` a human would write by hand. It is a rule parser: deterministic, free, no API key, and every filter is traceable to the phrase that produced it.

**Parse cost: 1.13 ms mean, 12.89 ms worst case.** No network, no tokens, no key.

**Filter agreement with the hand-written queries: 24/30 exact.**

Data source: StatsBomb.

## Retrieval quality

Same rubrics and the same 25 discriminating queries as `docs/retrieval_eval.md` (the 5 filter-dominated queries are excluded from the headline for the reason given there).

| query source | P@5 | P@10 | MRR |
|---|--:|--:|--:|
| hand-written structured | 0.608 | 0.600 | 0.751 |
| **parsed from English** | **0.680** | 0.664 | 0.827 |

All 30 queries: hand P@5 0.667, parsed P@5 0.733.

### One number in that table is not comparable

Where the two sides produce *different* filters, they are not ranking the same candidate set, and the rubric cannot always tell which choice was right. `q07` is the clear case: the parser reads "from the goalkeeper" as `From Keeper` where the hand-written query chose `From Goal Kick`, and q07's rubric only tests whether the possession reaches the final third — it never checks the play pattern, because the filter was supposed to guarantee it. The parser scores 1.0 against 0.2 on a distinction the rubric is blind to.

Restricted to the 19 discriminating queries where **both sides produced identical filters**, so only the ranking differs: hand P@5 0.632, parsed P@5 0.611. That is the honest like-for-like figure, and it is the one to quote.

## Per query

| id | query | filters agree | hand P@5 | parsed P@5 |
|---|---|:-:|--:|--:|
| q01 | right-wing cross into the box that ends in a shot | yes | 1.0 | 1.0 |
| q02 | left-wing cross into the box that ends in a shot | yes | 1.0 | 1.0 |
| q03 | switch of play from one flank to the other | yes | 1.0 | 1.0 |
| q04 | fast counter-attack ending in a shot | yes | 1.0 | 1.0 |
| q05 | corner headed goalward | yes | 1.0 | 1.0 |
| q06 | through ball played in behind the defence | yes | 0.2 | 0.6 |
| q07 | build-up from the goalkeeper that reaches the final third | no | 0.2 | 1.0 |
| q08 | long possession that ends in a shot | yes | 0.8 | 1.0 |
| q09 | high turnover in the final third leading immediately to a shot | no | 0.8 | 1.0 |
| q10 | dribble into the box | yes | 0.4 | 0.0 |
| q11 | Barcelona working the ball into the left half-space | no | 0.6 | 1.0 |
| q12 | Paris Saint-Germain counter-attack | yes | 1.0 | 1.0 |
| q13 | Bayer Leverkusen pressing high and scoring | yes | 0.4 | 0.4 |
| q14 | shot from outside the box after a lay-off | yes | 0.2 | 0.4 |
| q15 | overlapping run down the right ending in a cutback | no | 0.0 | 0.6 |
| q16 | throw-in routine that creates a shot | yes | 0.8 | 1.0 |
| q17 | free kick played into the box | yes | 1.0 | 1.0 |
| q18 | possession recycled backwards then switched and attacked again | yes | 0.2 | 0.0 |
| q19 | central combination play through the middle third | yes | 0.8 | 0.6 |
| q20 | goal scored from inside the six-yard box | yes | 1.0 | 0.6 |
| q21 | pressure-resistant build-up under heavy pressing | no | 0.8 | 0.8 |
| q22 | direct long ball from defence into the final third | no | 0.8 | 1.0 |
| q23 | England attacking the left channel in the final third | yes | 0.6 | 0.6 |
| q24 | Spain patient possession ending in a shot | yes | 1.0 | 1.0 |
| q25 | shot from a rebound after a blocked attempt | yes | 1.0 | 1.0 |
| q26 | attack down the right that ends with a shot from the right half-space | yes | 0.4 | 0.4 |
| q27 | interception in midfield turned into an immediate attack | yes | 0.6 | 0.4 |
| q28 | cutback from the byline | yes | 0.2 | 0.4 |
| q29 | quick one-two through the inside-left channel | yes | 0.2 | 0.2 |
| q30 | high-xG chance created from open play | yes | 1.0 | 1.0 |

## Held-out paraphrases — does it generalise?

The rules above were written while looking at failures on the 30 `text` fields, so that headline is an **in-sample** number. These paraphrases restate the same intents in deliberately different words and are judged by the original rubrics.

**23 paraphrases: P@5 0.748, versus 0.730 for the original wording of the same queries (+0.017).**

| id | paraphrase | original P@5 | paraphrase P@5 |
|---|---|--:|--:|
| q01 | delivery from the right flank into the penalty area leading to an attempt | 1.0 | 1.0 |
| q02 | ball whipped in from the left touchline into the penalty box producing an effort | 1.0 | 1.0 |
| q03 | diagonal ball changed flanks to the other side | 1.0 | 1.0 |
| q04 | rapid transition finishing with an effort | 1.0 | 1.0 |
| q05 | corner kick met with a header | 1.0 | 1.0 |
| q06 | slipped in behind the defence | 0.6 | 0.6 |
| q07 | playing out from the back that reaches the attacking third | 1.0 | 1.0 |
| q09 | won the ball in the final third and struck immediately | 1.0 | 1.0 |
| q10 | took on his man and drove into the penalty area | 0.0 | 0.0 |
| q11 | Barcelona working the ball into the inside left channel | 1.0 | 1.0 |
| q13 | Bayer Leverkusen counter-press that ends in a goal | 0.4 | 0.4 |
| q14 | effort from long range after a knock-down | 0.4 | 0.4 |
| q16 | throw in that produces an attempt | 1.0 | 1.0 |
| q17 | set-piece delivered into the penalty area | 1.0 | 1.0 |
| q19 | central combination played through midfield | 0.6 | 0.6 |
| q21 | playing out from the back under the press | 0.8 | 0.8 |
| q22 | route one pass from the back line into the attacking third | 1.0 | 1.0 |
| q26 | attack down the right wing finishing from the inside right | 0.4 | 0.4 |
| q27 | cut out in midfield and broke forward | 0.4 | 0.4 |
| q28 | pulled it back from the right byline | 0.4 | 0.4 |
| q29 | give-and-go through the inside-left channel | 0.2 | 0.2 |
| q30 | clear-cut chance created from open play | 1.0 | 1.0 |
| q20 | goal finished off from inside the six yard box | 0.6 | 1.0 |

### Known limits (included deliberately)

- `q12` — "PSG on the break" scored 1.0. Ignored: *PSG*

A rule parser has no alias table, so a nickname the database has never seen is simply not a team. An LLM would resolve it. This is the concrete cost of the trade, kept in the report rather than dropped from it.

## Known limitations

**Short-possession bias.** A thin hint retrieves thin possessions. Ask for "Barcelona working the ball into the left half-space" and the top hits are three-token, one-second fragments — which the rubric scores as relevant (they do end in F-LI) while a human would call them worthless. The cause is TF-IDF cosine similarity, not the parser: a 3-token possession matching the hint exactly outscores a 20-token one containing the same tokens, because the longer vector is diluted. Padding the hint differently was tried and measured worse (P@5 0.680 -> 0.592), so it was reverted. The real fix is length normalisation in `core/retrieval.py`, which is a ranker change, not a planner change.

**Vocabulary coverage is the whole ceiling.** The parser understands the terms in `core/planner.py` and nothing else. It has no alias table and no paraphrase ability beyond the synonym sets — see the PSG case above. An LLM would generalise where this does not; that is the trade, and it is why the `ignored` field is surfaced in the UI rather than hidden, so a user can see when their words were not understood.

**The rubric cannot judge every disagreement.** Where the two filter sets differ, the programmatic rubric sometimes has no way to tell which was right (see q07 above). The like-for-like figure exists because of this.

## What the parser produced

**q01** — right-wing cross into the box that ends in a shot

- filters: `{'ended_in_shot': True}`
- hint: `RECV@F-R CARRY@F-R CROSS@F-R> RECV@F-C SHOT@F-C`
- words ignored: *that*

**q02** — left-wing cross into the box that ends in a shot

- filters: `{'ended_in_shot': True}`
- hint: `RECV@F-L CARRY@F-L CROSS@F-L> RECV@F-C SHOT@F-C`
- words ignored: *that*

**q03** — switch of play from one flank to the other

- filters: `none`
- hint: `RECV@M-C CARRY@M-C SWITCH@M-C+`
- words ignored: *from one flank to the other*

**q04** — fast counter-attack ending in a shot

- filters: `{'play_pattern': 'From Counter', 'ended_in_shot': True}`
- hint: `RECV@F-C CARRY@F-C SHOT@F-C`
- words ignored: *fast*

**q05** — corner headed goalward

- filters: `{'play_pattern': 'From Corner', 'ended_in_shot': True}`
- hint: `SETP@F-R> RECV@F-C SHOT@F-C`

**q06** — through ball played in behind the defence

- filters: `none`
- hint: `RECV@M-C CARRY@M-C THROUGH@M-C+ THROUGH@M-C+`
- words ignored: *the defence*

**q07** — build-up from the goalkeeper that reaches the final third

- filters: `{'play_pattern': 'From Keeper', 'start_band': 'D', 'end_band': 'F'}`
- hint: `RECV@D-C CARRY@D-C PASS@D-C RECV@D-LI PASS@D-LI PASS@D-LI+ RECV@M-LI`
- words ignored: *that reaches the*

**q08** — long possession that ends in a shot

- filters: `{'ended_in_shot': True, 'min_events': 25}`
- hint: `RECV@F-C CARRY@F-C SHOT@F-C`
- words ignored: *that*

**q09** — high turnover in the final third leading immediately to a shot

- filters: `{'start_band': 'F', 'ended_in_shot': True}`
- hint: `RECV@F-C CARRY@F-C RECOV@F-C SHOT@F-C`
- words ignored: *high in the leading immediately to a*

**q10** — dribble into the box

- filters: `none`
- hint: `RECV@F-C CARRY@F-C DRIB@F-C CARRY@F-C> RECV@F-C`

**q11** — Barcelona working the ball into the left half-space

- filters: `{'team': 'Barcelona'}`
- hint: `RECV@F-LI CARRY@F-LI CARRY@F-LI`
- words ignored: *working the ball into the*

**q12** — Paris Saint-Germain counter-attack

- filters: `{'team': 'Paris Saint-Germain', 'play_pattern': 'From Counter'}`
- hint: `RECV@F-C CARRY@F-C CARRY@F-C`

**q13** — Bayer Leverkusen pressing high and scoring

- filters: `{'team': 'Bayer Leverkusen', 'ended_in_goal': True}`
- hint: `RECV@F-C CARRY@F-C RECOV@F-C SHOT@F-C`
- words ignored: *and*

**q14** — shot from outside the box after a lay-off

- filters: `{'ended_in_shot': True}`
- hint: `SHOT@M-C PASS@M-C+ RECV@F-LI PASS@F-LI PASS@F-LI`
- words ignored: *from after a*

**q15** — overlapping run down the right ending in a cutback

- filters: `{'ended_in_shot': True}`
- hint: `CARRY@M-R PASS@M-R+ RECV@F-R PASS@F-R> RECV@F-C SHOT@F-C`
- words ignored: *ending in a*

**q16** — throw-in routine that creates a shot

- filters: `{'play_pattern': 'From Throw In', 'ended_in_shot': True}`
- hint: `SETP@F-R PASS@F-R RECV@F-C SHOT@F-C`
- words ignored: *routine*

**q17** — free kick played into the box

- filters: `{'play_pattern': 'From Free Kick'}`
- hint: `SETP@F-LI> RECV@F-C`
- words ignored: *played*

**q18** — possession recycled backwards then switched and attacked again

- filters: `none`
- hint: `SWITCH@M-C+ CARRY@M-C PASS@M-C+ RECV@F-C`
- words ignored: *possession recycled backwards then and again*

**q19** — central combination play through the middle third

- filters: `none`
- hint: `RECV@F-C CARRY@F-C PASS@F-C PASS@F-C`
- words ignored: *play*

**q20** — goal scored from inside the six-yard box

- filters: `{'ended_in_goal': True}`
- hint: `RECV@F-C CARRY@F-C SHOT@F-C`
- words ignored: *scored from inside the*

**q21** — pressure-resistant build-up under heavy pressing

- filters: `{'start_band': 'D'}`
- hint: `RECV@D-C^ CARRY@D-C^ PASS@D-C^ RECV@D-LI^ PASS@D-LI^ PASS@D-LI+^ RECV@M-LI^`
- words ignored: *under heavy pressing*

**q22** — direct long ball from defence into the final third

- filters: `{'start_band': 'D', 'end_band': 'F'}`
- hint: `PASS@D-C+ RECV@F-C RECV@F-C CARRY@F-C CARRY@F-C`
- words ignored: *long ball into the*

**q23** — England attacking the left channel in the final third

- filters: `{'team': 'England', 'end_band': 'F'}`
- hint: `RECOV@M-LI CARRY@M-LI PASS@M-LI+ RECV@F-LI`
- words ignored: *the in the*

**q24** — Spain patient possession ending in a shot

- filters: `{'team': 'Spain', 'ended_in_shot': True, 'min_events': 20}`
- hint: `RECV@F-C CARRY@F-C SHOT@F-C`
- words ignored: *possession*

**q25** — shot from a rebound after a blocked attempt

- filters: `{'ended_in_shot': True}`
- hint: `SHOT@F-C RECOV@F-C SHOT@F-C`
- words ignored: *from a after a blocked*

**q26** — attack down the right that ends with a shot from the right half-space

- filters: `{'end_zone': 'F-RI', 'ended_in_shot': True}`
- hint: `RECV@F-RI CARRY@F-RI SHOT@F-RI`
- words ignored: *attack that*

**q27** — interception in midfield turned into an immediate attack

- filters: `none`
- hint: `INT@M-C CARRY@M-C PASS@M-C+ RECV@F-C`
- words ignored: *in turned into an*

**q28** — cutback from the byline

- filters: `{'ended_in_shot': True}`
- hint: `RECV@F-R CARRY@F-R PASS@F-R> RECV@F-C SHOT@F-C`
- words ignored: *from the byline*

**q29** — quick one-two through the inside-left channel

- filters: `none`
- hint: `RECV@F-LI CARRY@F-LI PASS@F-LI PASS@F-LI`
- words ignored: *quick through the*

**q30** — high-xG chance created from open play

- filters: `{'play_pattern': 'Regular Play', 'min_xg': 0.3}`
- hint: `RECV@F-C CARRY@F-C CARRY@F-C`
- words ignored: *chance created from*

