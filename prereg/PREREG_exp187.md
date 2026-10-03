# PRE-REGISTRATION — exp187: DOES A CLEANLY BUILT UP DIRECTION STILL STEER THE BUNDLE?

Written 2026-10-02 20:22 IST, before any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh at 20:21:
"sure, queue it please." Frozen at the commit carrying this text.

Seen before freezing: exp116's script, config and saved results.

## Plain summary

exp116 (the Finding 2 experiment): add the UP direction into Pythia
1.4B while it completes prompts, and its answers shift toward
happier, higher status and more, with pool depth going up too. That
UP direction was built from isolated words with no start marker, the
construction that today's audit showed picks up word position along
with the concept. This rebuilds the direction cleanly and asks
whether it still steers, and whether it steers more than directions
built from random words do.

## Design

Pythia 1.4B, layer 12, steering by adding strength x direction to the
residual at every position, strengths {−12, −8, −4, −2, 0, 2, 4, 8,
12, 16}: all exp116's. The six outcome measures, their prompts, value
grids and word lists are exp116's, copied (exp116 loads its model
when imported, so it cannot be imported): patient height, pool depth,
door width, widget count (expected value of the number completed),
status and affect (probability mass of high against low words).

Directions:
- ORIGINAL: exp116's u111_clean and ulak_clean, rebuilt exactly as
  exp116 did (bare words, no start marker).
- **CLEAN UP (PRIMARY)**: exp180's 31 UP words minus its 31 DOWN
  words, each a single token, each state taken at the word's position
  inside exp180's eight neutral sentences with a start marker, mean
  over the sentences; frequency direction (built the same clean way)
  projected out, as exp116 did; unit length.
- Comparison directions, built the same clean way: VALENCE (exp180's
  20 positive words minus their 20 opposites); BUNDLE (first minus
  second members of exp185's 32 bundle pairs, Claude's sorting);
  NO-LINK (exp185's 29 no-link pairs).
- **RANDOM-WORD NULL**: N_NULL directions, each 31 random words minus
  31 other random words from the pool of all exp180 and exp185 pair
  words, built the same clean way.

Prompts: PRIMARY with a start marker in front. exp116's own format
(no start marker) is run for the ORIGINAL and CLEAN UP directions and
reported.

Replication gate: ORIGINAL u111_clean, exp116's format, must
reproduce exp116's saved values at strengths −12, 0 and +16 (patient
height 165.900 / 168.398 / 171.567; affect −0.645 / −0.181 / +0.356)
within GATE_EV and GATE_MC. Otherwise STOP.

Effect of a direction on a measure: value at strength +8 minus value
at strength −8. (The null directions are run at −8, 0 and +8 only.)

## RULE PARAMETERS (frozen)

  N_NULL = 30   SEED_NULL = 187   GATE_EV = 0.05   GATE_MC = 0.01

## Decision rule (CLEAN UP, start-marker prompts)

A measure "moves" if CLEAN UP's effect on it is positive and above
the 95th percentile of the RANDOM-WORD NULL's effects.
- **BUNDLE_STEERS**: affect moves, AND status or widget count moves.
- **AFFECT_ONLY**: affect moves, neither of the other two does.
- **NOT_BEYOND_RANDOM**: affect does not move.
Reported: every measure for every direction across all strengths;
whether patient height rises while pool depth falls (literal
verticality); the same rule applied to the ORIGINAL directions.

## Committed predictions

- P1 replication gate passes: **85%**
- P2 affect moves under CLEAN UP: **55%**
- P3 BUNDLE_STEERS: **30%**; AFFECT_ONLY: **25%**;
  NOT_BEYOND_RANDOM: **45%**
- P4 the VALENCE direction moves affect (a sanity check on the
  method): **90%**

## What I will NOT do

- No reading a dose-response curve as specific to UP without the
  random-word null beside it.
- No changing strengths, layer, measures or word lists after seeing
  numbers.

## Deviations and stubs

Flagged at freeze: exp116's scoring functions are copied, not
imported. The gate checks the copy against exp116's saved numbers.

## RESULT + GRADES (graded 23:41 IST, 2026-10-02)

Raw output: exp187_output.txt. Numbers: exp187_results.json; every
cell in exp187_cells.json. Mac GPU. No deviations beyond the flagged
copy of exp116's scoring, which the gate validated: in exp116's own
format the copy reproduces exp116's saved patient height (165.900 /
168.398 / 171.567) and affect (−0.645 / −0.181 / +0.356) at strengths
−12 / 0 / +16.

Direction similarities: cos(CLEAN UP, exp116's u111) +0.525,
cos(CLEAN UP, exp116's ulak) +0.511, cos(CLEAN UP, CLEAN VALENCE)
+0.165, cos(CLEAN UP, CLEAN BUNDLE) +0.406.

### Effect = value at strength +8 minus value at −8 (start-marker prompts)

| direction | patient height (cm) | pool depth (cm) | door width (cm) | widget count | status | affect |
|---|---|---|---|---|---|---|
| random-word null, 5th pct | −2.82 | −14.34 | −4.96 | −2.64 | −0.62 | −0.80 |
| random-word null, 95th pct | +2.66 | +16.32 | +3.77 | +1.29 | +0.61 | +0.70 |
| **CLEAN UP** | **+2.67*** | +8.96 | +3.45 | +0.52 | +0.47 | **+1.29*** |
| exp116's u111 | −0.71 | +2.32 | −0.41 | +1.29 | −0.10 | +0.79* |
| exp116's ulak | +0.15 | −1.54 | +0.69 | +0.78 | +0.34 | +1.10* |
| CLEAN VALENCE | −0.98 | +4.50 | +3.93* | +0.45 | +0.50 | +3.01* |
| CLEAN BUNDLE (more / big / strong ...) | +3.70* | +12.35 | +4.00* | +3.82* | +0.84* | +1.42* |
| CLEAN NO-LINK | +1.25 | +3.39 | +3.64 | −0.80 | −0.23 | +0.33 |

\* positive and above the null's 95th percentile.

**VERDICT (CLEAN UP): AFFECT_ONLY.** Affect moves beyond the random
directions; status and widget count do not.

### What this says
- A purely spatial UP direction, built cleanly, makes Pythia 1.4B's
  completions happier, by more than 95% of directions built from
  random words. Its overlap with the valence direction is small
  (cos +0.165), and the valence direction itself moves affect about
  2.3 times as far, so this is not the valence direction in disguise.
  HAPPY IS UP survives as a behavioural result with a random-word
  null. Patient height also just clears the null.
- The "bundle" half of Finding 2 does not: status and quantity sit
  inside what random directions do. exp116's original directions,
  given start-marker prompts, also move affect only.
- Random-word directions move the numeric measures a great deal
  (pool depth by ±15 cm), so the old pattern "pool depth goes up too"
  is not evidence of anything.
- The CLEAN BUNDLE direction (first minus second members of exp185's
  bundle pairs: more, big, strong, alive ...) moves every measure
  beyond the null: a "more of everything" direction. That is the
  pattern exp116 attributed to UP; it belongs to the MORE words.
- The NO-LINK direction moves nothing beyond random, a useful
  negative control.

### Grades
- P1 gate passes (85%): HIT.
- P2 affect moves under CLEAN UP (55%): HIT.
- P3 BUNDLE_STEERS (30%): did not occur; AFFECT_ONLY (25%): occurred.
- P4 VALENCE moves affect (90%): HIT.
