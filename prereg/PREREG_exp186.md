# PRE-REGISTRATION — exp186: IS THE SURVIVING CAUSAL EFFECT ABOUT "UN-" WORDS?

Written 2026-10-02 19:55 IST, before any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh at 19:52:
"we should tie off the BALANCE loose end". Frozen at the commit
carrying this text.

Seen before freezing: exp179's results; tokenizer-only counts of the
word pairs below.

## Plain summary

exp179: at layer 3 of GPT-2, making attention more diffuse moves
token states along the original BALANCE axis more than along any of
300 fake axes with the same token split. But the BALANCE lists
contain four exact "un-" pairs (balanced / unbalanced, equal /
unequal, even / uneven, stable / unstable). The fake axes matched how
many words were split, not that the split words were negations of
the words on the other side. This asks whether negation does it.

## Design

exp179's procedure, imported: GPT-2-medium, attention temperature at
one layer, tau grid unchanged, exp166's prompts, slope of mean
projection on mean achieved entropy; layer 3 PRIMARY, layer 8
reported; bare-word protocol; axes norm-orthogonalised.

Axes:
- the real BALANCE axis (gate: must reproduce −0.0083 at layer 3 and
  −0.0018 at layer 8 within SLOPE_TOL);
- **UN-AXES**: N_RAND axes, each from 15 word / un-word pairs drawn at
  random from a pool of 38 on concepts unrelated to balance (happy /
  unhappy, able / unable, fair / unfair, kind / unkind, clear /
  unclear, certain / uncertain, usual / unusual, likely / unlikely,
  safe / unsafe, sure / unsure, true / untrue, wise / unwise, clean /
  unclean, common / uncommon, pleasant / unpleasant, real / unreal,
  easy / uneasy, seen / unseen, done / undone, used / unused, paid /
  unpaid, told / untold, changed / unchanged, finished / unfinished,
  married / unmarried, expected / unexpected, natural / unnatural,
  official / unofficial, popular / unpopular, healthy / unhealthy,
  friendly / unfriendly, aware / unaware, necessary / unnecessary,
  important / unimportant, reasonable / unreasonable, available /
  unavailable, acceptable / unacceptable, employed / unemployed):
  positive pole the 15 plain words, negative pole their un- forms;
- **BALANCE_NO_UN**: the real lists with the four un- pairs' eight
  words removed (11 against 11);
- **MATCHED_NO_UN**: N_RAND fake axes from random pool words with
  BALANCE_NO_UN's token split, as exp179's MATCHED.

## RULE PARAMETERS (frozen)

  N_RAND = 300   SEED_RAND = 186   SLOPE_TOL = 0.0005

## Decision rule (layer 3)

- **NEGATION_SUFFICIENT**: the real BALANCE slope is NOT below the 5th
  percentile of the UN-AXES slopes. Axes built from un- pairs on
  unrelated concepts move the same way, as much.
- **SURVIVES_WITHOUT_NEGATION**: not NEGATION_SUFFICIENT, and the
  BALANCE_NO_UN slope is below the 5th percentile of MATCHED_NO_UN.
- **UNRESOLVED**: anything else.

## Committed predictions

- P1 gate passes: **95%**
- P2 NEGATION_SUFFICIENT **45%**; SURVIVES_WITHOUT_NEGATION **20%**;
  UNRESOLVED **35%**

## Note on the other candidate "reversed result"

Llama's sign flip in exp177 (+0.32 / +0.43 / +0.21 with nine
imbalance words) was already tested in exp178 Part 2: with the longer
list inside sentences it reads +0.06 to +0.13, inside the range of
random clean axes (about ±0.3). It is not retested here.

## RESULT + GRADES (graded 20:06 IST, 2026-10-02)

Raw output: exp186_output.txt. Numbers: exp186_results.json. CPU. No
deviations. Replication gate PASS at both layers (−0.0083, −0.0018).

### Layer 3 (PRIMARY)
- UN-AXES (300 axes, each from 15 word / un-word pairs on concepts
  unrelated to balance): median slope −0.0061, 5th percentile
  −0.0081, 95th −0.0041. **Every one of the 300 is negative.**
- The real BALANCE slope (−0.0083) sits at the 4th percentile of
  them: just past the 5th-percentile line.
- BALANCE with its four un- pairs removed (11 against 11): slope
  −0.0047, at the 9th percentile of its own token-matched fakes
  (5th percentile −0.0057). Not distinguishable from them.

**VERDICT: UNRESOLVED**, by the letter of the rule: the real axis is
below the 5th percentile of the un- axes by a hair (so not
NEGATION_SUFFICIENT), and what is left without the un- pairs does
not clear its matched control (so not SURVIVES_WITHOUT_NEGATION).

### Layer 8 (reported)
Real BALANCE at the 52nd percentile of UN-AXES (median −0.0018, all
300 negative). There the rule would read NEGATION_SUFFICIENT.

### What this says
- Axes built from plain words against their "un-" forms, on concepts
  with nothing to do with balance, all move with attention
  temperature the way the BALANCE axis does, at about three quarters
  of its size at layer 3 and the same size at layer 8.
- Take the four un- pairs out of BALANCE and its slope falls by
  almost half and stops standing out from token-matched fakes.
- So the causal effect that survived exp179 is mostly a property of
  "un-" negation pairs. Whether the small margin by which the full
  BALANCE axis exceeds typical un- axes at layer 3 is anything is not
  settled by this test, and it is small.
- Explains exp179: its fake axes matched how many words were split,
  not that the split words were un- forms of the words opposite.

### Grades
- P1 gate passes (95%): HIT.
- P2: UNRESOLVED occurred (given 35%; NEGATION_SUFFICIENT was the
  favourite at 45% and missed by one percentile).
