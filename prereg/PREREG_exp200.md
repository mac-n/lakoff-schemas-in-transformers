# PRE-REGISTRATION — exp200: IS "GOAL" A ROLE THE MODEL GIVES TO WHAT AN INSTRUCTION ASKS FOR, EVEN WITH NO SOURCE IN THE SENTENCE?

Written 02:54 IST, 03 Oct 2026, before any data for this experiment, by Claude (Fable
5.1). Niamh at 02:54: "can we somehow see if GOAL is not a property
of only from.. to? sentences? like would there be a way to extract it
from other instructions?" Frozen at the commit carrying this text.
exp199's result was NOT seen (it is still running).

## Plain summary

Every test so far gives the probe a pair: a start and an end in one
sentence. Here the sentence has only one of them. "Write a haiku
about rain" names an asked-for thing and no starting thing. If the
model marks the haiku as a GOAL anyway, goal-ness is a role of the
thing asked for, not a by-product of a from/to frame. "Read this
draft carefully" is the mirror: a thing to start from, nothing asked
for.

## Design

Llama-3.2-1B and Llama-3.2-1B-Instruct, CPU, layers 3, 6, 8, 10, 13.
Probe as exp199: ridge on exp198's cue-free literal journeys minus
the into / out-of sentences (29 sentences), source = 0, goal = 1.
Score of a token = its standardised projection on the probe (positive
= goal side).

Sets (32 each):
- **GOAL-ONLY**: instructions naming only an output ("Write a haiku
  about rain", "Draw me a map of the village", "Compose a lullaby").
- **SOURCE-ONLY**: instructions naming only an input ("Read this
  draft carefully", "Study the table", "Listen to the recording").
- **NO-TASK**: each of those 64 nouns in a sentence with no task
  ("The haiku was on the table"), scored at the same noun.
Nouns are disjoint from the training set and from exp199's.

Measure: for each noun, shift = score(instruction) − score(no-task).
Chance: N_SHUF label-shuffled probes give the distribution of the
mean shift.

## RULE PARAMETERS (frozen)

  N_SHUF = 300   SEED = 200   PRIMARY_LAYER = 6

## Decision rule (per model, layer 6)

- **GOAL_IS_A_ROLE**: the mean GOAL-ONLY shift is positive and above
  the shuffle 95th, AND the mean SOURCE-ONLY shift is below the
  shuffle 5th (the two kinds of instruction push their nouns opposite
  ways).
- **GOAL_ONLY_HALF**: GOAL-ONLY passes, SOURCE-ONLY does not.
- **SOURCE_ONLY_HALF**: the reverse.
- **NOT_A_ROLE**: neither passes.
Reported: all layers; the fraction of nouns shifted the predicted
way; instruct vs base.

## Committed predictions

- Base GOAL_IS_A_ROLE **30%**, GOAL_ONLY_HALF **25%**, NOT_A_ROLE
  **35%**, SOURCE_ONLY_HALF **10%**
- Instruct GOAL_IS_A_ROLE **40%**
- Instruct GOAL-ONLY shift larger than base's: **60%**

## What I will NOT do
- No changing the layer; no dropping instructions after the numbers.

## RESULT + GRADES (graded 03:28 IST, 03 Oct 2026)

Outputs: exp200_output_Llama-3.2-1B.txt, exp200_output_Llama-3.2-1B-Instruct.txt.
Shift = probe score in the instruction minus score of the same noun
in "The X was on the table yesterday"; positive = toward the goal end.

| | layer | goal-only shift (frac > 0) | source-only shift (frac < 0) | shuffle 5–95% |
|---|---|---|---|---|
| base | 6 (primary) | −0.08 (0.28) | −0.32 (0.94) | −0.40 / +0.48 |
| base | 10 | +0.21 (0.81) | −0.08 (0.72) | −0.39 / +0.38 |
| base | 13 | +0.44 (1.00) | +0.19 (0.16) | −0.40 / +0.42 |
| instruct | 6 (primary) | +0.02 (0.59) | −0.20 (0.91) | −0.32 / +0.38 |
| instruct | 10 | +0.37 (1.00) | +0.08 (0.25) | −0.30 / +0.28 |
| instruct | 13 | +0.40 (1.00) | +0.15 (0.16) | −0.32 / +0.33 |

**NOT_A_ROLE in both models at layer 6.** Nothing clears the shuffle
range there. The shuffle range is wide (about ±0.4) because the probe
has 58 training points in Llama's 2048 dimensions; this test is
under-powered as built.

What the deeper layers show, post-hoc: by layer 10–13, EVERY noun in
a goal-only instruction sits further toward the goal end than the
same word at rest (100% of 32 in both models; instruct reaches it by
layer 10, base by 13). But the source-only nouns move the same way
(+0.15 to +0.19, 84% of them toward goal, not away). So what the
deeper layers carry is "this word is in an instruction", which lies
on the goal side of the journey probe, not a source/goal split
between kinds of instruction. Niamh's question (02:54) is therefore
not answered yes by this run: being asked for is not shown to be a
distinct role; being in an imperative is.

### Grades
- Base GOAL_IS_A_ROLE 30% / GOAL_ONLY_HALF 25% / NOT_A_ROLE 35% /
  SOURCE_ONLY_HALF 10%: NOT_A_ROLE occurred.
- Instruct GOAL_IS_A_ROLE 40%: did not occur.
- Instruct goal-only shift larger than base's (60%): occurred at
  every layer (e.g. +0.37 vs +0.21 at layer 10), though neither
  clears the rule at 6.
