# PRE-REGISTRATION — exp194: DOES CLEAN UP STEERING MOVE AFFECT IN A SECOND MODEL?

Written 00:53 IST, 03 Oct 2026, before any code or model run for this experiment, by
Claude (session labelled Fable 5.1). Niamh at 00:52: "can you queue
it for after exp193?" Frozen at the commit carrying this text.

## Plain summary

exp187: in Pythia 1.4B, steering with a cleanly built spatial UP
direction makes completions happier, beyond directions built from
random words (affect only; status and quantity do not clear the
null). This repeats that test in GPT-2-medium, at three layers,
because the layer was never swept.

## Design

GPT-2-medium (24 layers), start-marker prompts. exp116's six outcome
measures and scoring, as copied in exp187. Steering by adding
strength x direction to the residual at every position.

Layers: 12 (PRIMARY), 6 and 18 (reported). Effect of a direction on a
measure = value at strength +8 minus value at −8. At layer 12 the
CLEAN UP direction is also run across exp116's full strength grid
(−12 .. +16) so the dose curve can be seen.

Directions, built the clean way at the steering layer (single-token
words inside exp180's eight neutral sentences; frequency direction
projected out; unit length), the same word lists as exp187:
- **CLEAN UP**: exp180's 31 UP words minus their 31 DOWN opposites.
- CLEAN VALENCE, CLEAN BUNDLE, CLEAN NO-LINK: as exp187.
- **RANDOM-WORD NULL**: N_NULL directions per layer, 31 random words
  minus 31 others from the exp180 + exp185 pool.

Sanity gate at each layer: CLEAN VALENCE must move affect above the
null's 95th percentile. If it does not, the layer is reported as
INVALID (steering at that layer does not reach affect at all).

## RULE PARAMETERS (frozen)

  N_NULL = 20   SEED_NULL = 194

## Decision rule (layer 12)

A measure moves if CLEAN UP's effect on it is positive and above the
null's 95th percentile.
- **REPLICATES**: affect moves at layer 12.
- **DOES_NOT_REPLICATE**: affect does not move at layer 12 while the
  sanity gate passes there.
- **INVALID**: the sanity gate fails at layer 12.
Reported: the same at layers 6 and 18; status, quantity and the
numeric measures; the dose curve.

## Committed predictions

- P1 REPLICATES: **50%**; DOES_NOT_REPLICATE: **40%**; INVALID: **10%**
- P2 affect moves at all three layers: **25%**
- P3 status or quantity moves beyond the null at layer 12: **15%**

## What I will NOT do

- No picking the layer after seeing numbers: layer 12 was named
  before the run.
- No reading "affect moves" at only one of three layers as more than
  what it is.
