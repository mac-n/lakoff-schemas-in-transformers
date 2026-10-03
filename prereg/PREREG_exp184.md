# PRE-REGISTRATION — exp184: WHEN DOES THE OPPOSITE-PAIR ASYMMETRY APPEAR IN TRAINING?

Written 2026-10-02 19:55 IST, before any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh asked for
it at 19:52. Frozen at the commit carrying this text.

## Plain summary

exp180: the first member of an opposite pair (up over down, big over
small) leaves a bigger hole in the rest of a sentence. Is that there
in an untrained model (then it is the machinery), or does it appear
as the model learns from text (then it was learned, and we see when)?

## Design

Pythia 410M at the six checkpoints saved on this machine: steps 0,
512, 4000, 16000, 64000 and the final model (143000). exp180's
stimuli, frames, measure and analysis, unchanged and imported. Each
checkpoint uses its own layer averages.

Gate: the final checkpoint must reproduce exp180's Pythia result
(vertical mean difference +0.203) within GATE_TOL.

Floor: if a checkpoint's mean hole (E_reach, all words) is below
NOISE_FLOOR, that checkpoint is reported as below the floor and not
tested.

## RULE PARAMETERS (frozen)

  ALPHA = 0.05   N_PERM = 10000   GATE_TOL = 0.01   NOISE_FLOOR = 0.0001

## Decision rule

A checkpoint "shows it" if the vertical mean difference is above zero
with sign-flip p < ALPHA (exp180's T_vert).
- **AT_INIT**: step 0 shows it. The asymmetry is in the machinery
  (tokens, initial embeddings), not learned.
- **LEARNED**: step 0 does not show it and the final model does. The
  first checkpoint that shows it is reported.
- **UNCLEAR**: anything else (including step 0 below the floor).
The same is reported for the polar pairs.

## Committed predictions

- P1 gate passes: **95%**
- P2 LEARNED: **80%**; AT_INIT: **10%**
- P3 given LEARNED, it first shows at step 4000 or earlier: **50%**

## What I will NOT do

- No reading the size of the effect across checkpoints as a learning
  curve without the hole's overall level beside it.

## CANCELLED (20:37 IST, 2026-10-02) at Niamh's request, before it ran

Niamh, 20:35: "the training checkpoints ... if its about this i dont
hink it's interesting enough to research further." It was about the
opposite-pair effect, since explained as "X and ___". Never started;
no data exists.
