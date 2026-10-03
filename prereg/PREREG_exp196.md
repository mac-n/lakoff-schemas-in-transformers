# PRE-REGISTRATION — exp196: CALIBRATING THE MEASURE WITH SPARSE-AUTOENCODER FEATURES

Written 01:16 IST, 03 Oct 2026, before any code or model run, by Claude (session
labelled Fable 5.1). Niamh at 01:15: "is it worth queueing the SAE
experiment?" Frozen at the commit carrying this text.

Seen before freezing: exp193's result; the cached SAE's files (an SAE
on Pythia 410M's layer-4 MLP output: 65,536 latents, k = 32, decoder
rows of unit length in the residual space).

## Plain summary

If "how much the later layers disagree after a push" measures whether
the model responds to a direction, then directions the model
demonstrably uses should disrupt agreement, and directions it does
not use should not. The SAE gives both: its decoder rows are the
directions the layer-4 MLP writes, and on a given text some latents
fire and most never do.

## Design

Pythia 410M, layer 4, exp193's sentences, strength and measures;
exp193's control and schema cells are reused unchanged (same model,
site, strength, seed), so the new directions are compared on
identical footing. New directions, pushed along the SAE decoder row:
- **USED (30)**: the 30 latents that fire on the most tokens of the
  120 sentences (encoder applied to the layer-4 MLP output).
- **UNUSED (30)**: 30 latents, drawn at random with SEED, that never
  fire on any token of the 120 sentences.
Both signs; drop = unsteered agreement minus the sign-averaged
steered agreement.

## RULE PARAMETERS (frozen)

  N_SAE = 30   SEED = 196

## Decision rule

The concentrated-control range is exp193's SINGLE + PAIR 5th–95th
percentiles.
- **MEASURE_DETECTS_USE**: the median USED drop is above the control
  95th percentile AND the median UNUSED drop is inside the control
  range.
- **USE_NOT_SPECIAL**: the median USED drop is inside the control
  range.
- **UNUSED_ALSO**: both USED and UNUSED medians are above the control
  95th (any SAE direction disrupts; the measure reads "on-manifold",
  not "used").
Reported: where UP-DOWN's drop (exp193, +0.0377) sits in the USED
distribution; the USED and UNUSED output KLs.

## Committed predictions

- MEASURE_DETECTS_USE **30%**; USE_NOT_SPECIAL **35%**; UNUSED_ALSO
  **35%**
- UP-DOWN above the USED median: **40%**

## RESULT + GRADES (graded 02:49 IST, 03 Oct 2026)

Output: exp196_output.txt; exp196_results.json; cells exp196_cells.npz.
16,718 of the 65,536 latents fire on at least one token of the 120
sentences. USED = the 30 firing most (on 818 down to 92 tokens);
UNUSED = 30 that never fire.

| push along | drop in later-layer agreement, median [5%, 95%] | output KL |
|---|---|---|
| USED SAE features (30) | +0.0009 [−0.0136, +0.0173] | 1.26 |
| UNUSED SAE features (30) | +0.0039 [−0.0005, +0.0187] | 1.37 |
| exp193 concentrated controls (60) | [−0.0091, +0.0115] | 2.3 |
| exp193 UP-DOWN | **+0.0377** (above every USED feature) | 2.79 |

**USE_NOT_SPECIAL.** Pushing along a feature the layer-4 MLP uses on
these very sentences disturbs the later layers' agreement no more
than pushing along one it never uses, and neither more than a
single-word push. So "how much the later layers disagree after a
push" does NOT read "is this a direction the model uses" in the SAE
sense. UP-DOWN still sits above all 60 SAE directions and all 120
other controls. Whatever UP-DOWN does to Pythia at layer 4, it is not
what a typical used feature does; the measure has found one odd
direction, not a general property.

### Grades
MEASURE_DETECTS_USE 30% / USE_NOT_SPECIAL 35% / UNUSED_ALSO 35%:
USE_NOT_SPECIAL occurred. UP above the USED median (40%): occurred,
above the USED maximum.
