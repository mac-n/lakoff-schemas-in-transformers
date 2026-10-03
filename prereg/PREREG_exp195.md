# PRE-REGISTRATION — exp195: exp193 IN A SECOND MODEL AND A SECOND LAYER

Written 01:16 IST, 03 Oct 2026, before any code or model run, by Claude (session
labelled Fable 5.1). Niamh at 01:15: "did we test 193 on the other
models/other layers?" Frozen at the commit carrying this text.

Seen before freezing: exp193's result (Pythia layer 4: UP_STANDS_OUT,
MEANING_MATTERS narrowly, drop–KL +0.11) and exp190's cells (in GPT-2,
random-word pushes lower later-layer agreement as much as UP does).

## Design

exp193's design, unchanged in every rule, run in two new cells:
GPT-2-medium at layer 4, and Pythia 410M at layer 8. Directions are
built at the steering layer from the words' states inside exp180's
eight neutral sentences (start token; anisotropy and frequency
directions projected out). The schema lists are exp175's, with any
word that is not a single token in the model's tokenizer dropped (a
pole with fewer than 5 words makes that axis non-binding). Nine real
axes, and 30 each of SINGLE, PAIR, MUSH and GAUSS controls. Strength
1.0 x the site's mean norm, both signs. Measures as exp193 (drop in
STRAIGHT, output KL, entropy change).

## RULE PARAMETERS (frozen)

  N_CTRL = 30   SEED = 195

## Decision rule (per cell, exp193's)

Q1 UP_STANDS_OUT / UP_NOT_SPECIAL; Q2 MEANING_MATTERS /
CONCENTRATION_SUFFICIENT / SCHEMA_WEAKER; Q3 the drop–KL correlation,
reported. Across cells: exp193's Q1 result **REPLICATES** if both new
cells return UP_STANDS_OUT; **PARTIAL** if one; **DOES_NOT_REPLICATE**
if neither.

## Committed predictions

- GPT-2 layer 4 UP_STANDS_OUT: **15%** (exp190 says its random pushes
  already move agreement)
- Pythia layer 8 UP_STANDS_OUT: **35%**
- REPLICATES: **8%**

## RESULT, GPT-2 cell (graded 03:28 IST, 03 Oct 2026; Pythia layer-8 cell still running)

First run was all NaN (my bug: the rare words are multi-token and I
had filtered them out, leaving the frequency direction empty); fixed
and re-run from scratch. Output: exp195_output_gpt2-medium_L4.txt.

GPT-2-medium, layer 4 (unsteered agreement 0.655):

| push along | drop | output KL |
|---|---|---|
| UP-DOWN | +0.063 | 2.57 |
| VALENCE (largest schema) | +0.101 | 2.41 |
| other schema axes | +0.028 to +0.081 | 2.6–3.2 |
| SINGLE (30), median [5%, 95%] | +0.067 [+0.049, +0.149] | 3.26 |
| PAIR (30) | +0.070 [+0.042, +0.142] | 3.17 |
| MUSH (30) | +0.076 [+0.043, +0.139] | 2.91 |
| GAUSS (30) | +0.130 [+0.087, +0.172] | 2.62 |

**UP_NOT_SPECIAL | CONCENTRATION_SUFFICIENT.** In GPT-2 every push
of this size, including pure noise, lowers the later layers'
agreement by about as much, and UP sits in the middle of the pack
(noise disturbs it most). corr(drop, KL) +0.15. As predicted (85%).
exp193's Pythia-layer-4 picture does not transfer to GPT-2.

## RESULT, Pythia layer-8 cell, and overall (graded 03:59 IST, 03 Oct 2026)

Output: exp195_output_pythia-410m_L8.txt. Unsteered agreement 0.254.

| push along | drop | output KL |
|---|---|---|
| UP-DOWN | +0.022 | 1.47 |
| VALENCE (largest schema) | +0.033 | 1.30 |
| other schema axes | +0.010 to +0.021 | 1.2–1.6 |
| SINGLE (30), median [5%, 95%] | +0.013 [+0.004, +0.021] | 1.64 |
| PAIR (30) | +0.016 [+0.006, +0.020] | 1.54 |
| MUSH (30) | +0.016 [+0.009, +0.020] | 1.46 |
| GAUSS (30) | +0.011 [+0.006, +0.017] | 0.93 |

**UP_NOT_SPECIAL | CONCENTRATION_SUFFICIENT.** At layer 8 UP-DOWN is
just above the concentrated controls' 95th (+0.022 vs +0.020) but is
not the largest schema axis (VALENCE is), and the controls now move
agreement too. corr(drop, KL) +0.13.

### Overall: DOES_NOT_REPLICATE (0 of 2 new cells).
exp193's picture, UP pushing the later layers apart far more than
anything else, is specific to Pythia at layer 4. Four layers deeper
in the same model, and in GPT-2, pushes of every kind lower the
later layers' agreement about equally. The agreement measure does
not separate real directions from concentrated or random ones in
general; exp196 showed it does not separate used from unused SAE
features either. One odd cell remains odd and is recorded as that.
Niamh's question, "does the model pay attention to this direction",
is not answered by this measure. Closed unless a mechanism for the
layer-4 oddity turns up.

### Grades
- GPT-2 L4 UP_STANDS_OUT (15%): did not occur.
- Pythia L8 UP_STANDS_OUT (35%): did not occur.
- REPLICATES (8%): did not occur.
