# PRE-REGISTRATION — exp179: THE CAUSAL TEST, AGAINST MATCHED CONTROLS

Written 2026-10-02 18:21 IST, BEFORE any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Item 3 of what
Niamh asked for at 18:04. Frozen at the git commit that carries this
text.

Seen before freezing: exp168's saved output, exp177's results, and a
post-hoc check that exp168's schema slopes do not clearly track the
schemas' token gaps (r −0.44, n 8). exp178 is running and has
produced no results at the time of writing.

## Plain summary

exp168 (June) made GPT-2's attention sharper or more diffuse at one
layer and found that token states move along the BALANCE axis as
attention gets more diffuse (toward "unbalanced"), more than along
any of the other seven schema axes. That was the causal leg of the
BALANCE–entropy story.

The other seven schemas are a weak comparison, because none of them
has BALANCE's lopsided token split. This experiment repeats exp168
and compares the BALANCE axis with fake axes built from random words
with the same token split, and repeats it with the clean in-sentence
BALANCE axis from exp178.

## Design

GPT-2-medium. exp168's procedure, imported where possible: attention
scores at one layer divided by tau, tau in {0.5, 0.7, 0.85, 1.0,
1.25, 1.5, 2.0}; exp166's 40 prompts; per prompt, the mean projection
of token states at that layer on each axis and the mean achieved
entropy; per axis, the slope of mean projection against mean achieved
entropy across the seven tau values. Layer 3 is PRIMARY (as exp168),
layer 8 is reported.

Axes, all norm-orthogonalised as in exp168:
- the real BALANCE axis (bare-word protocol, as exp168);
- MATCHED: N_RAND fake axes from random pool words with BALANCE's
  token split (9 single + 6 split against 15 split), as exp178 Part 1;
- SINGLE-ONLY: N_RAND fake axes, 15 single-token words a side;
- the clean in-sentence BALANCE axis (exp178 Part 2's 27 and 22
  words, four frames);
- CLEAN-RANDOM: N_RAND fake clean axes, 27 against 22 single-token
  pool words in the same frames.

Replication gate: the real BALANCE slope must reproduce exp168's
(−0.0083 at layer 3, −0.0018 at layer 8) within SLOPE_TOL, and the
achieved entropies must rise with tau. Otherwise STOP.

## RULE PARAMETERS (frozen)

  N_RAND = 300   SEED_RAND = 179   SLOPE_TOL = 0.0005   N_BOOT = 1000

## Decision rule (layer 3)

Original protocol:
- **CAUSAL_BALANCE_SPECIFIC**: the real BALANCE slope is below the
  5th percentile of the MATCHED slopes.
- **CAUSAL_TOKEN_STRUCTURE_SUFFICIENT**: not specific, and the MATCHED
  median slope is negative with at least 75% of MATCHED slopes
  negative. Diffuse attention pushes states along any axis with that
  token split.
- **CAUSAL_NEITHER**: anything else.

Clean axis:
- **CLEAN_CAUSAL_SPECIFIC**: the clean BALANCE slope lies outside the
  5th–95th percentile range of CLEAN-RANDOM slopes, and its
  prompt-level bootstrap CI excludes zero. Sign reported.
- **CLEAN_CAUSAL_NOT_SPECIFIC**: anything else.

Reported beside the verdicts: the same at layer 8; the SINGLE-ONLY
distribution; the share of each random family with a negative slope.

## Committed predictions

- P1 replication gate passes: **90%**
- P2 original protocol: TOKEN_STRUCTURE_SUFFICIENT **50%**,
  BALANCE_SPECIFIC **20%**, NEITHER **30%**
- P3 clean axis CLEAN_CAUSAL_SPECIFIC: **15%**

## What I will NOT do

- No changing layers, tau grid or thresholds after seeing numbers.
- No reading a Spearman of −1.00 across seven tau values as evidence:
  any axis with a monotone response gives it.

## Deviations and stubs

None at freeze.

## RESULT + GRADES (graded 18:29 IST, 2026-10-02)

Raw output: exp179_output.txt. Numbers: exp179_results.json. Mac GPU.
No deviations. Replication gate PASS at both layers: the real BALANCE
slope is −0.0083 CI[−0.0088, −0.0078] at layer 3 and −0.0018 at
layer 8, exactly exp168's; achieved entropy rises with tau.

### Layer 3 (PRIMARY)
- MATCHED fake axes (same token split as BALANCE, 300 of them):
  median +0.0005, 5th percentile −0.0053, 95th +0.0071, 44% negative.
  **The real BALANCE slope (−0.0083) is below every one of the 300.**
- SINGLE-ONLY fake axes: median +0.0003 [−0.0050, +0.0051].
- Clean in-sentence BALANCE axis: slope +0.0020 CI[+0.0016, +0.0024],
  the opposite sign, at the 71st percentile of CLEAN-RANDOM
  (median +0.0001 [−0.0054, +0.0057]).

**VERDICT: CAUSAL_BALANCE_SPECIFIC (original protocol) |
CLEAN_CAUSAL_NOT_SPECIFIC (clean axis).**

### Layer 8 (reported)
- Real BALANCE slope −0.0018, at the 12th percentile of MATCHED
  (median +0.0001 [−0.0027, +0.0026]). Not specific.
- Clean axis slope −0.0026 CI[−0.0031, −0.0020], at the 6th
  percentile of CLEAN-RANDOM. Not specific by the rule (5th needed).

### What this does and does not say
- At layer 3, making GPT-2's attention more diffuse moves token
  states along the ORIGINAL BALANCE axis more than along any of 300
  axes built from random words with the same token split. The token
  split does not explain the causal result. That is a registered
  outcome and it stands.
- It does NOT follow that the effect is about balance. A BALANCE axis
  built cleanly, from single-token words inside sentences, does not
  show it at layer 3 (opposite sign, unremarkable among random clean
  axes), and at layer 3 the two axes are nearly unrelated
  (cos +0.18, from exp178). So the effect belongs to something in the
  original axis's particular 30 words and how they tokenize, beyond
  the count of pieces.
- An untested candidate: the kind of fragment. Four of the fifteen
  imbalance words are "un-" negations and the rest end in pieces such
  as "-ed", "-ky", "-ly", "-ing"; the matched fake axes copy how many
  words are split, not what the pieces are. A control that matches
  morphology (for example positive words against their "un-" forms,
  on unrelated concepts) would test that.
- The absolute size is small: the mean projection moves from −0.022
  to −0.026 across the whole range of tau.
- Contrast with exp178 Part 1 (GPT-2): for the CORRELATION between
  the BALANCE projection and entropy, matched fake axes reproduce the
  published value. For the CAUSAL slope they do not.

### Grades
- P1 replication gate passes (90%): HIT.
- P2 original protocol: BALANCE_SPECIFIC occurred (given 20%;
  TOKEN_STRUCTURE_SUFFICIENT was the favourite at 50% and did not
  occur).
- P3 CLEAN_CAUSAL_SPECIFIC (15%): did not occur.
