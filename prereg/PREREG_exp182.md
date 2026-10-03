# PRE-REGISTRATION — exp182: "UPNESS IS LIKE POTENTIAL ENERGY", POSITION BY POSITION

Written 2026-10-02 19:26 IST, BEFORE any analysis code or model run for
this experiment, by Claude (session labelled Fable 5.1). Niamh's idea
(19:21, verbatim in HYPOTHESIS_UP_grounding_2026-10-02.md): "it
actually does seem like perhaps UPness is like potential energy". She
asked for this test at 19:25. Frozen at the git commit that carries
this text. The only exp182 code at freeze is
`exp182_build_stimuli.py` (tokenizers only).

## What had already been seen (so it cannot count as a test)

- exp175, exp180: blanking a DOWN word changes the guess at the word's
  own position more; blanking an UP word changes the guesses at later
  positions more. Other opposite pairs do not cross over like that.
- exp180b (Pythia finished at freeze time, GPT-2 still running): the
  UP advantage is positive at 21 of 23 layers in Pythia.
"Later positions" has only ever been looked at as ONE average. How the
UP-minus-DOWN gap changes with distance from the word has not been
looked at by anyone.

## The working translation (Claude's, offered to Niamh at 19:21; she
asked for the test without changing it)

Potential energy = capacity held back and paid out later. Prediction:
past the word itself, the UP word's lead over its opposite keeps
GROWING with distance along the sentence. If instead the lead is a
bump just after the word that fades, the picture is wrong.

## Design

Measure: exp175 / exp180's, unchanged (drop the word's state to the
layer mean at one layer; KL from the clean guess to the dropped guess;
summed over the 24 layers), but kept separately for each position
k = 0..12 after the word (k = 0 is the word's own position).

Stimuli (exp182_stimuli.json): exp180's 31 vertical, 30 polar and 20
valence pairs, unchanged. Eight NEW frames, each with at least 13
tokens after the slot and containing none of the 162 pair words.
Models: Pythia 410M and GPT-2-medium, each analysed separately.

Per word and k: the mean over the eight frames of ln(summed KL at
distance k). Per pair and k: gap_k = first word minus second word.
Per pair: the OLS slope of gap_k on k over k = 1..12. k = 0 is left
out of the slope because the jump from k = 0 to later positions is
the part already seen.

Tests, per model:
- T_slope: the mean slope over the vertical pairs, with a sign-flip
  permutation p (N_PERM, two-sided).
- T_spec: mean vertical slope minus mean polar slope, with a
  label-shuffle permutation p.
- Reported: the gap at every k for all three sets (the curve itself);
  the gap at k = 0 and the mean gap over k = 1..12 as replications of
  exp180 on new sentences; the same slopes for valence pairs.

## RULE PARAMETERS (frozen)

  ALPHA = 0.05   N_PERM = 10000   K = 12   IDENT_TOL = 0.001

## Decision rule (per model)

- **RISES_SPECIFIC**: T_slope mean > 0 with p < ALPHA, AND T_spec > 0
  with p < ALPHA. The UP lead grows with distance, and more than for
  other opposite pairs.
- **RISES_GENERIC**: the first condition without the second.
- **FADES**: T_slope mean < 0 with p < ALPHA. The lead is a bump that
  shrinks with distance.
- **FLAT**: anything else. The lead neither grows nor shrinks
  detectably.

Across models: the potential-energy prediction is **SUPPORTED** only
if both models return RISES_SPECIFIC; **PARTLY** if both return a
RISES label; **NOT SUPPORTED** otherwise.

## Committed predictions

- P1 the exp180 pattern replicates on the new sentences in both
  models (gap at k = 0 negative, mean gap over k = 1..12 positive,
  for vertical pairs): **70%**
- P2 Pythia: FLAT **50%**, RISES (either) **25%**, FADES **25%**
- P3 GPT-2: FLAT **50%**, RISES (either) **25%**, FADES **25%**
- P4 SUPPORTED: **4%**; PARTLY or better: **10%**

## Integrity

- exp175's mechanics checks in each model before any measurement.
- Start token from the tokenizer, asserted.
- Rows written as each word finishes; the run resumes.
- Stimuli sha256:
  e141ee5bdf271ceb484521713eb883d445f58d1784d395c19061bc8a3f34734a

## What I will NOT do

- No choosing the range of k, or the pairs, after seeing the curve.
- No reading the k = 0 reversal as support: it was seen before the
  idea was stated.
- No calling RISES_GENERIC support for UP: it would be about default
  members of pairs.

## Deviations and stubs

None at freeze.

## AMENDMENT (19:30 IST, 2026-10-02): post-freeze, PRE-DATA

The run is in progress. No exp182 output has been read by anyone at
the time of writing. Nothing above is changed.

Niamh's objection to the working translation, verbatim (19:29):

> "so i dont see why the lead grows with distance is a prediction of
> the potential energy idea? i mean potential energy doesn't grow
> once uve moved a thing up."

She is right. "The lead keeps growing with distance" was Claude's
over-reading of "paid out later"; potential energy is a fixed store,
not a growing one. The decision rule above therefore tests Claude's
reading, not her idea, and its labels should be read that way:
RISES is not what her idea predicts.

What a fixed store does predict, stated now before any number is
seen: the UP lead does not run down with distance. Added reading
rule, computed from the saved per-position values:

- far gap = for each pair, the mean of gap_k over k = 7..12 (the far
  half of the measured range); tested across the vertical pairs with
  a sign-flip p (N_PERM, two-sided).
- **PERSISTS**: far gap mean > 0 with p < ALPHA, and the registered
  label is not FADES. The UP lead is still there six to twelve tokens
  on and is not shrinking.
- **RUNS_DOWN**: the registered label is FADES, or the far gap is not
  above zero at p < ALPHA.
- Specificity, reported: vertical far gap minus polar far gap, with a
  label-shuffle p.

This reading rule is still Claude's translation of "a fixed store".
Whether it is what Niamh means by potential energy is hers to say,
and is open at the time of writing.

Prediction: PERSISTS in both models **35%**.

## NOTE (19:34 IST, 2026-10-02): Niamh's own prediction, PRE-DATA

Still no exp182 output read. At 19:34 Niamh stated her prediction
(verbatim in the hypothesis note): potential energy would be seen
being spent, as a tail-off, "but it's not being spent when the tokens
say the same". exp182's later tokens are fixed. So her prediction for
this experiment is that the UP lead does not tail off: the
amendment's **PERSISTS**, in both models. A RUNS_DOWN here would count
against her reading; whether free generation shows spending is a
separate experiment, not yet designed.

## RESULT + GRADES (graded 19:56 IST, 2026-10-02)

Raw output: exp182_output_<model>.txt. Numbers:
exp182_results_<model>.json; rows in exp182_rows_<model>.jsonl. Mac
GPU. No deviations. Self-test and mechanics checks PASS.

### The curve: gap (first word minus second) at each distance k

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pythia, vertical | −0.12 | +0.48 | +0.17 | +0.07 | −0.03 | +0.09 | −0.01 | +0.02 | −0.09 | +0.00 | −0.03 | −0.05 | −0.03 |
| Pythia, polar | +0.10 | +0.37 | +0.03 | −0.01 | −0.10 | −0.06 | −0.09 | −0.14 | −0.17 | −0.14 | −0.14 | −0.07 | −0.07 |
| GPT-2, vertical | −0.17 | +0.24 | +0.06 | +0.02 | +0.11 | +0.11 | +0.04 | +0.05 | +0.01 | +0.04 | +0.07 | +0.04 | +0.03 |
| GPT-2, polar | +0.08 | +0.39 | +0.03 | +0.03 | −0.06 | +0.00 | −0.02 | −0.09 | −0.05 | −0.06 | −0.03 | −0.04 | −0.02 |

### Registered rule (Claude's "rises" reading)
- Pythia: vertical slope −0.0302 per position (p 0.0001; 4 of 31
  pairs rising). **FADES.**
- GPT-2: vertical slope −0.0094 (p 0.058). **FLAT.**
- Across models: **NOT SUPPORTED.**

### Amendment's rule (Niamh's reading: not spent in fixed text)
- Far gap (k = 7..12), vertical pairs: Pythia −0.030 (p 0.43);
  GPT-2 +0.039 (p 0.12). Neither is above zero at p < 0.05.
- **RUNS_DOWN in both models.** Her prediction for fixed text
  (no tail-off) is not met: clearly not in Pythia, and not
  established in GPT-2.

### What the curve shows
- The UP lead is almost all at the first token after the word
  (+0.48 Pythia, +0.24 GPT-2) and the second. By the third or fourth
  token it is gone in Pythia and small in GPT-2.
- So exp180's "bigger hole in the rest of the sentence" was mostly a
  bigger hole in the next one or two guesses. That measure averages
  KL over later positions, and KL is largest right after the word,
  so the near positions dominate it. "Rest of the sentence"
  overstated the reach.
- At the word's own position the gap is negative, as before (Pythia
  −0.12, p 0.12; GPT-2 −0.17, p 0.029).
- Polar and valence pairs: the same lead at k = 1, then the SECOND
  member has the bigger hole further out (polar far gap −0.12,
  p 0.0015 in Pythia; valence −0.25 and −0.13, p < 0.001). The
  vertical pairs do not go negative far out. Vertical minus polar far
  gap: +0.093 (p 0.069) in Pythia, +0.085 (p 0.014) in GPT-2.
  Reported, not a registered test of anything.
- Mean gap over k = 1..12 with every position weighted equally:
  Pythia +0.049 (p 0.11), GPT-2 +0.067 (p 0.009).

### Grades
- P1 exp180's pattern on new sentences, signs (70%): signs as
  predicted in both models; significant in GPT-2 only.
- P2 Pythia FADES (25%): occurred. P3 GPT-2 FLAT (50%): HIT.
- P4 SUPPORTED (4%), PARTLY (10%): did not occur.
- Amendment: PERSISTS in both (35%): did not occur.
