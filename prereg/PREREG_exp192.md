# PRE-REGISTRATION — exp192: DOES THE EIGHT-SCHEMA STRUCTURE (FINDING 3) SURVIVE A CLEAN TEST?

Written 00:15 IST, 03 Oct 2026, before any analysis code for this experiment, by Claude
(session labelled Fable 5.1). Niamh at 00:14: "well i dont see why it
would hold". Frozen at the commit carrying this text. No model run is
needed: the word states already exist (exp175's cache, single-token
words inside three neutral sentences, start token, all 24 layers).

## Plain summary

Finding 3 (exp123): the eight schema axes keep the same pattern of
similarities to each other from layer to layer (0.91, against 0.79
for scrambled axes), and the six pairs Lakoff links are more similar
than the twenty-two he doesn't (+0.21 against +0.00). Those axes were
built from bare words without a start token, so axes could resemble
each other through shared first-slot content. This reruns exp123's
two numbers with clean axes and exp123's own null.

## Design

Word states: exp175_cache_wordstates.npz (Pythia 410M, layers 0–23,
mean over three frames). Lists: the lab's eight schema lists as
filtered to single tokens in exp175_stimuli.json (UP-DOWN 35 / 39;
BALANCE 15 / 6; the other six as listed there). Frequency axis from
COMMON minus RARE in the same states; each axis frequency-stripped
as in exp123 (no other strip, to match exp123).

Per layer: the 8 x 8 matrix of cosines between the axes.
- **M2**: the mean cosine between layers of the matrix's upper
  triangle, over all pairs of distinct layers.
- **M3**: the mean cosine over exp123's six predicted pairs minus the
  mean over the other twenty-two, averaged over layers.
Null (exp123's "strong null"): K_NULL random partitions of the pooled
anchor words into eight pseudo-schemas with the same pole sizes,
everything else identical; the same two numbers for each.

## RULE PARAMETERS (frozen)

  K_NULL = 100   SEED_NULL = 192

## Decision rule

A number "stands out" if the real value is above the null's 95th
percentile.
- **STRUCTURE_HOLDS**: both M2 and M3 stand out.
- **STRUCTURE_PARTIAL**: one of them does.
- **STRUCTURE_GONE**: neither does.
Reported: the real and null values; which predicted pairs are
positive at most layers; whether BALANCE is the most-connected axis.

## Committed predictions

- STRUCTURE_GONE **50%**; PARTIAL **30%**; HOLDS **20%**.

## What I will NOT do

- No changing lists or the predicted-pair set after seeing numbers.
- No reading M2 alone as the finding: axes built from fixed word sets
  resemble themselves across layers whatever they measure; the null
  is what gives M2 meaning.

## Deviations and stubs

Flagged: BALANCE's negative pole has only six single-token words in
the lab's list. The axis is noisier than exp123's.

## RESULT + GRADES (graded 00:16 IST, 03 Oct 2026)

Raw output: exp192_output.txt. Numbers: exp192_results.json. No model
run; exp175's cached clean word states, all 24 layers.

| number | real | null median | null 95th | stands out? | exp123 (old) |
|---|---|---|---|---|---|
| M2, configuration similarity across layers | 0.937 | 0.853 | 0.905 | yes | 0.91 vs 0.79 |
| M3, predicted minus unpredicted couplings | +0.157 (+0.169 vs +0.012) | −0.005 | +0.080 | yes | +0.214 vs +0.004 |

**VERDICT: STRUCTURE_HOLDS.**

### Detail
- Four of the six predicted pairs are positive at all 24 layers:
  UP-DOWN–LIGHT-DARK +0.25, LIGHT-DARK–BALANCE +0.22,
  FORCE–DIFFICULTY-BURDEN +0.25, FORWARD-BACK–PATH-MOTION +0.24.
  Two are near zero: UP-DOWN–BALANCE +0.02, UP-DOWN–FORCE +0.03.
- Unpredicted pairs average +0.01.
- **BALANCE is no longer the centre.** On clean axes it is the LEAST
  connected of the eight (sum of |cos| to the others 0.55; LIGHT-DARK
  1.06, DIFFICULTY-BURDEN 1.02). The old "BALANCE at the centre of
  the system" was the token split: BALANCE had the most lopsided
  single-token split, which made it resemble every other axis.
- The null here is exp123's own (random partitions of the same clean
  words into eight pseudo-schemas of matched sizes), which is the
  stronger of the two nulls used today: it keeps the words and
  scrambles only which axis they belong to.

### Caveats
- BALANCE's negative pole has six words; its row is the noisiest.
- One model (Pythia 410M); the three frames of exp175.

### Grades
- STRUCTURE_HOLDS (20%): occurred. GONE was the favourite (50%).
