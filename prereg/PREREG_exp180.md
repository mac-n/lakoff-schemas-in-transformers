# PRE-REGISTRATION — exp180: "HOLDING AGAINST GRAVITY", WORD LEVEL, PAIRED RETRY

Written 2026-10-02 18:40 IST, BEFORE any analysis code or model run for
this experiment, by Claude (session labelled Fable 5.1). Niamh asked
for it at 18:06 ("i would like to retry 7"). Frozen at the git commit
that carries this text. The only exp180 code at freeze is
`exp180_build_stimuli.py` (tokenizers only).

Seen before freezing: exp175's full results (UP words left a bigger
hole than DOWN words, d +0.38, p 0.10, in Pythia, on three frames).

## Plain summary

Niamh's intuition (16:59): UP is "the holding against gravity ...
doing work in the world. effecting". The measure she agreed to: let a
token fall to the layer's average state at each layer and add up how
much the rest of the sentence's predictions change (the "hole").
exp175 found UP words leaving a bigger hole than DOWN words, not
significantly. This is the fairer retry:

- **matched opposite pairs** (up / down, ceiling / floor), so each UP
  word is compared with its own opposite;
- **eight new sentences** per word instead of three;
- **a second model** (GPT-2-medium), which has seen none of this;
- **two comparison sets**: opposite pairs on other dimensions
  (big / small, early / late) and valence pairs (happy / sad). If
  those show the same advantage for the first member, the effect is
  about "the positive, default member of a pair", not about UP.

## Design

Measure: E_reach exactly as PREREG_exp175.md defines it (drop the
state at one position to the layer mean, KL at later positions,
summed over 24 layers), natural log, real start token. Layer means
are taken over all non-start tokens of all exp180 texts in that
model. Word value: the mean over the eight frames of ln E_reach at
the word's position.

Stimuli (exp180_stimuli.json): 31 vertical pairs, 30 polar control
pairs, 20 valence pairs; in each pair the first word is the UP /
default / positive member; every word is one token with a leading
space in both tokenizers and is used once. Eight neutral frames that
mention the word, none reused from exp175, none containing a
vertical word.

Models: Pythia 410M and GPT-2-medium (both 24 layers), each analysed
separately.

Per pair: diff = value(first word) − value(second word).

Tests, per model:
- T_vert: mean diff over the vertical pairs, with a sign-flip
  permutation p (N_PERM, two-sided) and d_z.
- T_freq: the intercept of diff regressed on the pair's difference in
  word frequency (wordfreq zipf), with a bootstrap CI over pairs
  (N_BOOT). This is the UP advantage at equal frequency.
- T_spec: mean vertical diff minus mean polar-control diff, with a
  label-shuffle permutation p.
- Reported: the same T_vert-style statistics for the polar and
  valence sets; the next-word readout in place of reach.

## RULE PARAMETERS (frozen)

  ALPHA = 0.05   N_PERM = 10000   N_BOOT = 2000   IDENT_TOL = 0.001

## Decision rule (per model)

- **HOLDS_SPECIFIC**: T_vert mean > 0 with p < ALPHA, T_freq CI
  entirely above 0, AND T_spec > 0 with p < ALPHA.
- **HOLDS_GENERIC**: the first two conditions without the third. UP
  words leave bigger holes than DOWN words, and so do the first
  members of other opposite pairs.
- **WRONG_SIGN**: T_vert mean < 0 with p < ALPHA.
- **NULL**: anything else.

Across models: **REPLICATES** if both models return HOLDS_SPECIFIC or
HOLDS_GENERIC; **SPECIFIC_REPLICATES** if both return HOLDS_SPECIFIC.

## Committed predictions

Mechanism bets here are 0-for-9, with one lean (exp175) on these same
words in Pythia.

- P1 Pythia HOLDS (either kind): **35%**
- P2 GPT-2 HOLDS (either kind): **20%**
- P3 REPLICATES: **12%**
- P4 SPECIFIC_REPLICATES: **4%**
- P5 where a model HOLDS, it is GENERIC rather than SPECIFIC: **65%**

## Integrity

- exp175's mechanics checks rerun in each model before any hole is
  measured (own-state drop, last-layer drop, earlier positions), on
  frames holding common words. STOP on failure.
- Start token taken from the tokenizer and asserted.
- Rows written to disk as each word finishes; the run resumes.
- Stimuli sha256:
  a6f0a07f3acec1f6bb6552e7d3d0758b41fca23ef7c8c6fea9207a650aade5ab

## What I will NOT do

- No dropping or adding pairs after seeing numbers.
- No calling a Pythia-only result a replication.
- No reading HOLDS_GENERIC as support for the UP hypothesis: it would
  be a finding about default members of pairs.

## Deviations and stubs

None at freeze. Note for the record: Pythia has seen these vertical
words in exp175 (different frames); GPT-2 has not. GPT-2 is the
independent test.

## RESULT + GRADES (graded 18:58 IST, 2026-10-02)

Raw output: exp180_output_<model>.txt. Numbers:
exp180_results_<model>.json; rows in exp180_rows_<model>.jsonl. Mac
GPU. No deviations. Self-test PASS; mechanics checks PASS in both
models (all below 7e-7 against a tolerance of 1e-3).

### Reach (PRIMARY): first member minus second member, ln E_reach

| model | set | pairs | mean diff | d_z | sign-flip p | pairs positive | at equal frequency [95% CI] |
|---|---|---|---|---|---|---|---|
| Pythia 410M | vertical | 31 | +0.203 | +0.94 | 0.0001 | 27 | +0.199 [+0.123, +0.276] |
| Pythia 410M | polar control | 30 | +0.160 | +0.47 | 0.016 | 19 | +0.216 [+0.088, +0.345] |
| Pythia 410M | valence | 20 | −0.004 | −0.02 | 0.95 | 8 | +0.275 [+0.107, +0.404] |
| GPT-2-medium | vertical | 31 | +0.145 | +0.50 | 0.0075 | 23 | +0.143 [+0.043, +0.239] |
| GPT-2-medium | polar control | 30 | +0.143 | +0.50 | 0.0093 | 23 | +0.150 [+0.036, +0.266] |
| GPT-2-medium | valence | 20 | +0.031 | +0.13 | 0.60 | 11 | +0.276 [+0.097, +0.455] |

Vertical minus polar: Pythia +0.043 (p 0.56); GPT-2 +0.002 (p 0.97).

**VERDICT: HOLDS_GENERIC in Pythia, HOLDS_GENERIC in GPT-2.
Across models: REPLICATES. Not SPECIFIC.**

### What this says
- UP words leave a bigger hole in the rest of the sentence than their
  DOWN opposites. It holds in both models, including GPT-2, which had
  seen none of these words, and it holds at equal word frequency.
- The first members of other opposite pairs (big over small, early
  over late, open over closed) show the same advantage, the same size.
  So, per this prereg, it is a finding about the default / positive
  member of an opposition, not about UP in particular.
- Valence pairs show nothing raw and an advantage of the same size at
  equal frequency: positive words are more frequent, and frequency
  works the other way. All three sets agree once frequency is
  equalised (+0.14 to +0.28).

### Reported, not judged
- With the next-word readout the vertical pairs go the OTHER way in
  both models (Pythia −0.136, p 0.068; GPT-2 −0.158, p 0.047), as in
  exp175 (d −0.27). DOWN words change the very next prediction more;
  UP words change the rest of the sentence more. Polar and valence
  pairs do not show this reversal.
- Not tested: whether "default member of a pair" and "the UP side of
  Lakoff's metaphor bundle" (MORE, BIG, STRONG, BRIGHT) can be told
  apart. Many polar pairs here are both. A set of polar pairs with no
  UP mapping (left / right has no default; before / after, in / out,
  on / off do) against pairs with one would be the test, and it
  would need its own prereg.

### Grades
- P1 Pythia HOLDS (35%): occurred.
- P2 GPT-2 HOLDS (20%): occurred.
- P3 REPLICATES (12%): occurred.
- P4 SPECIFIC_REPLICATES (4%): did not occur.
- P5 where a model HOLDS it is GENERIC (65%): HIT, both models.

Calibration note: the main effect was given 12% and happened. The
"mechanism bets are 0-for-9" prior was applied to an intuition of
Niamh's that had already shown a lean in the right direction; that
lean deserved more weight than I gave it.

## POST-HOC frequency checks (19:13 IST, 2026-10-02; NOT registered)

Niamh asked at 19:12 whether the result is a frequency confound. The
registered control (dictionary frequency, wordfreq zipf) is T_freq
above. Added after the fact, from the saved word values plus one
forward pass per model to get each model's own log p(word | start
token):

- Vertical pairs: the UP word is the more frequent one in only 17 of
  31 pairs (mean gap +0.06 zipf). Advantage at equal dictionary
  frequency: Pythia +0.199 [+0.132, +0.275], GPT-2 +0.143 [+0.047,
  +0.239]. At equal model's-own probability: +0.204 [+0.134, +0.287],
  +0.153 [+0.043, +0.252]. With both controls: +0.200, +0.149.
- In the 13 vertical pairs where the UP word is the RARER one: Pythia
  mean +0.187 (11 of 13 positive), GPT-2 +0.161 (8 of 13).
- Polar pairs behave the same way (first-member-rarer subset positive
  in 9 of 11 in both models).
- Across all 162 words, more frequent words leave slightly SMALLER
  holes (r −0.22 Pythia, −0.17 GPT-2), so frequency works against the
  effect wherever the first member is the commoner word.
- Valence pairs cannot be judged: the positive word is the more
  frequent one in 19 of 20 pairs, and the answer changes with the
  frequency measure (Pythia: +0.275 at equal dictionary frequency,
  +0.001 at equal own probability).
- Caveat: "model's own probability" is the probability right after
  the start token, a crude proxy. In Pythia it correlates only +0.14
  with dictionary frequency (+0.80 in GPT-2).

## FOLLOW-UP exp180b (19:16 IST, 2026-10-02): layer-by-layer breakdown — DESCRIPTIVE

Niamh asked for it at 19:15 ("yes please"). Written before the run.
Same stimuli, models, frames and measure as exp180; the only change
is that each layer's contribution to the hole is kept instead of only
the total. No decision rule and no verdict: this describes where in
the stack the exp180 advantage comes from.

Fixed in advance, so the picture is not chosen after the fact:
- per word and layer L (0..22; the last layer contributes zero to
  reach by construction): the mean over the eight frames of
  ln(reach at layer L);
- per layer: the mean paired difference (first member minus second)
  for the vertical pairs and for the polar pairs, with a sign-flip p;
- per layer: the share of the total hole that layer contributes;
- check: the layer contributions must sum to exp180's totals.

One prediction: the vertical advantage is positive at 18 or more of
the 23 layers in both models (spread through the stack, not
concentrated in a few layers): **50%**.

### exp180b RESULT (19:30 IST, 2026-10-02) — descriptive

Outputs: exp180b_output_<model>.txt, exp180b_results_<model>.json.
Layer contributions sum to exp180's totals (relative difference
below 1e-15).

- **Where the hole comes from.** Early layers contribute most and the
  share falls steadily: Pythia 10.8% at layer 0 down to 0.1% by layer
  19; GPT-2 16.2% at layer 0 down to 0.2% by layer 19. (A state
  dropped late has few layers left through which to reach later
  words.)
- **The UP advantage, by layer.** Pythia: positive at 21 of 23
  layers, p < 0.05 at layers 0–15, largest at layers 10–12 (+0.29 to
  +0.30), gone from layer 16 up. GPT-2: positive at 23 of 23,
  p < 0.05 at layers 0–14, roughly level at +0.13 to +0.18.
- It is present from layer 0, the first layer measured.
- The other opposite pairs show the same profile in both models.
- Pythia layer 22: the reach is exactly zero for some words, so the
  log is undefined; that row is not interpretable.
- Prediction (positive at 18 or more of 23 layers in both models,
  50%): HIT (21 and 23).

## POST-HOC EXPLANATION (20:37 IST, 2026-10-02; NOT registered) — the effect is "X and ___"

Niamh, 20:29, verbatim: "so basically it's about the order in which
they appear when u list both of them at once. like so wehn u say the
default its half expecting u to say the other half of the pair a few
words later". Recorded in the hypothesis note before any check. Two
checks followed, both on existing or trivially cheap data.

1. **Split by what follows the slot.** In five of exp180's eight
   sentences the slot is followed directly by "and"; in three it is
   not. First-minus-second gap (Pythia / GPT-2):

   | set | "and" follows | something else follows |
   |---|---|---|
   | up/down pairs | +0.349 / +0.247 | −0.040 / −0.025 |
   | other opposites | +0.243 / +0.255 | +0.022 / −0.044 |
   | new fixed pairings (here/there) | +0.433 / +0.507 | −0.002 / +0.103 |
   | new scale pairs (huge/tiny) | +0.076 / −0.062 | −0.031 / +0.053 |
   | valence | +0.025 / +0.055 | −0.053 / −0.009 |

   The whole effect is in the sentences where "and" comes next.

2. **The partner's probability.** After "... up and" the model gives
   "down" 16.0% (Pythia) and 24.5% (GPT-2). After "... down and" it
   gives "up" 0.8% and 0.3%. Here/there: 10.4% against 0.7%, and
   11.3% against 0.3%. Give/take: 5.5% against 0.4%, 6.4% against
   0.4%. Huge/tiny: about 0.1% both ways. By set, the first-to-second
   expectation is stronger by 4.7x / 3.0x (up/down), 3.5x / 3.7x
   (other opposites), 7.3x / 9.6x (fixed pairings), and only 1.5x /
   1.3x for the new scale pairs and 1.6x / 1.7x for valence.

**So exp180's result is this:** English says "up and down", "here and
there", "give and take" in that order. After the first word and
"and", the model expects the second. Blank the first word and that
expectation collapses, which is a big change in the guess made at
"and". The reverse order sets up no such expectation. The sentences
were written by Claude and happened to put "and" after the slot in
most cases; that choice produced the effect. It is a fact about word
order in fixed pairings, learned from text, and has nothing to do
with UP, effecting, or stored capacity.

It also accounts for the rest: the lead sitting at the first token
after the word (exp182: the position of "and"), prepositions showing
it most ("up and down", "over and under"), the new scale pairs not
showing it, the no-link fixed pairings showing it most (exp185).

**Standing of exp180's verdict:** HOLDS_GENERIC is what the frozen
rule returned and the numbers are correct. The interpretation
"default members of opposite pairs leave bigger holes" is withdrawn
in favour of the above.
