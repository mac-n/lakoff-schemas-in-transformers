# PRE-REGISTRATION — exp183: IS THE UP LEAD "SPENT" WHEN THE MODEL WRITES ON ITS OWN?

Written 2026-10-02 19:38 IST, BEFORE any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh's
prediction (19:34) and her choice of measure (19:36) are verbatim in
HYPOTHESIS_UP_grounding_2026-10-02.md. Frozen at the git commit that
carries this text. exp182 (the fixed-text test) is running; none of
its output has been read at the time of writing.

## Plain summary

Niamh: if UPness is like potential energy it should be seen being
spent, as a tail-off, and it is not spent while the later words are
fixed. It would be spent when the model writes on its own. And the
measure should be KL divergence, not happiness.

So: end the prompt at the word, let the model write forty tokens
itself, and at each step of its own writing measure how much its
guess changes if the word is blanked. The model's own text can absorb
the word's influence; a fixed sentence cannot.

Her prediction for this experiment: early in the model's own writing,
blanking an UP word changes the text more than blanking its DOWN
opposite; and that lead tails off as the writing goes on.

## Design

Models: Pythia 410M and GPT-2-medium, each analysed separately.
Pairs: exp180's 31 vertical, 30 polar and 20 valence pairs, unchanged.
Prompts: exp182's eight frames cut off right after the slot (for
example "the card at the centre of the table said up"), real start
token.

For each word and prompt: N_SAMPLES continuations of T_GEN tokens,
sampled from the model at temperature 1 with fixed seeds. For each
sampled text: the clean run, and runs with the word's state dropped
to the layer mean at one layer, for each layer in DROP_LAYERS (the
layers where exp180b found the UP advantage; using all 24 would not
fit tonight). KL from clean to dropped at every position, summed over
those layers. Layer means are taken over all non-start tokens of all
clean runs in that model.

Position k counts tokens after the word: k = 0 is the guess for the
first written token; k = 1..40 are guesses made along the model's own
text. Bins of five: 1–5, 6–10, ..., 36–40. Per word and bin: the mean
over prompts and samples of ln(mean KL in the bin). Per pair and bin:
gap = first word minus second word.

Noise floor: a bin whose mean KL over all words is below NOISE_FLOOR
is left out of every test and reported as below the floor. (Both
members' effects fading into numerical noise would look like a
tail-off and must not be counted as one.)

Tests, per model, on the bins above the floor:
- T_early: mean gap over the first two bins (k = 1..10), vertical
  pairs, sign-flip p.
- T_tail: per-pair OLS slope of gap on bin index; mean over vertical
  pairs, sign-flip p.
- T_total: gap in ln(total KL over k = 1..40): does the UP word move
  the whole text more.
- Specificity, reported: vertical minus polar, for T_early and T_tail.
- Reported: the curves for all three sets; the absolute KL by bin;
  k = 0.

## RULE PARAMETERS (frozen)

  ALPHA = 0.05   N_PERM = 10000   N_SAMPLES = 3   T_GEN = 40
  NOISE_FLOOR = 0.0001   DROP_LAYERS = 0 2 4 6 8 10 12 14

## Decision rule (per model)

- **SPENT**: T_early mean > 0 with p < ALPHA, AND T_tail mean < 0
  with p < ALPHA. An UP lead early in the model's own writing that
  runs down. This is Niamh's prediction.
- **LEAD_NOT_SPENT**: T_early holds, T_tail does not.
- **NO_LEAD**: T_early does not hold.

Her full prediction across the two experiments: exp182 PERSISTS
(fixed text) AND exp183 SPENT (own text), in both models.

## Committed predictions

- P1 T_early holds: Pythia **50%**, GPT-2 **40%**
- P2 SPENT: Pythia **20%**, GPT-2 **15%**; both **8%**
- P3 her full prediction across exp182 and exp183, both models: **4%**

## What I will NOT do

- No changing bins, layers, the floor or the pairs after seeing
  numbers.
- No counting a tail-off that is only both effects reaching the noise
  floor.
- No reading SPENT without its specificity line: if polar pairs do
  the same, it is about default members of pairs, as in exp180.

## Deviations and stubs

Flagged at freeze: the hole is summed over 8 of the 24 layers, not
all 24. This is a cost choice, stated here and in the code. The
skipped layers are the odd ones up to 15 and everything from 16 up,
where exp180b found little or no UP advantage.

## STOPPED (20:37 IST, 2026-10-02) at Niamh's request

Niamh, 20:35: "i dont think we need to do the free writing test."
By then the opposite-pair effect had been explained as "X and ___"
(PREREG_exp180.md, post-hoc explanation), so the question this
experiment asked no longer had an object. The run was killed.

For the record, Pythia had finished before the stop: **NO_LEAD**
(early lead +0.083, p 0.16; 16 of 31 pairs; whole-text gap +0.103,
p 0.053). GPT-2 was stopped partway and has no result. Nothing here
is used anywhere.
