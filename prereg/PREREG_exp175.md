# PRE-REGISTRATION — exp175: IS UP GROUNDED IN HOLDING AGAINST GRAVITY (EFFECTING)?

Written 2026-10-02 17:08 IST, BEFORE any analysis code or model run for this
experiment, by Claude (session labelled Fable 5.1) with Niamh. Frozen
at the git commit that carries this text; the commit time is the
freeze time. The only exp175 code that exists at freeze is
`exp175_build_stimuli.py` (tokenizer only).

## Plain summary

exp174 asked whether UP is grounded in the layers agreeing with each
other and came back NULL. Niamh then sharpened the intuition (verbatim,
16:59): "it's the holding against gravity. upness is like u are doing
work in the world. effecting". The full trail of her words is in
HYPOTHESIS_UP_grounding_2026-10-02.md.

The measure she agreed to (17:02, "YESSS"): at each layer, let one
token fall: replace its state with the layer's average state and see
how much the model's output changes. Add that up over all 24 layers.
A token doing work in the world leaves a big hole when it drops. Two
readouts: the hole in the very next word, and the hole in the rest of
the sentence after it ("reach"). Reach decides.

Two parts:
- **Part A (words).** Do spatial UP words leave a bigger hole than
  spatial DOWN words?
- **Part B (sentences).** Fresh happy / sad / neutral sentences, intact
  and scrambled. Does the UP readout follow the size of the hole, or
  follow happy?

## The measure (E)

Pythia 410M, TransformerLens, `blocks.{L}.hook_resid_post`, L = 0..23.
Every text is BOS + tokens; token positions are 1..T.

- mu[L]: the mean state at layer L over every non-BOS token position
  of every text in that part (Part A and Part B each have their own).
- Drop(p, L): run the model with the state at layer L, position p
  replaced by mu[L]. Nothing else is touched.
- KL_q(p, L): KL divergence from the clean next-token distribution at
  position q to the dropped one.
- E_next(p)  = sum over L of KL_p(p, L).
- **E_reach(p) = sum over L of the mean over q > p of KL_q(p, L).
  PRIMARY.** (The last layer contributes zero by construction: nothing
  after it can pass the change sideways.)
- Token value: ln E. The last token of a text has no reach and is left
  out of reach averages.

## Part A — words

Stimuli (frozen in exp175_stimuli.json): a purely spatial list, 35 UP
and 39 DOWN words, each a single token when written with a leading
space; valence lists (22 positive, 19 negative); the single-token pole
words of the lab's other seven schemas. Every word is placed in three
neutral frames that mention it ("she wrote the word ___ on the board
before the lesson began that morning"), each with at least eight words
after the slot.

Word value: the mean over the three frames of ln E_reach at the word's
position.

Binding test: Cohen's d between UP and DOWN word values, with a
label-shuffle p-value (N_PERM shuffles, two-sided). Computed twice:
raw, and after regressing word frequency (wordfreq zipf) out of the
word values over all Part A analysis words.

Reported beside it:
- Specificity: the same frequency-controlled d for each of the other
  seven schemas (positive pole against negative pole). UP's rank by
  |d| among the eight.
- Valence: the same d for positive against negative valence words.
- Distance anatomy: d after also regressing out how far the word's
  state sits from the average (mean over layers of ln ||state - mu||).
  A hole can be big simply because the token was far from average;
  this separates "far" from "effective per unit of distance".
- E_next in place of E_reach.

## Part B — sentences

Stimuli: 40 NEW matched triples (happy / sad / neutral), 120
sentences, none reused from exp174, same rules (lowercase, no
punctuation, no UP / DOWN / height / posture word). Each sentence
intact and in N_SHUFFLES frozen scrambles with exactly the same
tokens. Presented once (exp174's play-it-twice control is dropped: it
confounded repetition with position).

Per token: ln E_reach, ln E_next, surprisal, and readouts at layers
{4, 8, 12, 16, 20} (unit state projected on the axis). Text value: the
mean over tokens.

Axes, built from Part A's word states (mean over the three frames,
anisotropy and frequency directions projected out as in exp154 /
exp170):
- **UP_spatial (PRIMARY)**: mean of the 35 UP words minus mean of the
  39 DOWN words.
- VAL: positive minus negative valence words.
- UP_literal_bare: exp174's bare-word spatial axis, rebuilt from its
  cached word states. Reported beside the primary, never substituting.

Gates (a failed gate makes the dependent test UNINFORMATIVE):
- G1: scrambling moves E_reach: |d_z| >= GATE_D, intact against
  scrambled, paired over the 120 sentences. Direction reported.
- G2: VAL readout happy above sad, intact, paired by triple, d_z >=
  GATE_D at >= LAYERS_MAJ layers.

Tests (each at 5 layers, majority rule):
- T1: at fixed valence, the UP readout shifts toward the order with
  the bigger hole: d_z >= EFFECT_D in BOTH happy and sad at >=
  LAYERS_MAJ layers.
- T2: the crossing cells chosen by G1's direction. If intact has the
  bigger hole: sad-intact against happy-scrambled. Otherwise
  sad-scrambled against happy-intact. Holds at d_z >= EFFECT_D at >=
  LAYERS_MAJ layers.
- T3: regression over every (sentence x order) cell, 240 cells:
  z(UP readout) on z(ln E_reach) + z(surprisal) + valence. Bootstrap
  over sentences, N_BOOT. Holds if the CI for the E_reach beta is
  entirely above 0 at >= LAYERS_MAJ layers.
- Reported, non-binding: T0 (UP readout, happy against sad, intact);
  T2b (the crossing in Niamh's own phrasing: sad-intact against
  happy-scrambled, whatever G1's direction); T4 (does valence itself
  move E_reach: happy against sad, intact, by triple); D1 (within
  intact sentences, the token-level correlation between ln E_reach and
  the UP readout after removing token position, averaged over
  sentences).

## RULE PARAMETERS (frozen; asserted against this file at runtime)

  D_HI = 0.50        D_LO = 0.20        ALPHA = 0.05
  N_PERM = 10000     GATE_D = 0.50      EFFECT_D = 0.30
  FLAT_D = 0.20      LAYERS_MAJ = 3 (of 5)
  N_BOOT = 2000      N_SHUFFLES = 3     IDENT_TOL = 0.001

## Decision rule — Part A (d = UP minus DOWN; "ctrl" = frequency
regressed out)

- HOLDS_SPECIFIC: d_raw >= D_HI and d_ctrl >= D_HI, both with
  p < ALPHA, AND UP has the largest |d_ctrl| of the eight schemas.
- HOLDS_GENERIC: the first condition without the second.
- WRONG_SIGN: d_raw <= -D_HI and d_ctrl <= -D_HI, both with
  p < ALPHA. DOWN words do more work. A miss for the hypothesis,
  reported as its own finding.
- NULL: p_ctrl >= ALPHA, or |d_ctrl| < D_LO.
- WEAK: anything else.

## Decision rule — Part B (requires G1 and G2)

- UP_FOLLOWS_EFFECT: T1 and T3.
- STRONG_DISSOCIATION: T1, T3 and T2.
- PERPLEXITY_NOT_EFFECT: T1 holds but T3 does not.
- UP_FOLLOWS_VALENCE_ONLY: |d_z| < FLAT_D in both valences at >=
  LAYERS_MAJ layers.
- MIXED: anything else, reported cell by cell.

## Committed predictions (Claude's; Niamh delegated the odds at 16:36)

Calibration: mechanism bets in this lab are now 0-for-9 (exp174 was
the ninth).

- P1 Part A HOLDS_SPECIFIC: **8%**
- P2 Part A d_raw >= +0.50 with p < 0.05: **15%**
- P3 Part A NULL: **55%**
- P4 G1 passes: **80%**; given a pass, intact has the bigger hole:
  **60%**
- P5 T1 holds: **20%**
- P6 STRONG_DISSOCIATION: **5%**
- P7 valence moves E_reach (T4 |d_z| >= 0.50): **20%**
- META: the modal outcome is Part A NULL or WEAK with Part B MIXED or
  UP_FOLLOWS_VALENCE_ONLY: **~60%**

## Integrity

- Mechanics checks on the real model BEFORE the run, on frames holding
  common words only (so nothing about UP is seen): (i) dropping a
  token to its OWN clean state changes nothing (max KL < IDENT_TOL);
  (ii) dropping at the last layer changes nothing at later positions;
  (iii) dropping changes nothing at earlier positions. Any failure:
  STOP.
- Synthetic self-test of both verdict paths on planted worlds before
  the model run. Part A worlds: UP only; every schema, another one
  larger; a difference that is pure word frequency; nothing. Part B
  worlds: UP follows the hole; UP follows valence only; UP follows
  surprisal; G1 fails.
- Results written to disk as each word and each text finishes; the
  run resumes from what is there.
- Stimuli sha256 (exp175_stimuli.json):
  39cc3d1da0a1ff9dd3b0b25333016a6aff770f89cdbf19f4000023b9b2f23397

## Where this differs from what Niamh was told at ~17:00

She agreed to the measure and to "the same two questions as before",
a bigger spatial word list, and fresh sentences. Choices made after
that, before any data, which she has not seen:
1. Part A asks the question directly (do UP words leave a bigger hole
   than DOWN words) instead of through exp174's carrier-direction
   cosine. exp174's self-tests showed that cosine is a weak statistic.
2. Words are measured inside three neutral frames, not bare. A bare
   word has no "rest of the sentence" to reach into, and inside a
   sentence nearly every word is one token.
3. The "first exploratory look" on exp174's sentences is skipped. The
   mechanics checks cover what it was for, and skipping it keeps the
   old data out of the new design.
4. Three scrambles per sentence, not five, and one presentation, not
   two.

## What I will NOT do

- No changing a threshold, the primary readout (reach), or the primary
  axis after seeing numbers.
- No reporting the raw Part A difference without the
  frequency-controlled one beside it.
- No treating a failed gate as a result in either direction.
- No promoting E_next, UP_literal_bare, or the distance-controlled
  value if the primary comes back null.

## Deviations and stubs

None at freeze. Any that arise go here, in the code as
`# STUB — not the real mechanism: <what's missing>`, and at the top of
the report to Niamh.

## POST-FREEZE LOG (pre-data; rules above unchanged)

Written 17:13 IST, 2026-10-02. The first launch (17:11) crashed in the
first clean pass of Part A, before any hole was measured and before
any number about UP existed. Nothing in the rule sections was edited.

1. Crash: the script asked the Mac GPU backend for 64-bit floats,
   which it does not support. Fixed by converting on the CPU.
2. **Start-token bug found.** In this TransformerLens build,
   `default_prepend_bos` is False for Pythia: `model.to_tokens(w)`
   does NOT add a start token. Checked directly:
   to_tokens("the") = [783]. Consequences:
   - exp175's script took its "BOS" from `to_tokens("a")[0, 0]`,
     which is the token "a". Fixed before any data: the start token
     now comes from the tokenizer (id 0), with assertions. This makes
     the code match this prereg ("every text is BOS + tokens").
   - exp174 did the same thing: every exp174 sentence was prefixed
     with "a", not a start token; and its bare words had no start
     token at all. See the ERRATUM added to PREREG_exp174.md.
   - UP_literal_bare (the secondary axis here, rebuilt from exp174's
     cached bare-word states) is built from states at mixed sequence
     positions, some at position 0. It is still reported beside the
     primary as registered, and it should be read as bent.
   - exp175's own word states are unaffected by design: words sit
     mid-frame, at position 5 or later.
3. Backend check prompted by a TransformerLens warning ("MPS backend
   may produce silently incorrect results", PyTorch 2.12.0): the same
   five texts run on CPU and on the Mac GPU agree to a relative
   residual difference of at most 4e-5 and a KL of at most 1.5e-6;
   a batched hooked run agrees to a KL of 8e-5. IDENT_TOL is 1e-3.

## RESULT + GRADES (graded 17:55 IST, 2026-10-02)

Integrity: self-test PASS; stimuli sha and constants asserted;
mechanics checks PASS (own-state drop, last-layer drop and
earlier-position effects all below 6e-7 against a tolerance of 1e-3;
real drops change the next-word distribution by 30–53 nats). Real
start token used throughout. Raw output: exp175_output.txt.

### Part A — frozen verdict: **NULL**
- 35 spatial UP words against 39 spatial DOWN words, each inside three
  neutral frames. ln E_reach: UP +0.800, DOWN +0.668.
- d_raw +0.38 (p 0.104). Frequency-controlled d +0.40 (p 0.098). The
  rule needed d >= 0.50 with p < 0.05 in both.
- The direction is the hypothesised one. UP rank 3 of 8 by |d|
  (BALANCE +0.58 on 15 / 6 words and FORWARD-BACK −0.54 are larger;
  neither is significant).
- Valence words show no such difference (d −0.19, p 0.53).
- Non-binding: also controlling distance from the average state,
  d +0.54 (p 0.024). With the next-word readout in place of reach the
  sign flips (d −0.27, p 0.26).
- Read plainly: not established. A moderate effect in the predicted
  direction that this sample could not confirm (power for d = 0.4 at
  35 / 39 is about 40%). Not promoted.

### Part B — frozen verdict: **MIXED** (UP_literal_bare beside it: MIXED)
- G1 PASS: intact sentences leave the bigger hole (reach d_z +1.61;
  next-word version +3.32). G2 PASS (+2.2..+3.0).
- T1 (at fixed valence, UP readout shifts toward the bigger hole):
  holds at L4 only (happy +1.17, sad +0.59). At L16 and L20 it
  reverses strongly (−1.0..−1.7). 1 of 5: False.
- T2 and T2b (same cells here): happy-scrambled reads more UP than
  sad-intact at every layer, d_z −1.07..−2.12. UP goes with happy,
  not with the hole.
- T3: beta on E_reach above zero at L4 only (+0.15), below zero at L20
  (−0.26). False.
- T4: valence does not move the hole (happy against sad, d_z +0.18).
- T0: happy sentences read more UP than sad ones on the clean
  in-sentence spatial axis at every layer, d_z +1.22..+1.62, with no
  UP / DOWN / height word in any sentence. This repeats exp174's side
  result on a clean instrument and fresh sentences.
- D1: token-level correlation inside intact sentences is negative at
  L4 (−0.15) and L8 (−0.07), positive at L20 (+0.10). No consistent
  sign.

### Grades
- P1 HOLDS_SPECIFIC (8%): did not occur.
- P2 d_raw >= +0.50 with p < 0.05 (15%): did not occur.
- P3 Part A NULL (55%): HIT.
- P4 G1 passes (80%): HIT; intact has the bigger hole (60%): HIT.
- P5 T1 holds (20%): did not occur.
- P6 STRONG_DISSOCIATION (5%): did not occur.
- P7 valence moves E_reach (20%): did not occur.
- META (~60%): HIT.

### What this does and does not say
- In Pythia 410M, the UP readout of a sentence follows its valence and
  does not follow how much its tokens are holding up, under this
  measure. That is two measures of "something tracking multiple
  layers" now (agreement in exp174 on a bent instrument, effecting
  here on a clean one) with no text-level support.
- At word level there is a lean in the predicted direction that is
  worth one properly powered replication: more spatial words, more
  frames, preregistered, and nothing else changed.
- One model.

### Deviations and stubs
None in the code. The four design choices Niamh had not seen at
freeze are listed above; the two pre-data fixes are in the
POST-FREEZE LOG.

## POST-HOC note (19:15 IST, 2026-10-02; NOT registered): distance from the average, by layer

Prompted by Niamh's question at 19:14 ("so the word pushes against
gravity?"). From exp175's cached in-sentence word states (Pythia
410M, 35 UP and 39 DOWN words, three frames): at every one of the 24
layers the UP words sit slightly CLOSER to the layer average than the
DOWN words (ratio 0.956..0.994; effect size −0.11..−0.58; all-layer
mean distance UP 19.16, DOWN 19.65, d −0.55). So the bigger hole UP
words leave (exp175 lean, exp180 replication) is not a matter of
standing further from the average. It is more effect per unit of
distance, which is why the distance-controlled comparison in Part A
was the larger one (d +0.54 against +0.40). Not checked in GPT-2;
exp180 did not save word states.
