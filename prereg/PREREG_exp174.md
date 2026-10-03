# PRE-REGISTRATION — exp174: IS UP GROUNDED IN LAYER AGREEMENT?

Drafted 2026-10-02 16:33–16:39 IST (file times), BEFORE any analysis
code or model run, by Claude (session labelled Fable 5.1) with Niamh.
FROZEN on Niamh's "go." (16:40 IST) at the git commit that carries
this text; the commit time is the freeze time. The only code that
existed at freeze is `exp174_build_stimuli.py` (tokenizer only).
Read the FREEZE AMENDMENTS section at the bottom: where it conflicts
with the text above it, the amendments win.

(exp173 is reserved for the grammar-of-time design in CONSULT_LOG 22:18,
still unrun.)

## Plain summary

Niamh's hypothesis: UP is grounded in the layers of the model agreeing
with each other, the way upright posture in a human is hips, legs and
shoulders all in agreement. Her words are on record, timestamped, in
HYPOTHESIS_UP_grounding_2026-10-02.md.

Two parts, one run:
- **Part A (words).** The BALANCE-and-norm recipe, pointed at UP. Give
  every word a "layer agreement" score. Find the direction in the model
  along which that score varies. Ask whether the UP axis lines up with
  it, more than the other seven schemas do.
- **Part B (texts).** Niamh's dissociation. Happy and sad sentences,
  each intact and with the words scrambled. Ask whether the UP readout
  follows coherence or follows happiness.

Not in this experiment: the causal test (disturbing layer agreement
from inside the model and watching UP). That is exp175, and it gets
designed only if exp174 does not come back NULL.

## Provenance

All grounding work so far concerns BALANCE (exp154, 161, 164, 166, 168,
170, 172). Nothing has asked what UP is grounded in. Known about UP:
steering it moves the metaphor bundle (MORE / HAPPY / HIGH-STATUS) and
not literal verticality (exp116, exp143); cos(UP_clean, MAG_clean) ~ 0
within layer (exp141, as quoted in exp143's docstring).

Not the same thing as the hypothesis, and not evidence for it: exp140's
cross-layer stability of the UP *axis* (+0.74). That says the UP
direction points the same way at neighbouring layers. The hypothesis
is about a *state*: whether a given token's representation agrees with
itself from layer to layer, and whether UP rides on that.

## The carrier: layer agreement (C)

Pythia 410M, TransformerLens, `blocks.{L}.hook_resid_post`, L = 0..23,
last-token residual r[L].

- Centring: r~[L] = r[L] minus the mean residual at layer L over the
  reference set (Part A: all words; Part B: all stimulus tokens).
  Without this, every residual agrees with every other through the
  shared anisotropy direction.
- **C_adj (PRIMARY)** = mean over L = 0..22 of cos(r~[L], r~[L+1]).
  This is the measure Niamh agreed to at 16:31: how similar a
  representation at one layer is to the next layer up.
- C_straight (secondary) = ||r[23] - r[0]|| / sum_L ||r[L+1] - r[L]||.
  Straightness of the path: 1 means every layer pushed the same way.
- Path length (diagnostic) = sum_L ||r[L+1] - r[L]||.

Known weakness, registered now: a token the layers barely touch has
high agreement by inaction. A body lying flat also has its segments
"in agreement". corr(C_adj, path length) is therefore a required
instrument check, and a strongly negative value means C_adj is
measuring inactivity.

Demoted, on Niamh's objection: "the prediction settling early"
(logit-lens agreement). She pointed out it may just be low perplexity.
It is not a measure of the hypothesis here. Surprisal is a named
confound instead (Part B).

## Which UP? (found ~16:38 while building stimuli; settled 16:40)

The lab's schema-system axis "UP-DOWN" (LAKOFF_SCHEMAS_MML, 57 pairs)
is the whole metaphor bundle. Its UP pole contains happy, cheerful,
joyful, elated, euphoric, good, best, better; its DOWN pole contains
sad, depressed, gloomy, dejected, bad, worse. An axis built partly
from happy-minus-sad cannot be dissociated from happy: it reads happy
text as UP by construction.

Two UP axes are therefore measured everywhere in this experiment:
- **UP_literal** — exp141's spatial lists only (up, rise, rose, rising,
  ascend, raise, climb, lift, above, over, top, high, higher, upward /
  down, fall, fell, falling, descend, drop, sink, below, under, bottom,
  low, lower, downward).
- **UP_bundle** — the 57-pair LAKOFF_SCHEMAS_MML axis used in exp123 /
  exp170, like-for-like with the other seven schemas.

UP_literal is PRIMARY (it binds every verdict below); UP_bundle is
reported beside it and never substitutes for it. Wherever this file
says "UP" it means the primary axis. Put to Niamh as "freeze with the
spatial UP as primary ... or tell me you want the bundle axis to
decide instead"; her answer, verbatim: "go.".

## Part A — word-level coupling

exp154 word set, residual collection and strip protocol, unchanged
(aniso = mean direction; freq = COMMON minus RARE, orthogonalised;
both projected out; unit-normalised). Schema axes as exp170
(`schema_dir`, mean of positive pole minus mean of negative pole, so
positive = the UP pole). Layers {4, 8, 12, 16, 20}.

Directions per layer, built with exp170's `cov_dir`:
- **d_coh_single (PRIMARY)** — z(C_adj) over SINGLE-TOKEN words only.
  Token count cannot vary, so the exp170 confound is removed by
  construction.
- d_coh_resid — all words, z(C_adj) residualised on token count.
- d_coh_resid2 — all words, residualised on token count and z(norm).
- d_coh_orig — all words, no control. Upper bound only, never headline.
- d_tokcount, d_norm_single — for confound anatomy.

Measures:
1. Instrument checks on C_adj: mean and sd; corr with token count,
   with norm at each layer, with zipf frequency (as exp157b), with
   path length. Carrier sanity per direction (corr of projection with
   its own target), UNINFORMATIVE below CARRIER_MIN.
2. HEADLINE: cos(UP-DOWN_stripped, d_coh_single) at each layer.
3. Specificity: the same cosine for the other seven schemas
   (IN-OUT_CLEAN, FORWARD-BACK, PATH-MOTION, LIGHT-DARK, FORCE,
   BALANCE, DIFFICULTY-BURDEN). UP's rank by |cos|.
4. Null: N_NULL axes from random partitions of the pooled schema
   anchor words, pole sizes matched to UP-DOWN (exp123 style).
5. Valence anatomy: cos(VAL, d_coh_single), VAL built from exp141's
   valence lists; and the headline recomputed after projecting VAL out
   of the UP axis. This is the word-level form of Niamh's dissociation.

## Part B — the happy / incoherent dissociation

Stimuli: 40 matched triples (happy / sad / neutral versions of the same
sentence frame), 120 sentences, 13–16 words, lengths within a triple
differing by at most one word. Lowercase, no punctuation, so a scramble
has no stray capitals or full stops. No word from either UP list or
its simple inflections, and none from a supplementary height / posture
list, so the UP readout cannot be lexical. Written by the drafting
Claude, built and checked by `exp174_build_stimuli.py` (tokenizer only,
no model), frozen in `exp174_stimuli.json` together with the shuffle
permutations.

Conditions per sentence:
- INTACT.
- SHUFFLED: word order scrambled, N_SHUFFLES fixed seeds, results
  averaged. Each word is tokenised with its leading space and moved as
  a group, so intact and shuffled contain exactly the same tokens.
  The script asserts this.
- Each text is presented twice in a row (BOS, text, text). Second-pass
  tokens are predictable even when scrambled. This separates "hard to
  predict" from "incoherent", which is the confound Niamh raised.

Per token: C_adj; UP readout (unit residual projected on
UP-DOWN_stripped[L], word-built axis from Part A); VAL readout;
surprisal (minus log probability of the token given its prefix).
Aggregated to text level per pass.

Gates (a failed gate makes the dependent test UNINFORMATIVE, never a
pass and never a fail):
- G1 shuffling moves layer agreement: |d_z| of C_adj intact vs shuffled
  >= GATE_D over the 120 sentences. Direction is reported, not assumed.
  Shuffling makes the *text* incoherent; the hypothesis is about
  agreement *inside the model*; G1 is the check that one produces the
  other.
- G2 valence manipulation worked: VAL readout happy > sad, d >= GATE_D.
- G3 repetition lowers surprisal: shuffled second pass mean surprisal
  <= 0.5 x shuffled first pass. If it fails, the perplexity control
  falls back to regression only and is labelled weaker.

Tests (first pass unless stated; each at 5 layers, majority rule):
- T1 UP readout, intact vs shuffled, within happy and within sad.
- T2 the crossing cells: sad-intact vs happy-shuffled.
- T3 text-level regression over every (sentence x order x pass) cell:
  UP readout on z(C_adj) + z(surprisal) + valence. Standardised betas,
  95% CI by bootstrap over sentences.

## RULE PARAMETERS (frozen; asserted against this file at runtime)

  COUPLE_HI = 0.30   COUPLE_LO = 0.10   LAYERS_MAJ = 3 (of 5)
  CARRIER_MIN = 0.30 NULL_PCT = 95      N_NULL = 100
  GATE_D = 0.50      EFFECT_D = 0.30    N_SHUFFLES = 5
  REPL_TOL = 0.02    N_PERM = 200       N_BOOT = 2000
  FLAT_D = 0.20

COUPLE_HI is lower than exp170's 0.50 on purpose. That figure was set
against an already-observed +0.70. No coupling has been observed here,
and exp170's own single-token variant read +0.05..+0.21 for BALANCE.

## Decision rule — Part A

- COUPLED_SPECIFIC: cos(UP, d_coh_single) >= +COUPLE_HI at >=
  LAYERS_MAJ layers, AND above the NULL_PCT percentile of |null| at
  those layers, AND UP rank 1 of 8 by |cos| at >= LAYERS_MAJ layers.
- COUPLED_GENERIC: coupling and null conditions met, but another
  schema couples more strongly. Layer agreement is a carrier several
  schemas ride, not UP's own (cf. exp165c).
- WEAK: cos in [COUPLE_LO, COUPLE_HI) at most layers, or d_coh_single
  and d_coh_resid disagree across COUPLE_LO.
- NULL: |cos| < COUPLE_LO at >= 4/5 layers in both clean variants.
- WRONG_SIGN: cos <= -COUPLE_HI at >= LAYERS_MAJ layers. Coupled, but
  the DOWN pole is the agreeing one. A miss for the hypothesis as
  stated, reported as its own finding.

## Decision rule — Part B (requires G1 and G2)

- UP_FOLLOWS_COHERENCE: T1 d_z >= EFFECT_D in the direction of higher
  C_adj, in both valences, AND T3 beta on C_adj has a CI excluding 0
  with surprisal in the model.
- STRONG_DISSOCIATION: the above, AND T2 shows the higher-agreement
  cell above the happier cell.
- PERPLEXITY_NOT_COHERENCE: T1 holds, but T3's C_adj CI includes 0
  once surprisal is in.
- UP_FOLLOWS_VALENCE_ONLY: T1 |d_z| < 0.20 in both valences while G2
  holds.
- MIXED: anything else, reported cell by cell without a label.

## Committed predictions (the drafting Claude's; Niamh's go beside them)

Calibration note carried from exp170: mechanism bets in this lab ran
0-for-8; replication-with-controls bets were the reliable kind. This
is a mechanism bet.

- P1 Part A COUPLED_SPECIFIC: **10%**
- P2 cos(UP, d_coh_single) >= +0.10 at >= 3/5 layers: **30%**
- P3 G1 passes in the predicted direction (intact has higher C_adj):
  **45%**. Scrambled text may give the layers less to do, which would
  raise agreement, the inaction problem above.
- P4 T1 holds in both valences: **35%**
- P5 STRONG_DISSOCIATION: **10%**
- P6 all-word C_adj is badly confounded, |corr(C_adj, token count)|
  >= 0.5: **60%**
- META: modal Part A outcome is WEAK or NULL: **~65%**

Niamh's odds: delegated. Her words, 16:36: "i dont know yet about the
odds. u can do the odds i'll just do the intuitions". The odds above
are the only ones on record; the intuitions are hers and are in
HYPOTHESIS_UP_grounding_2026-10-02.md.

## Integrity

- SYNTHETIC self-test BEFORE the model run, four planted worlds:
  (A) C driven purely by token count -> clean variants read NULL;
  (B) C coupled to planted UP only -> COUPLED_SPECIFIC;
  (C) C coupled to all eight schemas -> COUPLED_GENERIC;
  (D) no coupling -> NULL. The self-test calls the real `verdict`
  path (housekeeping debt from 10 Jul).
- Replication gate: reproduce exp170's cos(BALANCE, d_norm_orig) =
  +0.701..+0.784 within REPL_TOL per layer, or STOP: protocol drift,
  not a finding. The tolerance is registered here (the other 10 Jul
  housekeeping debt).
- Token-identity assertion for every shuffled text.
- Results written to disk as each word batch and each sentence
  finishes; the run resumes from what is there.
- Stimuli sha256 (exp174_stimuli.json, built 2026-10-02 16:39 IST, no
  model loaded):
  f50c7eff0f99a4e4c80e8c039c3ef3197f8a585aabe709cd93f7f020fffffc84

## What I will NOT do

- No changing a threshold, the primary measure, or the primary
  direction after seeing numbers.
- No reporting d_coh_orig without the clean variants beside it.
- No treating a failed gate as a result in either direction.
- No promoting C_straight to primary if C_adj comes back null. It is
  reported as secondary whatever it says.
- No reading Part B as evidence about the inside of the model if G1
  fails.

## Deviations and stubs

None at draft time. Any that arise go here, in the code as
`# STUB — not the real mechanism: <what's missing>`, and at the top of
the report to Niamh.

## FREEZE AMENDMENTS (16:42–16:50 IST, pre-data, pre-analysis-code)

Written after Niamh's "go" and before the freeze commit, while working
out how the rules would be computed. Niamh saw the draft above, not
these; they are listed in the first report to her. Every one makes the
test harder to pass or removes an ambiguity. None was prompted by data.

1. **Carrier sanity is cross-validated.** exp170's carrier check
   correlates a direction with the same words that built it. In 1024
   dimensions with ~220 words that can read high for pure noise. Here:
   20 seeded split-halves, direction built on one half, correlation
   measured on the other, averaged. CARRIER_MIN applies to this value.
   The in-sample value is printed beside it for comparison with
   exp170. (Whether exp170's own carriers were inflated is NOT tested
   here; flagged as a lead.)
2. **Shuffled-carrier null.** N_PERM permutations of z(C_adj) among the
   single-token words, each rebuilt into a direction; the headline
   cosine must beat the NULL_PCT percentile of |cos(UP, d_perm)|. This
   is in addition to the random-partition axis null, whose pole sizes
   are matched to the axis under test (14 / 13 for UP_literal).
3. **Part A rule, as computed** (per layer L, primary direction
   d_coh_single; sig_L means |cos_L| beats both nulls):
   a. UNINFORMATIVE if cross-validated carrier < CARRIER_MIN at >= 3/5
      layers.
   b. WRONG_SIGN if cos_L <= -COUPLE_HI and sig_L at >= LAYERS_MAJ.
   c. COUPLED if cos_L >= +COUPLE_HI and sig_L at >= LAYERS_MAJ;
      SPECIFIC if UP is rank 1 of 8 by |cos| at >= LAYERS_MAJ layers,
      otherwise GENERIC.
   d. NULL if |cos_L| < COUPLE_LO at >= 4/5 layers in both clean
      variants (single, resid), OR |cos_L| does not beat the
      shuffled-carrier null at >= 3/5 layers.
   e. WEAK otherwise.
   The eight schemas for ranking are UP_literal plus the other seven.
4. **Two extra Part A readouts, descriptive and non-binding:**
   (i) mean C_adj of UP-pole words against DOWN-pole words (Cohen's d,
   single-token anchors only): the plain-language version of the
   question. (ii) d_coh_heldout: the primary direction rebuilt with
   every UP / DOWN anchor word (both lists) left out, then its cosine
   with UP. This asks whether agreement learned from other words
   points along UP.
5. **Part B specifics.**
   - G1: one paired test over the 120 sentences, first pass.
   - G2: paired by triple (n = 40), intact, first pass, VAL readout
     happy minus sad, d_z >= GATE_D at >= LAYERS_MAJ layers.
   - T1 holds if, at >= LAYERS_MAJ layers, d_z >= EFFECT_D toward the
     higher-agreement order in BOTH happy and sad (n = 40 each).
   - T2 cells are chosen by G1's direction: if intact has the higher
     agreement, sad-intact against happy-shuffled; if shuffled has it,
     sad-shuffled against happy-intact. Paired by triple; holds at
     d_z >= EFFECT_D at >= LAYERS_MAJ layers.
   - T3 holds if the bootstrap CI for the C_adj beta is entirely above
     0 at >= LAYERS_MAJ layers (N_BOOT resamples of sentences).
   - UP_FOLLOWS_VALENCE_ONLY uses FLAT_D: |d_z| < FLAT_D in both
     valences at >= LAYERS_MAJ layers.
   - T0 (descriptive): UP readout happy against sad, intact, paired by
     triple. Neutral sentences enter G1 and T3; their T1 is reported
     descriptively.
6. **Self-test expectations, corrected.** World (A), where C is purely
   token count, must come out UNINFORMATIVE or NULL and never COUPLED
   (with a cross-validated carrier the single-token variant has no
   carrier at all). Part B gets its own planted worlds: UP follows
   agreement; UP follows valence only; UP follows surprisal; G1 fails.
   The `compute_C` function gets a unit test on planted trajectories.
7. **Replication gate targets** (exp170_output.txt, d_norm_orig):
   L4 +0.784, L8 +0.729, L12 +0.701, L16 +0.718, L20 +0.729, each
   within REPL_TOL, computed on exp170's exact word set.
8. **Word set for Part A:** exp170's set plus exp141's literal UP /
   DOWN and valence lists. Anisotropy and frequency directions are
   computed on this full set.
9. **Git:** frozen on branch `exp174-up-grounding`, not `main`.
   CONSULT_LOG.md's uncommitted July changes are left untouched.

## POST-FREEZE LOG (pre-data; rules above unchanged)

Written 16:50 IST, 2026-10-02, after the freeze commit f2e30a4 (16:43:24)
and before any model run. Nothing in the rule sections was edited.

- Time correction: the FREEZE AMENDMENTS header says "16:42–16:50".
  The freeze commit landed at 16:43:24; the amendments were written
  16:42–16:43. The commit time is the authority.
- The analysis script `exp174_up_layer_agreement.py` was written after
  the freeze. Its synthetic self-test FAILED twice before passing. Each
  time the frozen rule was left alone and only the planted synthetic
  world was changed. What the failures showed:
  1. First version: the planted carrier was the dominant direction of
     the whole synthetic word cloud. There, any randomly weighted sum
     of word vectors points along it, so the shuffled-carrier null is
     as large as the true cosine. A truly coupled world read NULL, and
     an uncoupled world gave a raw cosine of 0.5 from noise alone
     (caught by the carrier gate). Kept as world E: uncoupled, dominant
     carrier, raw cosine −0.89, and the nulls correctly stop it reading
     COUPLED.
  2. Second version: UP words planted 2 sd above DOWN words in
     agreement (a very large effect) still read WEAK, because random
     word partitions reach |cos| ≈ 0.5 against the carrier direction.
  3. Passing version: modest carrier, UP words planted 3 sd above DOWN.
- **Consequence, stated plainly: the frozen Part A rule is strict. It
  says COUPLED only for a very large effect. A NULL or WEAK verdict
  does not rule out a moderate real effect.** A raw cosine on its own
  means little either way; only the comparison with the two nulls
  does.
- Because of that, two EXPLORATORY statistics were added to the script
  before any data existed. They are non-binding and can never change
  the verdict: (i) a label-shuffle p-value for the plain UP-vs-DOWN
  difference in agreement; (ii) across held-out single-token words,
  the correlation between a word's position on the UP axis and its
  agreement, with a permutation p-value.
- Lead for later, NOT tested here: exp154 / exp170 report cosines
  between a schema axis and a carrier direction without a
  shuffled-carrier null, and with an in-sample carrier check. The
  synthetic worlds above suggest both can read high by construction.
  Whether that touches the BALANCE–norm numbers is an open question
  for the decisive norm experiment.

## RESULT + GRADES (graded 16:54 IST, 2026-10-02, same session)

Integrity: self-test PASS; stimuli sha and rule constants asserted;
replication gate PASS (cos(BALANCE, d_norm_orig) +0.784 / +0.729 /
+0.701 / +0.718 / +0.729, matching exp170 to three decimals; 489
words, 219 single-token). Token-identity assertions passed for all 600
scrambles. Raw output: exp174_output.txt. Numbers: exp174_results.json,
exp174_rows.jsonl.

### Part A — frozen verdict: **NULL**
- cos(UP_literal, d_coh_single) by layer: +0.005, −0.015, +0.009,
  −0.005, −0.032. Shuffled-carrier null 95%: 0.12–0.17.
  Random-partition null 95%: 0.25–0.31. UP rank among the eight:
  7, 6, 7, 7, 7.
- The carrier itself is real: cross-validated +0.69, +0.62, +0.52,
  +0.50, +0.47 (in-sample +0.79..+0.64). So the NULL is not an
  uninformative instrument and not a borderline miss of a strict
  rule: the value is at zero.
- The control mattered. The uncontrolled all-word direction gives
  cos(UP_literal, d_coh_orig) = +0.23..+0.28 at every layer, and that
  direction is the token-count direction (cos(d_orig, d_tok) −0.80..
  −0.94). Residualised on token count: +0.11..+0.18. On token count
  and norm: −0.04..+0.03.
- UP_bundle vs d_coh_single: −0.03, −0.18, −0.11, −0.10, −0.14, none
  beyond its shuffled-carrier null (0.16–0.21).
- Exploratory, held-out words: corr(position on UP axis, C) between
  −0.08 and +0.04, p 0.33–0.84.
- Secondary carrier C_straight: +0.06..+0.09 at every layer, below its
  null (0.12–0.16). Reported, not promoted.
- Instrument: C_adj has a narrow range (single-token mean 0.894, sd
  0.007). corr(C, token count) −0.42 over all words. Among
  single-token words corr(C, norm) is +0.47..+0.49 at L4–L12 and
  cos(d_coh_single, d_norm_single) is +0.44..+0.56 at L4–L16: the
  agreement carrier is about half shared with the norm carrier.
  corr(C, path length) among single-token words −0.17 (inaction is
  not the main driver at word level).
- The "plain version" (UP words vs DOWN words) is UNUSABLE: only 3 UP
  and 5 DOWN literal anchors are single-token as bare words.
- Not tested, noted only: DIFFICULTY-BURDEN (−0.12..−0.25) and
  FORWARD-BACK (+0.11..+0.19) keep one sign across all five layers
  against d_coh_single. Neither was a registered question and neither
  clears the random-partition null.

### Part B — frozen verdict: **MIXED** (UP_bundle beside it: MIXED)
- G1 PASS, in the direction NOT predicted: scrambled text has HIGHER
  layer agreement than intact text (d_z −1.37; means 0.8922 vs
  0.8889). G2 PASS (d_z +1.4..+1.9). G3 PASS (scrambled surprisal 7.96
  first pass, 1.21 second pass).
- T1 (does UP rise with agreement, at fixed valence): holds at L8 and
  L12 only (happy +0.91 / +0.52, sad +1.12 / +0.81), fails at L4, L16,
  L20. 2 of 5 layers: False.
- T2 (higher-agreement cell above happier cell): False at every layer,
  d_z −1.0..−1.6. The happier cell wins.
- T3: beta on C_adj is NEGATIVE at every layer (−0.58..−0.91, CIs
  excluding 0). **Not interpretable as agreement lowering UP**: it is
  carried by the first-pass / second-pass difference. The repeat
  lowered agreement (d_z −5.1) and raised the UP readout (d_z +2.9..
  +6.6), and pass is confounded with position in the sequence. This
  is a flaw in the repeat control as designed.
- T0 (descriptive): happy sentences read more UP than sad ones on the
  purely spatial axis at every layer, d_z +1.09..+1.78 (UP_bundle:
  +2.1..+3.0), with no UP / DOWN / height word in any sentence.

### Post-hoc descriptive (exp174_posthoc.py; NOT registered)
- The crossing as Niamh phrased it (text coherence): sad-intact
  against happy-scrambled. Happy-scrambled reads more UP at every
  layer, d_z −0.84..−2.12. Happy / incoherent states exist and they
  read as UP.
- Happy and sad sentences have the same layer agreement (d_z 0.00)
  while differing strongly on the UP readout.
- Scrambled text is not "layers doing nothing": its path is longer
  (d_z −1.33 intact minus scrambled) and much straighter (−3.34).
- Within the first pass, across sentences, the change in agreement
  from scrambling correlates −0.24..−0.35 with the change in UP
  readout at L8–L20 (+0.10 at L4).

### Grades for the committed predictions
- P1 COUPLED_SPECIFIC (10%): did not occur.
- P2 cos >= +0.10 at >= 3/5 layers (30%): did not occur (0/5).
- P3 G1 passes with intact higher (45%): did not occur. G1 passed
  the other way round.
- P4 T1 holds in both valences (35%): did not occur (2/5 layers).
- P5 STRONG_DISSOCIATION (10%): did not occur.
- P6 |corr(C, token count)| >= 0.5 (60%): MISS (−0.42).
- META modal Part A outcome WEAK or NULL (~65%): HIT (NULL).

### What this does and does not say
- In Pythia 410M, with layer agreement measured as adjacent-layer
  cosine, UP is not coupled to it at word level, and at text level
  the UP readout goes with happy, not with agreement or coherence.
- One model, one definition of coherence, one stimulus set. GPT-2 and
  Llama ground BALANCE differently from Pythia, so UP could too.
- Per this prereg, exp175 (the causal arm) is designed only if exp174
  is not NULL. Part A is NULL, so it is not designed here.
- On the exp170 lead in the post-freeze log: in real data the
  in-sample carrier check overstated the cross-validated one by about
  0.1–0.17 here. Mild, and still untested on exp170 itself.

### Deviations and stubs
None in the code. Additions to the draft Niamh approved are all listed
under FREEZE AMENDMENTS and POST-FREEZE LOG, and were made before any
data existed.

## ERRATUM (17:20 IST, 2026-10-02): exp174 ran without a start token

Found while debugging exp175; details in START_TOKEN_BUG_2026-10-02.md.

- `model.to_tokens` does not prepend a start token for Pythia in this
  build. exp174's bare words had none. Its "single-token" group (227
  words) is really the TWO-token words; the 225 true one-token words
  were counted as zero and sat at sequence position 0, where the norm
  is about 45 times larger.
- Part A's primary direction used a constant-token-count group, so
  the token-count control still did its job. But the UP axes, the
  anisotropy direction and the centring means were all built from a
  mix of position-0 and later-position states. The "plain version"
  (3 UP words against 5 DOWN) compared two-token words only.
- The instrument checks that mention norm or token count (corr(C,
  token count) −0.42, corr(C, norm) +0.45..+0.53 over all words, and
  the +0.25 "token-count coupling") are position-0 effects in whole
  or in part.
- Part B: every text was "a" + sentence + sentence, not start token +
  sentence + sentence. All analysed tokens were at position 1 or
  later, so within-experiment comparisons hold, but the readout axes
  were the bent bare-word axes.
- The replication gate passed because exp170 has the same bug.

Standing of the verdicts: Part A NULL and Part B MIXED are what the
frozen rules returned on a bent instrument. They are not overturned
and not confirmed. exp175 re-tests the happy-reads-as-UP side result
(T0) with clean, in-sentence axes.
