# PRE-REGISTRATION — exp165: CAUSAL steer of BALANCE → attention entropy
Written 2026-06-13, BEFORE any code or model run. (Claude Opus 4.8;
Niamh to review before execution.) Unblocked by exp166 (BALANCE↔entropy
confirmed causal-worthy in GPT-2 and Llama; NOT Pythia).

## Provenance / what this adds
exp160/161/164/166 established a robust CORRELATION in GPT-2-medium:
a token's BALANCE-axis projection (norm-orthogonalised) is negatively
coupled to its attention entropy, at every layer L0–L17, peak shallow
(L3 C2q ≈ −0.32 to −0.44). Correlation cannot say which way the arrow
runs, nor whether a third variable drives both. exp165 is the causal
test: INJECT the BALANCE direction into the residual stream and measure
whether attention entropy moves — with the magnitude confound controlled.

## The hypothesis, and the deadly alternative
- H (causal, direction-specific): pushing the residual along the
  norm-independent BALANCE direction CAUSES attention to become more
  peaked (lower entropy). Mechanistic licence: ‖q‖² = LN(x)ᵀWqᵀWq LN(x)
  reads residual DIRECTION; BALANCE direction → ‖q‖ → entropy.
- DEADLY ALTERNATIVE (the one the design must kill): adding ANY vector
  to the residual raises ‖resid‖, hence ‖q‖, hence changes entropy.
  "BALANCE steering moves entropy" could be a pure MAGNITUDE effect that
  any direction produces. The experiment is built to force these apart.

## Scope (tight, per causal-validation discipline)
GPT-2-medium ONLY. It is the clean 18-layer backbone with the
best-characterised shallow peak. Llama (also confirmed in exp166) is the
replication target for exp165b, NOT this prereg. Arrow B (manipulate
attention entropy → read BALANCE projection) is a future mirror, not
scoped here. One clean causal instance first.

## Design
1. PROMPTS: reuse the exp166 third set (40, frozen, checksum
   4d54ff4297bd7e2c). These never trained the steering; they are the
   substrate the forward pass runs on. (No new prompt freeze needed —
   we are not measuring a fresh correlation, we are intervening.)
2. STEER POINT (mechanism-matched): attention entropy at layer L is
   driven by resid_pre[L] = resid_post[L−1]. To causally drive entropy
   at the shallow PEAK (GPT-2 L3, strongest coupling), inject at
   blocks.2.hook_resid_post. PRIMARY: steer L_s=2, measure attn entropy
   at L=3. ROBUSTNESS: also steer L_s=1 (measure L2) and L_s=3 (measure
   L4). PROPAGATION (report only): attn entropy at L8/12/16.
3. DIRECTION: the BALANCE axis built at resid_post[L_s] exactly as
   exp161.build_layer_dirs (aniso+freq stripped, orthogonalised to
   d_norm_ho — the SAME bal_pn the correlation isolated). Unit vector.
   We steer the norm-INDEPENDENT BALANCE component on purpose.
4. INJECTION: add c · m · d_unit to ALL token positions at the steer
   hook (ActAdd; reuse exp17 make_hook). m = median per-token
   resid_post[L_s] norm on these prompts (so c is in natural-scale
   units). DOSE GRID c ∈ {−2,−1,−0.5,−0.25, 0, +0.25,+0.5,+1,+2}.
5. DV: mean per-query attention entropy (attn_entropy_lib, C5-excluded
   tokens) at the measured layer, per coefficient c. Also record the
   induced Δ‖resid‖ at each c (to show controls are magnitude-matched).

## Controls (the experiment IS the controls)
- RANDOM baseline (kills the deadly alternative): K=12 random unit
  directions in residual space, each aniso+freq-stripped and
  orthogonalised to d_norm exactly like BALANCE, steered on the SAME c
  grid at the SAME injection norm. They induce the SAME ‖resid‖ bump →
  the SAME mechanical entropy pressure. Their slope distribution is the
  magnitude-only null.
- OTHER-SCHEMA baseline (specificity): the 7 non-BALANCE schema axes
  (norm-orthogonalised), steered identically. Does BALANCE beat
  meaningful-but-different directions, or do all semantic directions
  move entropy alike?
- SIGN control: the design predicts a SIGNED monotone curve (+c lowers
  entropy, −c raises it). A symmetric ∪/∩ response (both signs same
  direction) would indicate a pure-magnitude artefact, not a directional
  push — logged as a diagnostic.
- COHERENCE/saturation guard: if the random controls themselves produce
  erratic or floored entropy at the extreme c (model breaking), restrict
  to the coherent c-range and log it; do not read slopes off a saturated
  regime. (We measure entropy on existing tokens — no generation — so
  this is a softer constraint than generative steering, but checked.)

## Primary estimand + statistic
- slope_BAL = OLS slope of (mean attn entropy at L3) on c, over the dose
  grid, on the BALANCE direction.
- null = slope distribution of the 12 random directions (matched norm).
- Prompt-level CLUSTER bootstrap (resample the 40 prompts, 1000 reps)
  for a 95% CI on slope_BAL AND on (slope_BAL − mean random slope).
- Report Spearman(c, entropy) for monotonicity.

## Decision rule
- CAUSAL_DIRECTION_CONFIRMED: slope_BAL < 0, monotone (Spearman<0, CI<0),
  AND slope_BAL more negative than the random-direction null — concretely
  slope_BAL below the MINIMUM of the 12 random slopes (or bootstrap CI on
  [slope_BAL − mean_random] excludes 0 on the negative side) AND steeper
  than a MAJORITY of the 7 other schemas. → BALANCE direction CAUSALLY
  and SPECIFICALLY steers attention entropy.
- GENERIC_MAGNITUDE: slope_BAL < 0 and monotone, but NOT distinguishable
  from the random-direction null. → the effect is the magnitude/‖q‖ push,
  not BALANCE-specific. The correlation stands as correlation; no causal
  DIRECTION claim. (This is a real, publishable negative — it bounds the
  claim.)
- NULL: slope_BAL ≈ 0 (CI includes 0). → no causal effect at this layer;
  the correlation is not generated by BALANCE→entropy here (entropy→
  BALANCE, or common cause). Pivots interest to Arrow B.
- INVALID: random controls erratic/saturated across the whole grid →
  shrink c, rerun. Fix before reading slope_BAL.

## Committed predictions (calibration: this is a CAUSAL/mechanism claim;
the project's MECHANISM bets are 0-for-8/9, so the deflationary prior
applies HARDER here than to exp166's replication)
- P1 (BALANCE moves entropy in predicted sign): bet MODERATE (~0.65).
  The correlation is rock-solid and the mechanism is motivated, but
  causal ≠ correlational — the coupling could be downstream-generated,
  not driven at L2→L3.
- P2 (SPECIFICITY — BALANCE beats matched-magnitude RANDOM): bet
  DEFLATIONARY (~0.40). THIS is where the experiment lives and where
  mechanism claims die. I expect a genuine chance the entropy move is
  mostly the norm bump that any push gives. If BALANCE clearly beats
  random, that is BIGGER news than I am betting — and it is the result
  that makes the paper's causal section.
- P3 (monotone dose-response, conditional on P1): bet STRONG.
- P4 (other schemas): I expect SOME other schemas (esp. orientation-
  cluster ones, exp167) also steer entropy; BALANCE need not be unique,
  but should rank among the steepest. No strong bet on uniqueness.
- Calibration note recorded: if P2 hits, update that "causal steering of
  a robustly-correlated, mechanistically-licensed direction" is a third
  bet-type distinct from mechanism-stories (which lose) and replication
  (which the post-hoc-band caveat governs).

## Integrity checks (before any model; conventions 2026-06-13)
- SYNTHETIC discrimination test FIRST: a toy linear "attention" where a
  known direction drives a known entropy-analogue plus a magnitude
  confound; confirm the slope-vs-random-null machinery (a) recovers a
  planted DIRECTIONAL effect as beating random, and (b) classifies a
  planted PURE-MAGNITUDE effect as GENERIC (BALANCE ≈ random). Check the
  synthetic test discriminates (exp164 lesson: a vacuous synthetic
  catches nothing).
- Reuse, don't re-type: build_layer_dirs (exp161), make_hook (exp17),
  attn_entropy_per_query (attn_entropy_lib), frozen prompts (exp166).
- Random directions seeded (SEED=165) and logged; same seed → same 12.

## What I will NOT do
- No reading slope_BAL without the random-direction null beside it.
- No quoting an entropy drop as "BALANCE causes it" if random directions
  drop it the same amount (that is GENERIC_MAGNITUDE, and I will say so).
- No widening the c grid post-hoc to find an effect; grid is frozen here.
- No promoting Llama/Arrow B into this experiment.
