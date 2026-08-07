# PRE-REGISTRATION — exp165b: ALL-LAYER causal steer of BALANCE → entropy
Written 2026-06-13, BEFORE any code/model run for the all-layer design.
(Claude Opus 4.8; Niamh's design call; review before execution.)

## Provenance / why all-layer
exp165 steered ONE shallow layer (resid_post[2]→attn L3) and got a
BORDERLINE result: slope_BAL −0.071 (monotone, Spearman −0.93), beat the
random-direction MEAN (~2σ, CI excluded 0) and all 7 schemas — but did
NOT beat the most extreme of 12 random directions, and FLIPPED SIGN one
layer over (L3→L4 +0.023). Also a rule-drift was caught (code AND vs
prereg OR). Verdict held as weak/fragile, not banked.

Niamh's two points reframe the design:
1. WITHIN-LAYER CORRELATION ≠ DOWNSTREAM CAUSAL CHANNEL. exp160–166
   measured "BALANCE-proj and entropy co-occur at layer L." exp165 asked
   "does injecting BALANCE at L_s change entropy downstream." A single
   shallow injection is too weak to answer — it is largely overwritten by
   the 20+ layers after it.
2. ORTHOGONALITY MAY BE NECESSARY. Pushing along what a token already
   encodes may do nothing; steering may only bite when the injected
   direction adds signal the stream was not carrying. All-layer injection
   along a consistent BALANCE direction is the way to accumulate that.

All-layer steering is also the intervention that historically MOVED this
project's models (exp17 UP→"reached for the stars", DARK→repetition
collapse). It is the principled, defensible strong test: if a BALANCE→
entropy causal channel exists anywhere, accumulating the push at every
layer should reveal it; if even that does nothing, the causal claim is
genuinely weak.

## Design
1. PROMPTS: exp166 frozen set (40, checksum 4d54ff4297bd7e2c).
2. INJECTION: at EVERY layer L = 0..23, add c · m_L · d_L to
   resid_post[L] simultaneously (all-layer ActAdd), where d_L = the
   norm-orthogonalised BALANCE axis at layer L (bal_pn from
   build_layer_dirs), m_L = median per-token resid_post[L] norm. Per-layer
   matched scale so c is a single global knob.
3. DOSE (gentler than exp165 — all-layer accumulates): c ∈
   {−0.5,−0.25,−0.1,−0.05, 0, +0.05,+0.1,+0.25,+0.5}. Coherence/saturation
   reported (global entropy range); if attention saturates (entropy
   floored/uniform) across the grid, restrict to the in-regime range and
   log it.
4. DV: per-query attention entropy (C5-excluded tokens) at EVERY layer.
   PRIMARY = global mean entropy (mean over all layers + tokens) vs c.
   Also report PER-LAYER slope (where does entropy actually respond?) —
   this is the direct read on Niamh's within-vs-downstream point.
5. Predicted sign: +c (more BALANCE) → LOWER global entropy (negative
   slope), per the correlation's sign.

## Controls
- RANDOM null: 16 all-layer random directions (each = a per-layer random
  unit vector ⊥ d_norm_L, matched per-layer scale m_L), same dose grid.
  This is the magnitude-matched null AND, being all-layer, the correct
  comparator for an all-layer BALANCE push.
- SCHEMA: the 7 non-BALANCE schema axes, all-layer, norm-orthogonalised.
- Random directions seeded (SEED=165) and logged.

## Primary estimand + CORRECTED decision rule (asserted vs this file)
- slope_BAL = OLS slope of global-mean entropy on c.
- null = 16 random all-layer slopes; mean_R, sd_R.
- Prompt-cluster bootstrap (1000 reps): CI on slope_BAL, and CI on
  (slope_BAL − mean_R).
- Spearman(c, global entropy) for monotonicity.

RULE PARAMETERS (frozen; code asserts these exact values):
  Z_THRESH = −1.64   (one-tailed outlier vs random cloud, p≈0.05)
  SCHEMA_MAJORITY = strictly more than half of 7  (i.e. ≥4)

- CAUSAL_DIRECTION_CONFIRMED:
    slope_BAL < 0 AND Spearman < 0 AND CI(slope_BAL) excludes 0
    AND z = (slope_BAL − mean_R)/sd_R < Z_THRESH        [outlier below cloud]
    AND CI(slope_BAL − mean_R) upper bound < 0          [reliably below mean]
    AND slope_BAL steeper than ≥4 of 7 schemas.
  (The brittle exp165 "beat the MIN of N randoms" bar is REPLACED by the
  proper z-outlier test — beating the min gets harder as N grows and is
  not a real significance criterion.)
- GENERIC_MAGNITUDE: real (CI excl 0), negative, monotone, but NOT a
  random-cloud outlier. The effect is the matched push, not BALANCE.
- NULL: slope_BAL CI includes 0, or non-negative, or non-monotone.
- INVALID: random controls saturate/erratic across the whole grid (model
  broken) — shrink c, rerun.

## Committed predictions (calibration: still a causal/mechanism claim,
deflationary prior; but a STRONGER intervention than exp165, so P2 up a
little)
- P1 (BALANCE moves global entropy, predicted sign, monotone): ~0.75.
- P2 (BALANCE is a random-cloud OUTLIER — the real test): ~0.50. The
  all-layer push accumulates direction-specific signal, so I rate this
  ~even, up from exp165's 0.40. If it confirms, that is the causal result.
- P3 (per-layer: the response concentrates DEEP, not at the shallow
  correlation peak): soft prediction, ~0.6 — exp165's propagation hint
  (deeper slopes were steeper) suggests the lever is not where the
  correlation peaks. No gate; diagnostic for the within-vs-downstream
  question.
- P4 (some orientation-cluster schemas also move entropy; BALANCE need
  not be unique but should rank among steepest): no strong bet.

## Integrity
- Decision-rule CONSTANTS asserted against this file's RULE PARAMETERS
  (the exp165 rule-drift must not recur).
- Reuse: build_layer_dirs/proj_out (exp161), attn_entropy_per_query
  (lib), ols_slope/spearman (exp165), frozen prompts (exp166).
- SYNTHETIC discrimination test before run (directional → outlier;
  pure-magnitude → not), using the corrected z-rule.

## What I will NOT do
- No reading slope_BAL without the random cloud beside it.
- No reverting to the "beat the min" bar to manufacture a confirmation,
  nor loosening Z_THRESH post-hoc.
- No widening the dose grid post-hoc to find an effect.
- Logged idea, NOT this experiment: explicit orthogonal-vs-aligned
  injection test (Niamh's orthogonality point) → exp165c if warranted.
