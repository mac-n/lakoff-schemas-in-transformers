# PRE-REGISTRATION — exp165c: is BALANCE↔entropy a GOLDILOCKS fold?
Written 2026-06-13, BEFORE the finer-grid run. (Claude Opus 4.8; Niamh's
reframe and design call; review before execution.)

## Provenance / the reframe
exp165b (all-layer steer) was scored with a LINEAR slope and returned
GENERIC_MAGNITUDE. But its BALANCE dose-curve was a ∪, not a line:
  c:    −0.50 −0.25 −0.10 −0.05  0.00 +0.05 +0.10 +0.25 +0.50
  ent:  0.763 0.519 0.448 0.416 0.397 0.393 0.420 0.497 0.604
Minimum at c≈+0.05 (balanced side), rising at BOTH extremes, asymmetric
(imbalanced side −0.5 = 0.763 > over-balanced +0.5 = 0.604). A linear
slope is BLIND to a ∪ (it averages the two walls) — so exp165b tested the
wrong shape and may have produced a FALSE NULL.

Niamh's reframe: BALANCE is the prototypical GOLDILOCKS / homeostatic
primitive (causal-validation plan §5.2) — the good state is the MIDDLE,
deviation either way is imbalance. A ∪ dose-response is therefore the
PREDICTED shape, not a saturation artefact, IF the bowl is BALANCE-
specific. And "pushing BALANCE too hard is unbalancing" = the axis
contains its own opposite (☯): over-pushing +BALANCE produces the high-
entropy IMBALANCE signature. A folded axis, not a ruler.

## Hypothesis under test
H: BALANCE has a GOLDILOCKS (∪ / folded) causal relationship with
attention entropy that is SPECIFIC — its dose-curve bowls more sharply,
with its floor shifted toward +balance, than matched-magnitude RANDOM
directions do. The deadly alternative: ALL directions bowl (generic
saturation — any big all-layer push scrambles attention symmetrically),
so BALANCE's ∪ is not special.

## Design
- Same as exp165b (all-layer ActAdd, GPT-2-medium, exp166 frozen prompts,
  norm-orthogonalised BALANCE per layer, matched per-layer scale m_L),
  EXCEPT a FINER, DENSER, SYMMETRIC dose grid to resolve the bowl:
    c ∈ {−0.50,−0.35,−0.25,−0.15,−0.10,−0.05, 0,
         +0.05,+0.10,+0.15,+0.25,+0.35,+0.50}   (13 points)
  Denser near the optimum to pin curvature + minimum location. The finer
  grid is a FRESH measurement (the exp165b ∪ was observed post-hoc; the
  curvature statistics below are pre-committed HERE before this grid runs).
- DV: global mean attention entropy (all layers), per direction, per c.
  STORE every direction's full curve (exp165b stored only slopes — the
  reason this re-run is needed).
- Controls: 16 all-layer RANDOM directions + 7 SCHEMA directions, matched.

## Pre-committed statistics (the RIGHT shape statistics)
For each direction fit quadratic  entropy ≈ a + b·c + k·c²:
1. CURVATURE k (the bowl). BALANCE-Goldilocks needs k_BAL > 0.
2. MINIMUM LOCATION: vertex c* = −b/(2k), AND empirical argmin of the
   curve. Goldilocks predicts c*_BAL > 0 (floor on the balanced side).
3. ASYMMETRY A = mean[ent(−0.25),ent(−0.35),ent(−0.50)]
                − mean[ent(+0.25),ent(+0.35),ent(+0.50)].
   Goldilocks predicts A_BAL > 0 (imbalanced side costs more).
4. ☯ DIAGNOSTIC (reported, not gated): ent at max +c vs baseline (c=0)
   and vs the imbalanced pole (max −c). Does over-balancing approach the
   imbalance signature?
Null cloud: same four quantities for the 16 random directions
(mean, sd). Prompt-cluster bootstrap (1000 reps) for CIs on k_BAL, c*_BAL,
A_BAL, and on (k_BAL − mean_random_k).

## RULE PARAMETERS (frozen; code asserts these exact values)
  CURV_Z   = +1.64   (k_BAL outlier ABOVE random-curvature cloud, p≈.05)
  ASYM_Z   = +1.0    (softer; asymmetry is corroborating, not primary)
  SCHEMA_MAJORITY = 4 (> 7/2)

## Decision rule
- GOLDILOCKS_CONFIRMED:
    k_BAL > 0 AND bootstrap CI(k_BAL) excludes 0          [real bowl]
    AND z_k = (k_BAL − mean_random_k)/sd_random_k > CURV_Z [bowl is SPECIFIC]
    AND CI(k_BAL − mean_random_k) lower bound > 0
    AND c*_BAL > 0 (vertex on balanced side; empirical argmin ≥ 0 too)
    AND k_BAL steeper-bowl than ≥4 of 7 schemas.
  → BALANCE causally folds attention entropy around a balanced optimum;
    the exp165b "GENERIC" was a wrong-statistic false null.
- GENERIC_SATURATION:
    k_BAL > 0 (bowls) but NOT a random-cloud outlier (z_k ≤ CURV_Z), i.e.
    random directions bowl just as much. → the ∪ is generic saturation,
    not a BALANCE-specific fold. The honest deflation of the ☯ reading.
- NOT_GOLDILOCKS: k_BAL ≤ 0 or CI(k_BAL) includes 0 (no reliable bowl).

## Committed predictions (calibration: shape-specific causal claim;
post-hoc-derived from exp165b so the SHAPE existing is near-certain, but
its SPECIFICITY vs random is the real, open test)
- P1 (BALANCE bowls, k_BAL > 0, CI excl 0): STRONG ~0.85 (seen in 165b).
- P2 (bowl is a random-cloud OUTLIER — THE test): MODERATE ~0.50. If big
  all-layer pushes saturate, random will bowl too and this comes back
  GENERIC_SATURATION. This is the crux and I am genuinely uncertain.
- P3 (vertex c*_BAL > 0, floor on balanced side): MODERATE ~0.55 (165b
  argmin was +0.05, weak).
- P4 (asymmetry A_BAL > 0, imbalanced side costs more): MOD-STRONG ~0.65
  (seen in 165b: −0.5 gave 0.763 vs +0.5 gave 0.604).
- ☯ (over-balancing raises entropy well above baseline): ~0.8 (already
  visible: +0.5 → 0.604 vs baseline 0.397). Reported, not gated.

## Integrity
- Rule constants asserted vs THIS file at runtime (exp165 drift fixed).
- SYNTHETIC test: plant (A) a Goldilocks bowl on a known direction +
  symmetric saturation on all → confirm BALANCE flagged CONFIRMED and a
  pure-symmetric-saturation world flagged GENERIC_SATURATION. Discriminate.
- Reuse exp165b machinery; quadratic fit added here.

## What I will NOT do
- No reading k_BAL without the random-curvature cloud beside it.
- No loosening CURV_Z, no switching to a different curve family post-hoc.
- The "all qualia on one folded axis / ☯" idea is the MOTIVATION, NOT a
  claim this experiment tests. This tests ONE instance (BALANCE↔entropy
  in GPT-2). Generalisation is a separate, much larger question.

---

## RESULT — exp165c (2026-06-13). exp165c_output.txt.
VERDICT (frozen rule): **GENERIC_SATURATION** — but the label MISDIRECTS;
see below. Harness validated (rule asserted vs prereg; synthetic
discriminated Goldilocks-world from saturation-world).

THE FOLD IS REAL, and BALANCE shows every Goldilocks signature:
- curvature k_BAL = +1.084 CI[+1.061,+1.105] (clean ∪);
- vertex c*_BAL = +0.053, empirical argmin +0.05 (floor on the BALANCED
  side, exactly as predicted);
- asymmetry A_BAL = +0.075, z_asym +1.10 (imbalanced side costs more);
- ☯ diagnostic: over-balanced (+0.5) entropy 0.604 vs baseline 0.397
  (+0.207, toward the imbalance signature). Pushing BALANCE too hard IS
  unbalancing — measured.

BUT IT IS NOT BALANCE-SPECIFIC:
- BALANCE BEAT RANDOM cleanly: z_k = +3.04 (random curvature mean +0.342,
  sd 0.245), CI(k_BAL − mean_random) [+0.731,+0.756]. So this is NOT
  saturation — a random push bowls ~3× less. The verdict word
  "SATURATION" is wrong; better name: SCHEMA-GENERAL FOLD.
- It failed the SCHEMA-majority bar (3/7): every schema folds hard —
  UP +0.999, IN +1.247, FORW +1.050, PATH +1.392, LIGHT +1.087,
  FORC +1.194, DIFF +0.908. Several bow MORE than BALANCE.

HONEST FINDING: **folding (Goldilocks, contains-own-opposite) is a
property of MEANINGFUL SEMANTIC DIRECTIONS as a class — not BALANCE
alone. Perturb any schema → attention destabilises with an optimum near
the natural state; random directions barely do (z=3 gap).** Niamh's ☯
intuition survives in GENERALISED form (primitives are folded, not flat);
the BALANCE-is-THE-ground specific form does not.

INTERPRETATION (marked speculative): all schema bowls appear to floor
near c≈0 (the natural state). If confirmed across schemas, the model rests
at a JOINT STABILITY EQUILIBRIUM across schema coordinates — "balance is
the ground" as the equilibrium property of the resting state, not one
axis. Testable: exp165d (schema vertices).

Calibration: P1 (bowls) HIT, P2 (beats random) HIT (z=3), P3 (vertex >0)
HIT, P4 (asymmetry) HIT. The UNPREDICTED thing — all schemas fold equally
— sank specificity. Fits the project spine: STRUCTURE (folding) robust +
general; SPECIFIC claim (BALANCE special) fails. Third time today.
