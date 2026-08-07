"""
exp81 — Test ABS = MOD - SALIENCE hypothesis + rotation-imagining direct test.

PART A — ABS decomposition (Niamh's hypothesis):
  Does ABS reduce to a linear combination of MOD and ATT (salience analog)?
  If so, ABS is derivable rather than primitive — basis could drop back to 11.

  Tests: regress ABS onto (MOD, ATT), report coefficients, residual magnitude,
  signs (Niamh's specific prediction: β_ATT < 0, "abstract = imagined minus salient").
  Also compare to other subspace decompositions for context.

PART B — Rotation-imagining direct test:
  Compute mean(v(imagined_i) - v(real_i)) across probe pairs (the differences
  between matched real/imaginary word pairs). If MOD captures the rotation
  axis between real and imagined content, this delta vector should have high
  cosine with MOD.

  Predicted: cos(delta_vec, MOD) > 0.5 if MOD is genuinely the imagining-
  rotation direction. <0.3 if there are multiple unrelated dimensions
  separating real from imaginary content.
"""
import numpy as np
import gensim.downloader as api
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")

from project_axis_vocabulary import (
    TARGET_REWARD_COMPOSITE_PAIRS, TARGET_WEIGHT_PAIRS,
    ATTENTION_CLEAN_PAIRS, INTENTION_CLEAN_PAIRS,
    TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, TARGET_SURPRISAL_PAIRS,
    TARGET_DECISION_VERDICT_PAIRS, TARGET_MARKOV_BLANKET_PAIRS,
    TARGET_EPISTEMIC_VALUE_PAIRS,
    TARGET_SALIENCE_PAIRS,  # the deprecated original SALIENCE / A axis
)
from lakoff_canonical_vocabulary import IN_OUT_MML_CLEAN
from exp52_target_axis_validation import VALENCE_PAIRS, AROUSAL_PAIRS


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(wv, pairs):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    return unit(np.stack(offs).mean(axis=0))


def build_R(wv):
    eq = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    v = build_axis(wv, VALENCE_PAIRS)
    a = build_axis(wv, AROUSAL_PAIRS)
    r = eq - (eq @ v) * v
    r = r - (r @ a) * a
    return unit(r)


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")

print("Building all axes...")
C   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
W   = build_axis(wv, TARGET_WEIGHT_PAIRS)
ATT = build_axis(wv, ATTENTION_CLEAN_PAIRS)
INT = build_axis(wv, INTENTION_CLEAN_PAIRS)
R   = build_R(wv)
D   = build_axis(wv, TARGET_SURPRISAL_PAIRS)
IO  = build_axis(wv, IN_OUT_MML_CLEAN)
DV  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
MB  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
EV  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)

# The deprecated SALIENCE axis (= original A, before exp69 cleanup) —
# kept for reproducibility, useful as the "salience" Niamh is gesturing at.
SALIENCE_ORIG = build_axis(wv, TARGET_SALIENCE_PAIRS)

ABSTRACT_CONCRETE_PAIRS = [
    ("abstract", "concrete"),       ("theoretical", "practical"),
    ("conceptual", "physical"),     ("general", "specific"),
    ("idea", "object"),             ("principle", "instance"),
    ("intangible", "tangible"),     ("notion", "thing"),
    ("categorical", "particular"),  ("ideal", "material"),
]
ABS = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)

MOD_REFINED_PAIRS = [
    ("hypothetical", "actual"),
    ("imagined", "observed"),
    ("imaginary", "real"),
    ("fictional", "factual"),
    ("counterfactual", "demonstrated"),
    ("speculative", "confirmed"),
    ("conjectural", "verified"),
    ("presumed", "proven"),
    ("notional", "materialized"),
    ("alleged", "documented"),
]
MOD = build_axis(wv, MOD_REFINED_PAIRS)


# ============================================================================
# PART A — ABS decomposition: ABS onto (MOD, ATT)
# ============================================================================
print("\n" + "=" * 78)
print("PART A — Does ABS reduce to MOD + ATT?  (Niamh's hypothesis)")
print("=" * 78)


def regress_onto(target, basis_vecs):
    """OLS regression of target onto basis. Returns coeffs, residual, R^2."""
    B = np.stack(basis_vecs)  # (k, 300)
    # ABS = sum_i β_i * B_i + residual
    # least-squares: β = (B B^T)^-1 B target
    G = B @ B.T  # (k, k)
    b = B @ target  # (k,)
    coeffs = np.linalg.solve(G, b)
    fit = coeffs @ B  # (300,)
    residual = target - fit
    r2 = 1 - (residual @ residual) / (target @ target)
    return coeffs, residual, float(r2)


# ABS onto (MOD, ATT) — the hypothesis
print("\nABS regressed onto (MOD, ATT_CLEAN):")
print("  Hypothesis: ABS ≈ α·MOD + β·ATT  with β < 0  ('abstract = imagined minus salient')")
coeffs, res, r2 = regress_onto(ABS, [MOD, ATT])
res_norm = np.linalg.norm(res)
print(f"  α (MOD coefficient): {coeffs[0]:+.4f}")
print(f"  β (ATT coefficient): {coeffs[1]:+.4f}")
print(f"  residual magnitude:  {res_norm:.4f}  (R² = {r2:.4f})")
print(f"  variance of ABS explained by (MOD, ATT): {r2 * 100:.1f}%")

# Diagnose the sign
if r2 > 0.5:
    verdict = "SUPPORTED: ABS is largely in the (MOD, ATT) subspace"
elif r2 > 0.3:
    verdict = "PARTIAL: substantial ABS content in (MOD, ATT), but not all"
elif r2 > 0.15:
    verdict = "WEAK: small overlap with (MOD, ATT) subspace"
else:
    verdict = "REFUTED: ABS has independent content beyond (MOD, ATT)"
print(f"\n  Verdict on decomposition: {verdict}")

if coeffs[1] < 0:
    print(f"  Sign of β: NEGATIVE (matches Niamh's prediction)")
else:
    print(f"  Sign of β: POSITIVE (doesn't match the 'minus salience' framing)")


# Also try with original SALIENCE (the deprecated axis, broader than ATT)
print("\n\nABS regressed onto (MOD, SALIENCE_ORIG):")
print("  Using the original Russell-V×A-diagonal salience axis (deprecated A)")
coeffs2, res2, r2_2 = regress_onto(ABS, [MOD, SALIENCE_ORIG])
print(f"  α (MOD coefficient):      {coeffs2[0]:+.4f}")
print(f"  β (SALIENCE coefficient): {coeffs2[1]:+.4f}")
print(f"  residual magnitude:       {np.linalg.norm(res2):.4f}  (R² = {r2_2:.4f})")
print(f"  variance explained: {r2_2 * 100:.1f}%")


# Context: ABS onto MOD alone, ATT alone, and full 11-axis basis
print("\n\nFor context — ABS regressed onto various subspaces:")

subspaces = [
    ("MOD alone",           [MOD]),
    ("ATT alone",           [ATT]),
    ("SALIENCE_ORIG alone", [SALIENCE_ORIG]),
    ("(MOD, ATT)",          [MOD, ATT]),
    ("(MOD, SALIENCE_ORIG)", [MOD, SALIENCE_ORIG]),
    ("(MOD, ATT, INT)",     [MOD, ATT, INT]),
    ("(MOD, ATT, C, W)",    [MOD, ATT, C, W]),
    ("full 11-axis basis",  [C, W, ATT, INT, R, D, IO, DV, MB, EV, MOD]),
]

print(f"\n  {'subspace':<28}  {'R²':>8}")
print("  " + "-" * 42)
for name, basis in subspaces:
    _, _, r2_s = regress_onto(ABS, basis)
    print(f"  {name:<28}  {r2_s * 100:>6.1f}%")

# The diagnostic question: is ABS in the 11-axis basis (with MOD but no ABS)?
# If R^2 with full 11-axis is high (>0.6), ABS is derivable from the rest.
# If lower, ABS has substantial independent content even given the whole basis.


# ============================================================================
# PART B — Rotation-imagining direct test
# ============================================================================
print("\n" + "=" * 78)
print("PART B — Rotation-imagining direct test")
print("=" * 78)
print()
print("If imagining = rotation along the MOD axis, then v(imagined_X) − v(real_X)")
print("for matched semantic pairs should be consistently aligned with MOD.")

# Same probe pairs as exp80 + a few more for power
probe_pairs = [
    ("imagination", "perception"),
    ("fantasy",     "memory"),
    ("dream",       "experience"),
    ("fiction",     "history"),
    ("myth",        "fact"),
    ("speculation", "observation"),
    ("vision",      "witness"),
    ("supposition", "evidence"),
    ("conjecture",  "data"),
    ("rumor",       "report"),
    ("imaginary",   "real"),       # also includes an anchor pair, for sanity
    ("hypothetical", "actual"),    # also an anchor pair
    ("simulated",   "observed"),
    ("possible",    "actual"),
]

deltas = []
print(f"\n  {'imagined':<14}  {'real':<14}  |delta|     cos(delta, MOD)")
print("  " + "-" * 60)
for imag, real in probe_pairs:
    if imag not in wv.key_to_index or real not in wv.key_to_index:
        continue
    v_imag = wv[imag]
    v_real = wv[real]
    delta = v_imag - v_real
    delta_norm = np.linalg.norm(delta)
    c = cos(delta, MOD)
    deltas.append(delta)
    print(f"  {imag:<14}  {real:<14}  {delta_norm:>6.3f}    {c:>+.4f}")

# Mean delta vector
mean_delta = np.mean(deltas, axis=0)
mean_delta_norm = np.linalg.norm(mean_delta)
print(f"\n  Mean delta vector norm: {mean_delta_norm:.4f}")
print(f"  cos(mean_delta, MOD)  = {cos(mean_delta, MOD):+.4f}")
print(f"  cos(mean_delta, ABS)  = {cos(mean_delta, ABS):+.4f}")
print(f"  cos(mean_delta, ATT)  = {cos(mean_delta, ATT):+.4f}")
print(f"  cos(mean_delta, INT)  = {cos(mean_delta, INT):+.4f}")
print(f"  cos(mean_delta, C)    = {cos(mean_delta, C):+.4f}")

c_md_mod = cos(mean_delta, MOD)
print()
if c_md_mod > 0.7:
    print("  → STRONG: imagining ≈ rotation through MOD; the axis IS the imagining direction.")
elif c_md_mod > 0.5:
    print("  → MODERATE: imagining is substantially MOD-direction-aligned;")
    print("           some content lives off-axis.")
elif c_md_mod > 0.3:
    print("  → WEAK: imagining has MOD-aligned component but multiple directions involved.")
else:
    print("  → REFUTED: real/imaginary contrast lives in many directions, MOD is one of several.")

# Compare with what fraction lives in MOD vs the whole basis
basis_for_residualization = [C, W, ATT, INT, R, D, IO, DV, MB, EV, MOD, ABS]
basis_names = ["C", "W", "ATT", "INT", "R", "D", "IO", "DV", "MB", "EV", "MOD", "ABS"]
mean_delta_unit = unit(mean_delta)
print(f"\n  Mean_delta projection onto each basis axis:")
for n, v in zip(basis_names, basis_for_residualization):
    print(f"    {n:<6}  {cos(mean_delta_unit, v):+.4f}")


np.savez("/Users/macn/Documents/embeddingexp/exp81_results.npz",
         coeffs_MOD_ATT=coeffs,
         r2_MOD_ATT=r2,
         coeffs_MOD_SAL=coeffs2,
         r2_MOD_SAL=r2_2,
         mean_delta=mean_delta,
         cos_meandelta_MOD=cos(mean_delta, MOD),
         cos_meandelta_ABS=cos(mean_delta, ABS))
print("\nSaved exp81_results.npz")
