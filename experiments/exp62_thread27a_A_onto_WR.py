"""
exp62 — Thread 2.7(a) — A onto (W, R), free stage.

Regress A on (W, R). Compute regression coefficients α, β and the residual
|A − αW − βR| / |A|.

Predictions (from BASIS_TESTS_TODO.md):
  α > 0 (W coefficient positive — pleasant-calm has low W, unpleasant-aroused
         has high W; so A and W point similar directions on this reading)
  β > 0 (R coefficient positive — pleasant-calm has high R, unpleasant-aroused
         has low R)
  |A - αW - βR| / |A| → residual magnitude relative to A

Interpretation thresholds:
  residual < 0.3 → A substantially decomposes into (W, R). Remaining piece
                   is what SELECTION is hypothesized to provide. Half of
                   Thread 2.7 confirmed before Thread 1.
  residual 0.3-0.5 → A is touched by W and R but has independent content.
                     SELECTION might pick up some.
  residual > 0.5 → A is largely its own thing. Thread 2.7 fails or needs reframing.

For context (BASIS_REFERENCE §2): raw cosines from the 7-axis basis are
  cos(A, W) = +0.233
  cos(A, R) = +0.113
So we expect both α and β positive. The question is the residual magnitude.
"""

import numpy as np

print("Loading exp60 basis...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


A = unit(basis_raw["A_aff"])
W = unit(basis_raw["W_wgt"])
R = unit(basis_raw["R_per"])

# ---------------------------------------------------------------------------
# Reconfirm starting cosines
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Starting cosines (sanity check against BASIS_REFERENCE)")
print("=" * 68)
print(f"  cos(A, W) = {cos(A, W):+.4f}   (BASIS_REFERENCE: +0.233)")
print(f"  cos(A, R) = {cos(A, R):+.4f}   (BASIS_REFERENCE: +0.113)")
print(f"  cos(W, R) = {cos(W, R):+.4f}   (BASIS_REFERENCE: +0.009)")

# ---------------------------------------------------------------------------
# Least-squares regression of A onto subspace spanned by (W, R)
# ---------------------------------------------------------------------------
# A ≈ αW + βR + residual. Find α, β minimizing |A - αW - βR|^2.
# Solve (M^T M) [α; β] = M^T A where M = [W, R] (columns).
# All vectors live in R^300.

print("\n" + "=" * 68)
print("Regression: A ≈ αW + βR + residual")
print("=" * 68)

M = np.stack([W, R], axis=1)  # (300, 2)
# Normal equation
gram = M.T @ M                # (2, 2)
target = M.T @ A              # (2,)
coeffs = np.linalg.solve(gram, target)
alpha, beta = coeffs

print(f"  α (W coefficient) = {alpha:+.4f}   (predicted: > 0)  "
      f"{'PASS' if alpha > 0 else 'FAIL'}")
print(f"  β (R coefficient) = {beta:+.4f}   (predicted: > 0)  "
      f"{'PASS' if beta > 0 else 'FAIL'}")

# Reconstruction and residual
A_hat = alpha * W + beta * R
residual = A - A_hat
res_mag = np.linalg.norm(residual)
A_mag = np.linalg.norm(A)
rel_residual = res_mag / A_mag

print(f"\n  |A|              = {A_mag:.4f}  (should be 1.000, A is unit-normalized)")
print(f"  |αW + βR|        = {np.linalg.norm(A_hat):.4f}")
print(f"  |A - αW - βR|    = {res_mag:.4f}")
print(f"  relative residual = {rel_residual:.4f}")

# ---------------------------------------------------------------------------
# Variance-explained framing
# ---------------------------------------------------------------------------
# Fraction of A's variance captured by (W, R) subspace:
#   explained = |A_hat|^2 / |A|^2  (since A is unit, this is |A_hat|^2)
# Residual fraction:
#   residual_frac = res_mag^2 / |A|^2
# These sum to 1 only if A_hat is the orthogonal projection (which it is
# under least-squares).

explained_frac = np.linalg.norm(A_hat)**2 / A_mag**2
residual_frac = res_mag**2 / A_mag**2

print("\n" + "=" * 68)
print("Variance decomposition")
print("=" * 68)
print(f"  A^2 = (αW + βR)^2 + residual^2")
print(f"  variance explained by (W, R) = {explained_frac:.4f} ({100*explained_frac:.1f}%)")
print(f"  variance in residual         = {residual_frac:.4f} ({100*residual_frac:.1f}%)")
print(f"  (sum: {explained_frac + residual_frac:.4f})")

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Verdict")
print("=" * 68)
if rel_residual < 0.3:
    verdict = ("STRONG: A substantially decomposes into (W, R). The remaining\n"
               "  component is what SELECTION (Thread 1) is hypothesized to provide.\n"
               "  Half of Thread 2.7 confirmed before Thread 1 is even built.")
elif rel_residual < 0.5:
    verdict = ("MODERATE: A is touched by W and R but has substantial independent\n"
               "  content. SELECTION might cover some of the residual.")
else:
    verdict = ("WEAK: A is largely its own thing. (W, R) don't capture much of A.\n"
               "  Thread 2.7 reading needs reframing or fails.")
print(f"  relative residual = {rel_residual:.3f}")
print(f"  {verdict}")

# ---------------------------------------------------------------------------
# What's in the residual? Project onto other basis axes for a hint.
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("What's in the residual? (cos with each remaining basis axis)")
print("=" * 68)
res_unit = residual / res_mag
print(f"  residual is unit-normalized; cosines below show what content is left")
print(f"  {'axis':>8s}  {'cos(residual, axis)':>20s}")
for name in ["C_rew", "G_pol", "D_cmp", "IO_blk"]:
    ax = unit(basis_raw[name])
    print(f"  {name:>8s}  {cos(res_unit, ax):+.4f}")

# Also reproduce what A itself looks like in the basis for comparison
print(f"\n  For reference — cos(A_aff, each basis axis):")
for name in ["C_rew", "W_wgt", "G_pol", "R_per", "D_cmp", "IO_blk"]:
    ax = unit(basis_raw[name])
    print(f"  {name:>8s}  {cos(A, ax):+.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez("exp62_results.npz",
         alpha=alpha, beta=beta,
         residual=residual,
         rel_residual=rel_residual,
         explained_frac=explained_frac,
         A=A, W=W, R=R)
print("\nSaved exp62_results.npz")
