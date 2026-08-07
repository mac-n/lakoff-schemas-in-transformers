"""
exp65 — Does MARKOV_BLANKET help disentangle A and G?

Direct test of Niamh's question: cos(A, G) = +0.56 (the persistent
affect-policy entanglement). If we project out MB, DV (formerly
SELECTION/GATING), or IO from both A and G, does the entanglement
drop?

If yes — the axis we project out was mediating the entanglement,
and adding it to the basis recovers cleaner A-vs-G separation.

If no — A-G entanglement is structural in a way none of these
candidate axes touch, and we need to look elsewhere.

Method:
  A' = A - (A·X)·X   (A with X-component removed)
  G' = G - (G·X)·X   (G with X-component removed)
  cos(A', G') — the A-G cosine in the subspace orthogonal to X

If X mediates the entanglement: cos(A', G') < cos(A, G) substantially.
If X is independent of the entanglement: cos(A', G') ≈ cos(A, G).

Also try joint residualization: project out (X, Y) jointly.
"""

import numpy as np

print("Loading exp60 basis + exp63 SELECTION + exp64 MB...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
exp63 = np.load("exp63_results.npz", allow_pickle=True)
exp64 = np.load("exp64_results.npz", allow_pickle=True)


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def project_out(v, *axes):
    """Project axes (each unit-norm) out of v. Sequential Gram-Schmidt."""
    result = v.copy()
    for ax in axes:
        ax_u = unit(ax)
        result = result - (result @ ax_u) * ax_u
    return result


A  = unit(basis_raw["A_aff"])
G  = unit(basis_raw["G_pol"])
IO = unit(basis_raw["IO_blk"])
C  = unit(basis_raw["C_rew"])
W  = unit(basis_raw["W_wgt"])
R  = unit(basis_raw["R_per"])
D  = unit(basis_raw["D_cmp"])
DV = unit(exp63["SELECTION"])  # decision-verdict (formerly mislabelled)
MB = unit(exp64["MARKOV_BLANKET"])

baseline = cos(A, G)

# ---------------------------------------------------------------------------
# Single-axis residualization
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print(f"Baseline: cos(A, G) = {baseline:+.4f}")
print("=" * 68)
print(f"\n  {'project out':>20s}  {'cos(A,X)':>10s}  {'cos(G,X)':>10s}  "
      f"{'cos(A_res, G_res)':>18s}  {'Δ':>8s}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*18}  {'-'*8}")

candidates = [
    ("MB (MARKOV_BLANKET)", MB),
    ("DV (decision-verdict)", DV),
    ("IO_CLEAN", IO),
    ("C_rew (reward)", C),
    ("W_wgt (cost)", W),
    ("R_per (precision)", R),
    ("D_cmp (compression)", D),
]

for label, X in candidates:
    A_res = project_out(A, X)
    G_res = project_out(G, X)
    new_cos = cos(A_res, G_res)
    delta = new_cos - baseline
    cax = cos(A, X)
    cgx = cos(G, X)
    print(f"  {label:>20s}  {cax:+.4f}    {cgx:+.4f}    "
          f"{new_cos:+.4f}              {delta:+.4f}")

# ---------------------------------------------------------------------------
# Joint residualization — multiple axes at once
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Joint residualization (project out multiple axes together)")
print("=" * 68)

joint_specs = [
    ("MB + DV",        [MB, DV]),
    ("MB + DV + IO",   [MB, DV, IO]),
    ("DV + C",         [DV, C]),
    ("MB + C",         [MB, C]),
    ("all 5 candidates", [MB, DV, IO, C, W]),
    ("ALL non-A/G basis", [C, W, R, D, IO, DV, MB]),
]
print(f"\n  {'project out':>26s}  {'cos(A_res, G_res)':>18s}  {'Δ':>8s}")
print(f"  {'-'*26}  {'-'*18}  {'-'*8}")
for label, axes in joint_specs:
    A_res = project_out(A, *axes)
    G_res = project_out(G, *axes)
    new_cos = cos(A_res, G_res)
    delta = new_cos - baseline
    print(f"  {label:>26s}  {new_cos:+.4f}              {delta:+.4f}")

# ---------------------------------------------------------------------------
# What's left after joint projection? Magnitudes of the residuals
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Residual magnitudes after projecting out everything except A, G")
print("=" * 68)
all_others = [C, W, R, D, IO, DV, MB]
A_res_all = project_out(A, *all_others)
G_res_all = project_out(G, *all_others)
print(f"  |A residual|      = {np.linalg.norm(A_res_all):.4f}  "
      f"(starts at 1.0; fraction kept = {np.linalg.norm(A_res_all):.3f})")
print(f"  |G residual|      = {np.linalg.norm(G_res_all):.4f}  "
      f"(fraction kept = {np.linalg.norm(G_res_all):.3f})")
print(f"  cos(A_res, G_res) = {cos(A_res_all, G_res_all):+.4f}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Verdict")
print("=" * 68)

A_res_mb = project_out(A, MB)
G_res_mb = project_out(G, MB)
mb_delta = cos(A_res_mb, G_res_mb) - baseline

if abs(mb_delta) < 0.02:
    print(f"  MB does NOT mediate the A-G entanglement (Δ = {mb_delta:+.4f}).")
    print(f"  The entanglement lives in a part of the space that MB doesn't touch.")
elif mb_delta < -0.1:
    print(f"  MB substantially mediates the A-G entanglement (Δ = {mb_delta:+.4f}).")
else:
    print(f"  MB partially mediates the A-G entanglement (Δ = {mb_delta:+.4f}).")

np.savez("exp65_results.npz",
         baseline_cos_AG=baseline,
         single_residuals={
             label: cos(project_out(A, X), project_out(G, X))
             for label, X in candidates
         })
print("\nSaved exp65_results.npz")
