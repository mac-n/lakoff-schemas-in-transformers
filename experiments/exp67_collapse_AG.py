"""
exp67 — Collapse A and G into one axis.

Premise: cos(A, G) = +0.56 is structurally irreducible (exp65). Combined
with the exp64 A3 finding that attention/focus/decision/judgment vocab
loads on G+A (not on DV), the substrate-real reading says A and G are
TWO ANCHOR-SETS PROBING ONE UNDERLYING PRIMITIVE — the precision-weighted
attentional commitment that constitutes gating-process in agents-with-affect.

If that's right, the move isn't disentangle, it's COLLAPSE.

Method:
  GATING_PROCESS = unit(A + G)                  ← shared direction
  COMPLEMENT     = unit(A - G), then orthogonalized against GATING_PROCESS
                                                 ← what's differential between A and G

Equivalently, PCA over the stack [A; G]:
  PC1 direction (eigenvalue 1+r) = unit(A + G)
  PC2 direction (eigenvalue 1-r) = unit(A - G) (orthogonal to PC1 since A, G unit)

Tests:
  1. Inter-axis cosines of the collapsed basis (C, W, GATING_PROCESS, R, D, IO,
     DV, MB) — does collapsing simplify the orthogonality structure?
  2. Pole vocabulary for GATING_PROCESS and COMPLEMENT.
  3. Concept-word battery: does GATING_PROCESS predict attention/focus/
     decision/judgment more cleanly than A or G alone?
  4. Clinical fear/anxiety/panic distinction: does it survive the collapse?
  5. Variance explained: how much of the original (A, G) 2D subspace lives
     in GATING_PROCESS alone (i.e., how much does dropping COMPLEMENT lose)?
"""

import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

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
    result = v.copy()
    for ax in axes:
        u = unit(ax)
        result = result - (result @ u) * u
    return result


C  = unit(basis_raw["C_rew"])
W  = unit(basis_raw["W_wgt"])
A  = unit(basis_raw["A_aff"])
G  = unit(basis_raw["G_pol"])
R  = unit(basis_raw["R_per"])
D  = unit(basis_raw["D_cmp"])
IO = unit(basis_raw["IO_blk"])
DV = unit(exp63["SELECTION"])
MB = unit(exp64["MARKOV_BLANKET"])

# ---------------------------------------------------------------------------
# Build collapsed axis and complement
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Building the collapsed axes")
print("=" * 68)

r = float(A @ G)  # raw cosine
print(f"  cos(A, G) = {r:+.4f}  (the entanglement)")

GATING_PROCESS = unit(A + G)         # eigenvalue 1+r in PCA of stack[A;G]
DIFF_raw = A - G
DIFF = unit(project_out(DIFF_raw, GATING_PROCESS))  # complement axis, orthogonal to GP

print(f"  |A + G|              = {np.linalg.norm(A + G):.4f}")
print(f"  |A - G|              = {np.linalg.norm(A - G):.4f}")
print(f"  GATING_PROCESS = unit(A + G)")
print(f"  COMPLEMENT     = unit(A - G), GS-orthogonalized against GATING_PROCESS")
print(f"  Verify orthogonality: cos(GP, COMPLEMENT) = "
      f"{cos(GATING_PROCESS, DIFF):+.6f}")
print(f"  cos(GP, A)  = {cos(GATING_PROCESS, A):+.4f}  "
      f"(should be sqrt((1+r)/2) = {np.sqrt((1+r)/2):.4f})")
print(f"  cos(GP, G)  = {cos(GATING_PROCESS, G):+.4f}")
print(f"  cos(COMPLEMENT, A)  = {cos(DIFF, A):+.4f}")
print(f"  cos(COMPLEMENT, G)  = {cos(DIFF, G):+.4f}")

# Variance accounted for
# The (A, G) 2D subspace has gram matrix [[1, r], [r, 1]]
# Eigenvalues are 1+r and 1-r; total trace = 2
# Fraction of (A,G)-subspace variance in GATING_PROCESS = (1+r)/2
gp_var_fraction = (1 + r) / 2
diff_var_fraction = (1 - r) / 2
print(f"\n  Variance partition (A, G) subspace:")
print(f"    GATING_PROCESS: {gp_var_fraction:.4f}  ({100*gp_var_fraction:.1f}%)")
print(f"    COMPLEMENT:     {diff_var_fraction:.4f}  ({100*diff_var_fraction:.1f}%)")
print(f"  Dropping COMPLEMENT loses {100*diff_var_fraction:.1f}% of A-G subspace.")

# ---------------------------------------------------------------------------
# Inter-axis cosines of the collapsed basis
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Collapsed-basis inter-axis cosines")
print("=" * 68)
print("New basis: C, W, GP (GATING_PROCESS), R, D, IO, DV, MB  (+ optional COMPLEMENT)")
print()

collapsed_basis = [
    ("C",  C), ("W",  W), ("GP", GATING_PROCESS), ("R",  R), ("D",  D),
    ("IO", IO), ("DV", DV), ("MB", MB),
]
names = [n for n, _ in collapsed_basis]
mat = np.array([[cos(v1, v2) for _, v2 in collapsed_basis] for _, v1 in collapsed_basis])

print(f"  {'':>6s}  " + "  ".join(f"{n:>7s}" for n in names))
for i, name in enumerate(names):
    row = "  ".join(f"{mat[i, j]:+.4f}" for j in range(len(names)))
    print(f"  {name:>6s}  {row}")

# Compare to original A-G entanglement: max off-diagonal magnitude
print("\n  Off-diagonal magnitude statistics:")
off_mag = [abs(mat[i, j]) for i in range(len(names))
                          for j in range(len(names)) if i != j]
print(f"    max |off-diag| = {max(off_mag):.4f}")
print(f"    mean |off-diag| = {np.mean(off_mag):.4f}")
print(f"  For comparison (original 7-axis): max was A-G at +0.561, "
      f"second was C-W at -0.344.")

# Show GP's cosines with everything
print(f"\n  GATING_PROCESS cosines:")
for name, v in collapsed_basis + [("COMPLEMENT", DIFF)]:
    if name == "GP":
        continue
    print(f"    {name:>10s}  {cos(GATING_PROCESS, v):+.4f}")

print(f"\n  COMPLEMENT cosines (the 'what's-different-between-A-and-G' axis):")
for name, v in collapsed_basis:
    print(f"    {name:>10s}  {cos(DIFF, v):+.4f}")

# ---------------------------------------------------------------------------
# Pole vocabulary
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Pole vocabulary — GATING_PROCESS")
print("=" * 68)
print("\n  POSITIVE pole (committed-aimed-aroused side):")
for w, s in wv.similar_by_vector(GATING_PROCESS.astype(np.float32), topn=15):
    print(f"    {w:25s}  {s:+.4f}")
print("\n  NEGATIVE pole (uncommitted-unfocused-calm side):")
for w, s in wv.similar_by_vector((-GATING_PROCESS).astype(np.float32), topn=15):
    print(f"    {w:25s}  {s:+.4f}")

print("\n" + "=" * 68)
print("Pole vocabulary — COMPLEMENT (what differentiates A from G)")
print("=" * 68)
print("\n  POSITIVE pole (A side > G side):")
for w, s in wv.similar_by_vector(DIFF.astype(np.float32), topn=15):
    print(f"    {w:25s}  {s:+.4f}")
print("\n  NEGATIVE pole (G side > A side):")
for w, s in wv.similar_by_vector((-DIFF).astype(np.float32), topn=15):
    print(f"    {w:25s}  {s:+.4f}")

# ---------------------------------------------------------------------------
# Concept-word battery
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Concept-word battery — does GP predict gating-process content?")
print("=" * 68)

PROBES = [
    # Gating-process vocabulary (from exp64 A3 — these loaded on G+A)
    "attention", "focus", "decision", "judgment", "judgement",
    "vigilance", "alertness", "salience",
    # Clinical fear/anxiety/panic distinction
    "anxiety", "fear", "panic", "rage", "trauma", "depression",
    "shame", "guilt",
    # Affective-only
    "joy", "love", "curiosity", "boredom", "awe",
    # Goal-only
    "ambition", "drive", "purpose", "goal", "motivation",
    # Should be weak on GP (controls)
    "sausage", "marigold", "stapler", "pebble",
]
print(f"  {'word':>14s}  {'cos(., GP)':>10s}  {'cos(., A)':>10s}  "
      f"{'cos(., G)':>10s}  {'cos(., COMP)':>12s}")
print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")
for w in PROBES:
    if w not in wv.key_to_index:
        print(f"  {w:>14s}  OOV")
        continue
    v = unit(wv[w])
    print(f"  {w:>14s}  {cos(v, GATING_PROCESS):+.4f}    "
          f"{cos(v, A):+.4f}    {cos(v, G):+.4f}    "
          f"{cos(v, DIFF):+.4f}")

# ---------------------------------------------------------------------------
# Does GP explain more of PC2 than G alone? (Need to load PCs or recompute.)
# Skipped here — would need exp40 input axes. Note as follow-up.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez("exp67_results.npz",
         GATING_PROCESS=GATING_PROCESS,
         COMPLEMENT=DIFF,
         gp_var_fraction=gp_var_fraction,
         collapsed_inter_axis=mat,
         collapsed_names=np.array(names))
print("\nSaved exp67_results.npz")
