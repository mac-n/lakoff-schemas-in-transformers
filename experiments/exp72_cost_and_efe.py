"""
exp72 — target_COST (Thread 2) + C-W collapse experiment.

Two related questions about the cost-pole of expected free energy:
  Q1: Is W actually computational cost expressed through somatic vocabulary?
      Build target_COST from non-somatic anchors. cos(COST, W) tells us:
        > 0.80 → W IS cost-via-weight; rename W to COST in basis
        0.40 - 0.80 → mixed; somatic-weight is one modality of cost
        < 0.40 → genuinely separate primitives
  Q2: Should C and W collapse to a single EFE = unit(C - W) axis?
      cos(C, W) = -0.344 (largest off-diagonal in 9x9). Test whether the
      8-axis basis (EFE in place of C and W) explains Lakoff schemas about
      as well as the 9-axis basis does. If yes, basis drops to 8 axes.

Tests:
  1 — Build target_COST + pole-vocabulary sanity
  2 — Cross-bleed: cos(COST, basis) + cos(COST, DIFF)
  3 — 8-axis basis with EFE replacing C+W: schema decomposition
  4 — 9-axis basis with COST replacing W: schema decomposition
"""
import numpy as np
import gensim.downloader as api
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
)


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(pairs):
    offs, used, missing = [], [], []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
            used.append((a, c))
        else:
            missing.append((a, c))
    if not offs:
        return None, used, missing
    raw = np.stack(offs).mean(axis=0)
    return unit(raw), used, missing


# Load saved basis axes
print("Loading exp60 + exp64 + exp69 saved arrays...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
exp64 = np.load("exp64_results.npz", allow_pickle=True)
exp69 = np.load("exp69_results.npz", allow_pickle=True)

C   = unit(basis_raw["C_rew"])
W   = unit(basis_raw["W_wgt"])
R   = unit(basis_raw["R_per"])
D   = unit(basis_raw["D_cmp"])
IO  = unit(basis_raw["IO_blk"])
ATT = unit(exp69["ATTENTION_CLEAN"])
INT = unit(exp69["INTENTION_CLEAN"])
DV  = unit(exp64["GATING"])
MB  = unit(exp64["MARKOV_BLANKET"])


# ============================================================================
# TEST 1 — Build target_COST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 1 — Build target_COST (non-somatic anchors)")
print("=" * 72)

# Screened against W (no weight/burden), C (no flourish/suffer),
# and partial against ATT/INT.
COST_PAIRS = [
    ("expensive",  "free"),         # economic
    ("costly",     "cheap"),        # economic
    ("demanding",  "easy"),         # cognitive
    ("depleting",  "sustaining"),   # resource
    ("consuming",  "replenishing"), # resource
    ("extracting", "conserving"),   # resource
    ("effortful",  "trivial"),      # process — NOT effortless (W's neg pole)
    ("draining",   "renewing"),     # resource
    ("intensive",  "minimal"),      # quantity
]

COST, used, missing = build_axis(COST_PAIRS)
print(f"\nUsed {len(used)}/{len(COST_PAIRS)} pairs:")
for a, c in used:
    print(f"  ({a}, {c})")
if missing:
    print(f"\nMissing (OOV):")
    for a, c in missing:
        print(f"  ({a}, {c})  — "
              f"{a if a not in wv.key_to_index else ''}"
              f"{', ' if a not in wv.key_to_index and c not in wv.key_to_index else ''}"
              f"{c if c not in wv.key_to_index else ''}")

print("\nCOST positive pole (cost-side, top 12):")
for w_, s in wv.similar_by_vector(COST.astype(np.float32), topn=12):
    print(f"  {w_:25s}  {s:+.4f}")

print("\nCOST negative pole (no-cost side, top 12):")
for w_, s in wv.similar_by_vector((-COST).astype(np.float32), topn=12):
    print(f"  {w_:25s}  {s:+.4f}")


# ============================================================================
# TEST 2 — Cross-bleed: cos(COST, full 9-axis basis) + cos(COST, DIFF)
# ============================================================================
print("\n" + "=" * 72)
print("TEST 2 — Cross-bleed and the Q1 key number")
print("=" * 72)

basis_axes = [("C", C), ("W", W), ("ATT", ATT), ("INT", INT),
              ("R", R), ("D", D), ("IO", IO), ("DV", DV), ("MB", MB)]

print(f"\nCOST vs 9-axis basis:")
print(f"{'axis':<6}  {'cos(COST, .)':>14}")
for name, vec in basis_axes:
    val = cos(COST, vec)
    marker = "  ←" if name == "W" else ""
    print(f"{name:<6}  {val:>+14.4f}{marker}")

DIFF = build_axis(DIFFICULTY_BURDEN_MML)[0]
print(f"\nLakoff DIFF schema comparison:")
print(f"  cos(COST, DIFF) = {cos(COST, DIFF):+.4f}  (W-DIFF was +0.502 in exp60)")
print(f"  cos(W,    DIFF) = {cos(W, DIFF):+.4f}")

# ---- THE Q1 VERDICT ----
key_cw = cos(COST, W)
print(f"\n{'='*50}")
print(f"  Q1 VERDICT: cos(COST, W) = {key_cw:+.4f}")
print(f"{'='*50}")
if key_cw > 0.80:
    print("  → STRONG: W IS computational cost expressed via somatic vocabulary.")
    print("           Rename W to COST in the basis. Substantive evidence")
    print("           against strong embodied cognition for this primitive.")
    cost_w_verdict = "STRONG_REPLACE"
elif key_cw > 0.40:
    print("  → MIXED: W and COST overlap but aren't identical.")
    print("          Somatic-weight is one expressive modality of cost,")
    print("          not the whole thing. Schema decomposition in TEST 4")
    print("          will tell us whether to swap.")
    cost_w_verdict = "MIXED"
else:
    print("  → SEPARATE: W and COST are distinct primitives.")
    print("            Somatic-weight isn't just cost; it carries its own content.")
    cost_w_verdict = "SEPARATE"


# ============================================================================
# Gram-Schmidt helpers
# ============================================================================
def gs_orthogonalize(axes_dict, order):
    """Given dict of name→vector and an order, return list of GS-unit-vectors."""
    out = []
    for n in order:
        u = axes_dict[n].copy()
        for prev in out:
            u = u - (u @ prev) * prev
        u = unit(u)
        out.append(u)
    return out


def explained_in(v, gs_b):
    v_residual = v.copy()
    for u_gs in gs_b:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


# All Lakoff schemas + cluster axes
cluster_pairs = [
    ("VALENCE", VALENCE_PAIRS), ("AROUSAL", AROUSAL_PAIRS),
    ("COH", COHERENCE_PAIRS), ("SUC", SUCCESS_FAILURE_PAIRS),
    ("LOSS", LOSS_PAIRS),
]
lakoff_pairs = [
    ("UD", UP_DOWN_MML), ("FB", FORWARD_BACK_MML), ("LD", LIGHT_DARK_MML),
    ("PATH", PATH_MOTION_MML), ("EXIST", EXISTENCE_MML),
    ("FORCE", FORCE_MML), ("BAL", BALANCE_MML), ("DIFF", DIFFICULTY_BURDEN_MML),
]
all_schemas = {}
for name, pairs in cluster_pairs + lakoff_pairs:
    v, _, _ = build_axis(pairs)
    all_schemas[name] = v
all_schemas["IO_CLEAN"] = IO  # is a basis vector


# ============================================================================
# TEST 3 — 8-axis basis with EFE = unit(C - W) replacing C+W
# ============================================================================
print("\n" + "=" * 72)
print("TEST 3 — 8-axis basis with EFE = unit(C - W) replacing C+W")
print("=" * 72)

EFE = unit(C - W)

# 9-axis baseline GS basis (same as exp71)
gs_order_9 = ["C", "W", "MB", "D", "IO", "R", "DV", "ATT", "INT"]
gs_axes_9 = {"C": C, "W": W, "MB": MB, "D": D, "IO": IO,
             "R": R, "DV": DV, "ATT": ATT, "INT": INT}
gs_basis_9 = gs_orthogonalize(gs_axes_9, gs_order_9)

# 8-axis EFE
gs_order_8 = ["EFE", "MB", "D", "IO", "R", "DV", "ATT", "INT"]
gs_axes_8 = {"EFE": EFE, "MB": MB, "D": D, "IO": IO,
             "R": R, "DV": DV, "ATT": ATT, "INT": INT}
gs_basis_8 = gs_orthogonalize(gs_axes_8, gs_order_8)

# Verify EFE's cosines with the rest of the basis
print(f"\ncos(EFE, basis axes):")
for n in ["ATT", "INT", "R", "D", "IO", "DV", "MB"]:
    print(f"  EFE-{n:<4}  {cos(EFE, gs_axes_9[n]):+.4f}")

print(f"\nSchema decomp: 9-axis (with C, W) vs 8-axis (with EFE)")
print(f"{'schema':<10} {'9D expl':>8} {'8D-EFE expl':>12} {'Δ':>6}")
print("-" * 42)
deltas_8 = []
for sname, svec in all_schemas.items():
    e9 = explained_in(svec, gs_basis_9)
    e8 = explained_in(svec, gs_basis_8)
    d = (e8 - e9) * 100
    deltas_8.append(d)
    marker = "  ←lose>5%" if d < -5 else ("  ←gain>2%" if d > 2 else "")
    print(f"  {sname:<8}  {e9 * 100:>6.1f}%  {e8 * 100:>10.1f}%  {d:>+5.1f}%{marker}")

print(f"\nSummary: 8-axis EFE decomp loss")
print(f"  mean Δ = {np.mean(deltas_8):+.2f}%")
print(f"  median Δ = {np.median(deltas_8):+.2f}%")
print(f"  schemas losing >5%: {sum(1 for d in deltas_8 if d < -5)}/{len(deltas_8)}")
print(f"  worst loss: {min(deltas_8):+.2f}% on "
      f"{list(all_schemas.keys())[int(np.argmin(deltas_8))]}")
if np.mean(deltas_8) > -5:
    print("  → COLLAPSE LOOKS VIABLE: average loss < 5%, basis can drop to 8 axes.")
else:
    print("  → COLLAPSE LOSES INFORMATION: C and W carry independent content beyond EFE.")


# ============================================================================
# TEST 4 — 9-axis basis with COST replacing W
# ============================================================================
print("\n" + "=" * 72)
print("TEST 4 — 9-axis basis: COST replaces W")
print("=" * 72)

gs_order_9c = ["C", "COST", "MB", "D", "IO", "R", "DV", "ATT", "INT"]
gs_axes_9c = {"C": C, "COST": COST, "MB": MB, "D": D, "IO": IO,
              "R": R, "DV": DV, "ATT": ATT, "INT": INT}
gs_basis_9c = gs_orthogonalize(gs_axes_9c, gs_order_9c)

print(f"\nFirst check: cos(C, COST) = {cos(C, COST):+.4f}  (vs cos(C, W) = {cos(C, W):+.4f})")

print(f"\nSchema decomp: 9-axis-W vs 9-axis-COST")
print(f"{'schema':<10} {'with W':>8} {'with COST':>10} {'Δ':>6}")
print("-" * 40)
deltas_cost = []
for sname, svec in all_schemas.items():
    e9 = explained_in(svec, gs_basis_9)
    e9c = explained_in(svec, gs_basis_9c)
    d = (e9c - e9) * 100
    deltas_cost.append(d)
    marker = ""
    if sname == "DIFF":
        marker = "  ←DIFF (W's dominant schema)"
    elif d < -3:
        marker = "  ←lose>3%"
    elif d > 3:
        marker = "  ←gain>3%"
    print(f"  {sname:<8}  {e9 * 100:>6.1f}%  {e9c * 100:>8.1f}%  {d:>+5.1f}%{marker}")

print(f"\nSummary: COST-for-W swap")
print(f"  mean Δ = {np.mean(deltas_cost):+.2f}%")
print(f"  median Δ = {np.median(deltas_cost):+.2f}%")
print(f"  schemas gaining: {sum(1 for d in deltas_cost if d > 0)}/{len(deltas_cost)}")
print(f"  schemas losing: {sum(1 for d in deltas_cost if d < 0)}/{len(deltas_cost)}")


# Also: what does COST itself look like vs W in the rest of the basis?
print(f"\n9-axis-COST inter-axis cosines (the relevant top row):")
print(f"  cos(COST, MB)  = {cos(COST, MB):+.4f}")
print(f"  cos(COST, D)   = {cos(COST, D):+.4f}")
print(f"  cos(COST, IO)  = {cos(COST, IO):+.4f}")
print(f"  cos(COST, R)   = {cos(COST, R):+.4f}")
print(f"  cos(COST, DV)  = {cos(COST, DV):+.4f}")
print(f"  cos(COST, ATT) = {cos(COST, ATT):+.4f}")
print(f"  cos(COST, INT) = {cos(COST, INT):+.4f}")


# ============================================================================
# Save
# ============================================================================
np.savez("/Users/macn/Documents/embeddingexp/exp72_results.npz",
         COST=COST,
         EFE=EFE,
         cos_COST_W=cos(COST, W),
         cos_COST_DIFF=cos(COST, DIFF),
         cos_C_W=cos(C, W),
         deltas_8axis_EFE=np.array(deltas_8),
         deltas_9axis_COST=np.array(deltas_cost),
         schema_names=np.array(list(all_schemas.keys())))
print("\nSaved exp72_results.npz")
