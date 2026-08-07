"""
exp77 — Sanity checks before formalizing the 12-axis basis.

  PART A — cross-bleed: cos(ABSTRACT_CONCRETE, basis) and cos(MODAL_STATUS, basis)
           and cos(ABSTRACT, MODAL). Verify max |cos| < 0.35 (the project threshold).
  PART B — pole vocabulary of the first 10 PCs of deanisotropized GloVe.
           What is the "oracle 16pp gap" actually capturing as content?
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
    if not offs:
        return None
    raw = np.stack(offs).mean(axis=0)
    return unit(raw)


def build_R(wv):
    eq = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    valence = build_axis(wv, VALENCE_PAIRS)
    arousal = build_axis(wv, AROUSAL_PAIRS)
    r = eq - (eq @ valence) * valence
    r = r - (r @ arousal) * arousal
    return unit(r)


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")

print("Building current 10-axis basis...")
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

basis_10 = [("C", C), ("W", W), ("ATT", ATT), ("INT", INT), ("R", R),
            ("D", D), ("IO", IO), ("DV", DV), ("MB", MB), ("EV", EV)]


ABSTRACT_CONCRETE_PAIRS = [
    ("abstract", "concrete"),       ("theoretical", "practical"),
    ("conceptual", "physical"),     ("general", "specific"),
    ("idea", "object"),             ("principle", "instance"),
    ("intangible", "tangible"),     ("notion", "thing"),
    ("categorical", "particular"),  ("ideal", "material"),
]
MODAL_STATUS_PAIRS = [
    ("hypothetical", "actual"),     ("imagined", "observed"),
    ("fictional", "factual"),       ("counterfactual", "real"),
    ("possible", "actual"),         ("could", "can"),
    ("might", "is"),                ("simulated", "real"),
    ("speculative", "established"), ("theoretical", "empirical"),
]

print("Building ABSTRACT_CONCRETE and MODAL_STATUS candidates...")
ABS = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
MOD = build_axis(wv, MODAL_STATUS_PAIRS)


# ============================================================================
# PART A — cross-bleed
# ============================================================================
print("\n" + "=" * 72)
print("PART A — cross-bleed: cos(ABS, basis) and cos(MOD, basis)")
print("=" * 72)

print(f"\n{'axis':<6}  {'cos(ABS, .)':>14}  {'cos(MOD, .)':>14}")
print("  " + "-" * 38)
abs_cosines = {}
mod_cosines = {}
for name, vec in basis_10:
    ca = cos(ABS, vec)
    cm = cos(MOD, vec)
    abs_cosines[name] = ca
    mod_cosines[name] = cm
    print(f"{name:<6}  {ca:>+14.4f}  {cm:>+14.4f}")

abs_max = max(abs_cosines.items(), key=lambda kv: abs(kv[1]))
mod_max = max(mod_cosines.items(), key=lambda kv: abs(kv[1]))
print(f"\nMax |cos| with current basis:")
print(f"  ABS: {abs(abs_max[1]):.4f} (with {abs_max[0]})")
print(f"  MOD: {abs(mod_max[1]):.4f} (with {mod_max[0]})")
print(f"\nProject threshold: 0.35 — ABS {'PASSES' if abs(abs_max[1]) < 0.35 else 'FAILS'}; "
      f"MOD {'PASSES' if abs(mod_max[1]) < 0.35 else 'FAILS'}.")

print(f"\ncos(ABS, MOD) = {cos(ABS, MOD):+.4f}")
print(f"  (the two new axes' coupling — also should be < 0.35 if both pass)")


# ============================================================================
# PART A.2 — Pole vocabulary for ABS and MOD
# ============================================================================
print("\n" + "=" * 72)
print("PART A.2 — pole vocabulary for ABS and MOD")
print("=" * 72)

print("\nABSTRACT_CONCRETE positive pole (abstract side, top 12):")
for w, s in wv.similar_by_vector(ABS.astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")

print("\nABSTRACT_CONCRETE negative pole (concrete side, top 12):")
for w, s in wv.similar_by_vector((-ABS).astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")

print("\nMODAL_STATUS positive pole (hypothetical side, top 12):")
for w, s in wv.similar_by_vector(MOD.astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")

print("\nMODAL_STATUS negative pole (actual side, top 12):")
for w, s in wv.similar_by_vector((-MOD).astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")


# ============================================================================
# PART B — pole vocabulary for first 10 PCs of deanisotropized GloVe
# ============================================================================
print("\n" + "=" * 72)
print("PART B — pole vocabulary for first 10 PCs (deanisotropized GloVe)")
print("=" * 72)
print()
print("These are the 'oracle' axes we compared against — what do they actually")
print("represent semantically? Worth knowing to interpret the 16pp gap.")
print()

mu = wv.vectors.mean(axis=0)
all_vecs_da = wv.vectors - mu

# Use sample for PCA
np.random.seed(42)
sample_idx = np.random.choice(len(all_vecs_da), 50000, replace=False)
sample = all_vecs_da[sample_idx]

print("Computing PCA on 50K deanisotropized samples...")
U, S, Vt = np.linalg.svd(sample, full_matrices=False)
var_ratio = (S ** 2) / (S ** 2).sum()

for pc_i in range(10):
    pc = Vt[pc_i]
    print(f"\n--- PC{pc_i + 1} (variance: {var_ratio[pc_i] * 100:.2f}%) ---")
    print("  Positive pole (top 8):")
    for w, s in wv.similar_by_vector(pc.astype(np.float32), topn=8):
        print(f"    {w:25s}  {s:+.4f}")
    print("  Negative pole (top 8):")
    for w, s in wv.similar_by_vector((-pc).astype(np.float32), topn=8):
        print(f"    {w:25s}  {s:+.4f}")
    # Also check cosines with our basis (without ABS/MOD) to see if any
    # PC aligns with one of our existing axes
    pc_cosines = [(name, cos(pc, vec)) for name, vec in basis_10]
    pc_cosines.append(("ABS", cos(pc, ABS)))
    pc_cosines.append(("MOD", cos(pc, MOD)))
    top_align = max(pc_cosines, key=lambda kv: abs(kv[1]))
    print(f"  Strongest alignment with our basis: {top_align[0]} ({top_align[1]:+.3f})")


np.savez("/Users/macn/Documents/embeddingexp/exp77_results.npz",
         ABS=ABS, MOD=MOD,
         abs_cosines=np.array(list(abs_cosines.items()), dtype=object),
         mod_cosines=np.array(list(mod_cosines.items()), dtype=object),
         cos_ABS_MOD=cos(ABS, MOD))
print("\nSaved exp77_results.npz")
