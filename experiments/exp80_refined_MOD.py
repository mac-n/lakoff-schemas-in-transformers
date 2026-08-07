"""
exp80 — Refined MOD construction with cleaner anchors.

Original MOD had three problems:
  1. Words appearing in 2+ pairs ("actual" 2×, "real" 2×) — overweighted
  2. Huge-distribution function words (could, can, is) pulling compound-token noise
  3. "theoretical" cross-bleeding with ABS axis

Refined: 10 pairs, each word unique, no broad-distribution words, no ABS overlap.
Niamh added (imaginary, real) with the insight that this might be the foundational
primitive — REAL_IMAGINARY may be a deeper name for the axis than MODAL_STATUS.

Tests:
  1 — Build refined MOD, verify OOV
  2 — Cross-bleed: cos(MOD_refined, all 11 other basis axes incl ABS)
  3 — Pole vocabulary (does it cluster around real/imaginary specifically?)
  4 — Compare to original MOD: cos(MOD_refined, MOD_orig); did cleaning help?
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
    used = []
    missing = []
    offs = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
            used.append((a, c))
        else:
            missing.append((a, c, a in wv.key_to_index, c in wv.key_to_index))
    return unit(np.stack(offs).mean(axis=0)), used, missing


def build_R(wv):
    eq, _, _ = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    v, _, _ = build_axis(wv, VALENCE_PAIRS)
    a, _, _ = build_axis(wv, AROUSAL_PAIRS)
    r = eq - (eq @ v) * v
    r = r - (r @ a) * a
    return unit(r)


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")

print("Building current 11-axis basis (10 cognitive + ABS)...")
C, _, _   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
W, _, _   = build_axis(wv, TARGET_WEIGHT_PAIRS)
ATT, _, _ = build_axis(wv, ATTENTION_CLEAN_PAIRS)
INT, _, _ = build_axis(wv, INTENTION_CLEAN_PAIRS)
R   = build_R(wv)
D, _, _   = build_axis(wv, TARGET_SURPRISAL_PAIRS)
IO, _, _  = build_axis(wv, IN_OUT_MML_CLEAN)
DV, _, _  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
MB, _, _  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
EV, _, _  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)

ABSTRACT_CONCRETE_PAIRS = [
    ("abstract", "concrete"),       ("theoretical", "practical"),
    ("conceptual", "physical"),     ("general", "specific"),
    ("idea", "object"),             ("principle", "instance"),
    ("intangible", "tangible"),     ("notion", "thing"),
    ("categorical", "particular"),  ("ideal", "material"),
]
ABS, _, _ = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)

# Original MOD (for comparison)
MOD_ORIG_PAIRS = [
    ("hypothetical", "actual"),     ("imagined", "observed"),
    ("fictional", "factual"),       ("counterfactual", "real"),
    ("possible", "actual"),         ("could", "can"),
    ("might", "is"),                ("simulated", "real"),
    ("speculative", "established"), ("theoretical", "empirical"),
]
MOD_ORIG, _, _ = build_axis(wv, MOD_ORIG_PAIRS)

# Refined MOD
MOD_REFINED_PAIRS = [
    ("hypothetical", "actual"),
    ("imagined", "observed"),
    ("imaginary", "real"),              # Niamh's addition — may be THE primitive
    ("fictional", "factual"),
    ("counterfactual", "demonstrated"),
    ("speculative", "confirmed"),
    ("conjectural", "verified"),
    ("presumed", "proven"),
    ("notional", "materialized"),
    ("alleged", "documented"),
]

print("\n" + "=" * 72)
print("TEST 1 — Build refined MOD")
print("=" * 72)
MOD, used, missing = build_axis(wv, MOD_REFINED_PAIRS)
print(f"\nUsed {len(used)}/{len(MOD_REFINED_PAIRS)} pairs:")
for a, c in used:
    print(f"  ({a}, {c})")
if missing:
    print(f"\nMissing (OOV):")
    for a, c, in_a, in_c in missing:
        flag = []
        if not in_a:
            flag.append(f"'{a}' OOV")
        if not in_c:
            flag.append(f"'{c}' OOV")
        print(f"  ({a}, {c}) — {'; '.join(flag)}")


# ============================================================================
# TEST 2 — Cross-bleed against all 11 other basis axes
# ============================================================================
print("\n" + "=" * 72)
print("TEST 2 — cos(MOD_refined, basis)")
print("=" * 72)

basis = [("C", C), ("W", W), ("ATT", ATT), ("INT", INT), ("R", R),
         ("D", D), ("IO", IO), ("DV", DV), ("MB", MB), ("EV", EV),
         ("ABS", ABS)]

print(f"\n{'axis':<6}  {'cos(MOD_ref, .)':>16}  {'cos(MOD_orig, .)':>16}  delta")
print("-" * 60)
all_cs_ref = []
all_cs_orig = []
for name, vec in basis:
    cr = cos(MOD, vec)
    co = cos(MOD_ORIG, vec)
    all_cs_ref.append((name, cr))
    all_cs_orig.append((name, co))
    delta = cr - co
    marker = ""
    if name == "ABS" and abs(cr) < abs(co):
        marker = "  ← ABS cross-bleed reduced"
    print(f"{name:<6}  {cr:>+16.4f}  {co:>+16.4f}  {delta:+.4f}{marker}")

max_ref = max(all_cs_ref, key=lambda kv: abs(kv[1]))
max_orig = max(all_cs_orig, key=lambda kv: abs(kv[1]))
print(f"\nMax |cos| MOD_refined: {abs(max_ref[1]):.4f} with {max_ref[0]}")
print(f"Max |cos| MOD_orig:    {abs(max_orig[1]):.4f} with {max_orig[0]}")
print(f"\nProject threshold: 0.35 — MOD_refined "
      f"{'PASSES' if abs(max_ref[1]) < 0.35 else 'FAILS'}")


# ============================================================================
# TEST 3 — Pole vocabulary
# ============================================================================
print("\n" + "=" * 72)
print("TEST 3 — Pole vocabulary for refined MOD")
print("=" * 72)

print("\nMOD_refined positive pole (imaginary/hypothetical side, top 15):")
for w, s in wv.similar_by_vector(MOD.astype(np.float32), topn=15):
    print(f"  {w:25s}  {s:+.4f}")

print("\nMOD_refined negative pole (real/factual side, top 15):")
for w, s in wv.similar_by_vector((-MOD).astype(np.float32), topn=15):
    print(f"  {w:25s}  {s:+.4f}")


# ============================================================================
# TEST 4 — Compare refined to original
# ============================================================================
print("\n" + "=" * 72)
print("TEST 4 — Refined vs original MOD comparison")
print("=" * 72)

print(f"\ncos(MOD_refined, MOD_orig) = {cos(MOD, MOD_ORIG):+.4f}")
print("  (if > 0.7, they're substantially the same direction; refinement is incremental)")
print("  (if 0.3-0.7, partial overlap, refinement is substantive)")
print("  (if < 0.3, refined is a genuinely different direction)")

print(f"\ncos(MOD_refined, ABS) = {cos(MOD, ABS):+.4f}")
print(f"cos(MOD_orig,    ABS) = {cos(MOD_ORIG, ABS):+.4f}")
print(f"  Δ = {cos(MOD, ABS) - cos(MOD_ORIG, ABS):+.4f}")
print("  (cross-bleed with ABS should drop since we removed (theoretical, empirical))")


# ============================================================================
# TEST 5 — Real/imaginary semantic test
# ============================================================================
print("\n" + "=" * 72)
print("TEST 5 — Does refined MOD discriminate the real/imaginary pairs cleanly?")
print("=" * 72)
print()
print("Probe: pairs of words that semantically differ only in real/imaginary marking.")
print("If MOD captures the real/imaginary primitive, the imagined-side word should")
print("load positive on MOD and the real-side word should load negative.")
print()

probe_pairs = [
    ("imagination", "perception"),
    ("fantasy", "memory"),
    ("dream", "experience"),
    ("fiction", "history"),
    ("myth", "fact"),
    ("speculation", "observation"),
    ("vision", "witness"),
    ("supposition", "evidence"),
    ("conjecture", "data"),
    ("rumor", "report"),
]

print(f"{'imagined-side':<18} {'real-side':<18} {'cos(imag, MOD)':>15} {'cos(real, MOD)':>15} {'Δ':>8}")
print("-" * 78)
for imag, real in probe_pairs:
    if imag not in wv.key_to_index or real not in wv.key_to_index:
        print(f"  ({imag}, {real}) — one OOV")
        continue
    v_imag = unit(wv[imag])
    v_real = unit(wv[real])
    c_imag = cos(v_imag, MOD)
    c_real = cos(v_real, MOD)
    delta = c_imag - c_real
    marker = "  ← consistent" if delta > 0 else "  ← reversed!"
    print(f"{imag:<18} {real:<18} {c_imag:>+15.4f} {c_real:>+15.4f} {delta:>+8.4f}{marker}")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp80_results.npz",
         MOD_refined=MOD,
         MOD_orig_for_comparison=MOD_ORIG,
         cos_refined_orig=cos(MOD, MOD_ORIG),
         cos_refined_ABS=cos(MOD, ABS))
print("\nSaved exp80_results.npz")
