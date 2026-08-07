"""
exp85 — Anchor reconstruction of residualized axes.

Goal: find anchor pairs that directly construct C_resid, INT_resid, DV_resid,
REAL_IMAGINARY_resid via standard build_axis(pairs), so the basis is portable
without requiring the residualization procedure.

For each:
  1. Build original axis from current anchors
  2. Compute and inspect pole vocabulary of original vs residualized version
  3. Identify which original anchors are likely UD-pulling
  4. Propose an alternative anchor list
  5. Build reconstructed axis from alternative anchors
  6. Compute cos(reconstructed, residualized) and cos(reconstructed, UD)
  7. Verdict: clean reconstruction, partial, or failure
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
    ABSTRACT_CONCRETE_PAIRS, REAL_IMAGINARY_PAIRS,
)
from lakoff_canonical_vocabulary import IN_OUT_MML_CLEAN, UP_DOWN_MML
from exp52_target_axis_validation import VALENCE_PAIRS, AROUSAL_PAIRS


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(wv, pairs):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    used = [(a, c) for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    missing = [(a, c) for a, c in pairs
               if a not in wv.key_to_index or c not in wv.key_to_index]
    return unit(np.stack(offs).mean(axis=0)), used, missing


def residualize(axis, against):
    return unit(axis - (axis @ against) * against)


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")

print("Building UD + 4 trouble axes (original + residualized versions)...")
UD, _, _ = build_axis(wv, UP_DOWN_MML)

C_orig, _, _ = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
INT_orig, _, _ = build_axis(wv, INTENTION_CLEAN_PAIRS)
DV_orig, _, _ = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
RI_orig, _, _ = build_axis(wv, REAL_IMAGINARY_PAIRS)

C_resid = residualize(C_orig, UD)
INT_resid = residualize(INT_orig, UD)
DV_resid = residualize(DV_orig, UD)
RI_resid = residualize(RI_orig, UD)


def reconstruct_test(axis_name, pairs_alt, target_resid, target_orig):
    """Build axis from alternative pairs, compare to residualized target."""
    print(f"\n--- {axis_name} ---")
    rec, used, missing = build_axis(wv, pairs_alt)
    cos_with_resid = cos(rec, target_resid)
    cos_with_orig = cos(rec, target_orig)
    cos_with_ud = cos(rec, UD)
    print(f"  Pairs used: {len(used)}/{len(pairs_alt)}")
    if missing:
        print(f"  Missing: {missing}")
    print(f"  cos(reconstructed, residualized_target):  {cos_with_resid:+.4f}")
    print(f"  cos(reconstructed, original_axis):        {cos_with_orig:+.4f}")
    print(f"  cos(reconstructed, UD):                   {cos_with_ud:+.4f}")
    if cos_with_resid > 0.90:
        verdict = "CLEAN reconstruction"
    elif cos_with_resid > 0.80:
        verdict = "GOOD reconstruction"
    elif cos_with_resid > 0.70:
        verdict = "PARTIAL — captures most but not all"
    else:
        verdict = "FAILS — anchor changes can't replicate residualization"
    print(f"  Verdict: {verdict}")
    return rec, cos_with_resid, cos_with_ud


# ============================================================================
# C_resid: try removing (prospering, declining) which is explicitly vertical
# ============================================================================
print("\n" + "=" * 78)
print("C_resid reconstruction attempts")
print("=" * 78)
print(f"\nOriginal C-UD overlap: {cos(C_orig, UD):+.4f}")
print(f"Target: C_resid (orthogonal to UD by construction)")

# Show pole vocab of C_resid for reference
print("\nC_resid positive pole (top 10):")
for w, s in wv.similar_by_vector(C_resid.astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")
print("\nC_resid negative pole (top 10):")
for w, s in wv.similar_by_vector((-C_resid).astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")

# Attempt 1: drop (prospering, declining) — most explicitly vertical
C_alt1 = [p for p in TARGET_REWARD_COMPOSITE_PAIRS if p != ("prospering", "declining")]
reconstruct_test("C attempt 1: drop (prospering, declining)",
                 C_alt1, C_resid, C_orig)

# Attempt 2: drop (prospering, declining) + (fulfilled, ruined) [ruined has fall sense]
C_alt2 = [p for p in TARGET_REWARD_COMPOSITE_PAIRS
          if p not in [("prospering", "declining"), ("fulfilled", "ruined")]]
reconstruct_test("C attempt 2: drop (prospering, declining), (fulfilled, ruined)",
                 C_alt2, C_resid, C_orig)

# Attempt 3: focus on non-vertical wellbeing
C_alt3 = [
    ("flourishing", "suffering"),
    ("thriving", "struggling"),
    ("blessed", "cursed"),
    ("fortunate", "unfortunate"),
    ("graced", "plagued"),
    ("charmed", "hexed"),
    ("favored", "disfavored"),
    ("lucky", "unlucky"),
    ("wholesome", "broken"),
    ("cherished", "despised"),  # new
    ("treasured", "abandoned"),  # new
]
reconstruct_test("C attempt 3: non-vertical wellbeing terms (with 2 new pairs)",
                 C_alt3, C_resid, C_orig)


# ============================================================================
# INT_resid: original anchors aren't obviously vertical
# ============================================================================
print("\n" + "=" * 78)
print("INT_resid reconstruction attempts")
print("=" * 78)
print(f"\nOriginal INT-UD overlap: {cos(INT_orig, UD):+.4f}")

print("\nINT_resid positive pole (top 10):")
for w, s in wv.similar_by_vector(INT_resid.astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")
print("\nINT_resid negative pole (top 10):")
for w, s in wv.similar_by_vector((-INT_resid).astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")

# Attempt 1: drop (aiming, drifting) — "aiming" has vertical sense ("aim high")
INT_alt1 = [p for p in INTENTION_CLEAN_PAIRS if p != ("aiming", "drifting")]
reconstruct_test("INT attempt 1: drop (aiming, drifting)",
                 INT_alt1, INT_resid, INT_orig)

# Attempt 2: focus on planning/commitment without rise-coded vocab
INT_alt2 = [
    ("planning", "improvising"),
    ("deciding", "deferring"),
    ("committing", "hedging"),
    ("choosing", "defaulting"),
    ("designing", "improvising"),
    ("resolving", "postponing"),
    ("scheduling", "winging"),
    ("plotting", "freelancing"),
    ("intending", "drifting"),
    ("intending", "stumbling"),
    ("plan", "improvise"),
]
reconstruct_test("INT attempt 2: minus (aiming, drifting)",
                 INT_alt2, INT_resid, INT_orig)

# Attempt 3: same as INT_CLEAN but also drop (committing, hedging)
# "committing" might pull verticality via "stand by your commitment" etc
INT_alt3 = [p for p in INT_alt2 if p != ("committing", "hedging")]
reconstruct_test("INT attempt 3: also drop (committing, hedging)",
                 INT_alt3, INT_resid, INT_orig)


# ============================================================================
# DV_resid: anchors are decision-verbs, mostly non-vertical
# ============================================================================
print("\n" + "=" * 78)
print("DV_resid reconstruction attempts")
print("=" * 78)
print(f"\nOriginal DV-UD overlap: {cos(DV_orig, UD):+.4f}")

print("\nDV_resid positive pole (top 10):")
for w, s in wv.similar_by_vector(DV_resid.astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")
print("\nDV_resid negative pole (top 10):")
for w, s in wv.similar_by_vector((-DV_resid).astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")

# Attempt 1: try the original anchors as-is and see what cos(DV, DV_resid) IS
DV_alt1_test = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)[0]
print(f"\n  Note: cos(DV_orig, DV_resid) = {cos(DV_orig, DV_resid):+.4f}")
print(f"  (This is the maximum achievable from original anchors)")

# Attempt 2: drop (favored, excluded) — "favored" pulls UP (smile-upon)
DV_alt2 = [p for p in TARGET_DECISION_VERDICT_PAIRS
           if p not in [("favored", "excluded"), ("preferred", "overlooked")]]
reconstruct_test("DV attempt 2: drop (favored, excluded), (preferred, overlooked)",
                 DV_alt2, DV_resid, DV_orig)

# Attempt 3: stick to neutral verdict verbs
DV_alt3 = [
    ("selected", "rejected"),
    ("chose", "refused"),
    ("picked", "discarded"),
    ("admitted", "denied"),
    ("accepted", "declined"),
    ("kept", "removed"),
    ("chosen", "eliminated"),
    ("designated", "omitted"),
    ("highlighted", "neglected"),
]
reconstruct_test("DV attempt 3: minimal verdict-only set",
                 DV_alt3, DV_resid, DV_orig)


# ============================================================================
# REAL_IMAGINARY_resid: anchors are non-vertical
# ============================================================================
print("\n" + "=" * 78)
print("REAL_IMAGINARY_resid reconstruction attempts")
print("=" * 78)
print(f"\nOriginal REAL_IMAG-UD overlap: {cos(RI_orig, UD):+.4f}")

print("\nREAL_IMAG_resid positive pole (top 10):")
for w, s in wv.similar_by_vector(RI_resid.astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")
print("\nREAL_IMAG_resid negative pole (top 10):")
for w, s in wv.similar_by_vector((-RI_resid).astype(np.float32), topn=10):
    print(f"  {w:25s}  {s:+.4f}")

# Attempt 1: original anchors — see the ceiling
print(f"\n  Note: cos(RI_orig, RI_resid) = {cos(RI_orig, RI_resid):+.4f}")

# Attempt 2: drop a few that might be UD-laden distributionally
# "demonstrated, confirmed, verified, proven" all have a "stand up to scrutiny" sense
RI_alt2 = [
    ("hypothetical", "actual"),
    ("imagined", "observed"),
    ("imaginary", "real"),
    ("fictional", "factual"),
    ("speculative", "documented"),  # replace "confirmed" with "documented"
    ("conjectural", "established"),  # replace "verified" with "established"
    ("notional", "instantiated"),  # replace "materialized" with "instantiated"
    ("alleged", "recorded"),  # replace "documented" with "recorded"
    ("presumed", "witnessed"),  # replace "proven" with "witnessed"
    ("counterfactual", "actualized"),  # replace "demonstrated" with "actualized"
]
reconstruct_test("RI attempt 2: replace stand-up-to-scrutiny terms",
                 RI_alt2, RI_resid, RI_orig)


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 78)
print("SUMMARY — anchor reconstruction feasibility")
print("=" * 78)
print("""
The residualization removes the UD-projection from each axis. Anchor changes
work to the extent they shift the axis-direction away from UD via lexical
substitution. If the UD-content of an axis comes mostly from distributional
clustering (words "pulling" UD-content even if not vertical themselves),
anchor changes alone cannot fully replicate residualization.

The honest methodology framing:
- Either reconstructions cleanly approximate the residualized targets (cos > 0.9)
  and we can use anchor-reconstructions as the canonical axes
- Or reconstructions partially work (cos 0.8-0.9) and we accept "anchor pairs
  plus residualization against UD" as a two-step methodology
- The two-step methodology is still reproducible — both steps are deterministic
""")

np.savez("/Users/macn/Documents/embeddingexp/exp85_results.npz",
         C_resid=C_resid, INT_resid=INT_resid,
         DV_resid=DV_resid, RI_resid=RI_resid)
print("Saved exp85_results.npz")
