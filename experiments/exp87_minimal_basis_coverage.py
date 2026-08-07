"""
exp87 — Minimal-basis coverage tests.

How much of word-vector content is captured by just a few foundational axes?

  VALENCE alone (1 axis)
  UD alone (1 axis)
  VALENCE + UD (2 axes)
  VALENCE + UD + AROUSAL (3 axes — Russell affect + Lakoff verticality)

Comparison points:
  Random 1, 2, 3-axis baselines (sqrt(k/300) magnitude expected ≈ 5.8, 8.2, 10.0%)
  Cog-13 (~41% from exp86)
  Roget-13 (~39% from exp86)

If VALENCE + UD captures 25-35%, then 2 axes do a substantial fraction of
what 13 do — suggesting foundational organizational dimensions dominate.
If <20%, the 13-axis basis is doing genuinely distributed work that few-axis
bases can't replicate.

Tests:
  1 — Individual axis coverage
  2 — Joint 2-axis coverage (VALENCE + UD)
  3 — Joint 3-axis coverage (VALENCE + UD + AROUSAL)
  4 — Residual after VALENCE+UD: what content do the other axes uniquely add?
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
    return unit(np.stack(offs).mean(axis=0))


def build_R(wv):
    eq = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    v = build_axis(wv, VALENCE_PAIRS)
    a = build_axis(wv, AROUSAL_PAIRS)
    r = eq - (eq @ v) * v
    r = r - (r @ a) * a
    return unit(r)


def gs_orthogonalize(axes):
    out = []
    for v in axes:
        u = v.copy()
        for p in out:
            u = u - (u @ p) * p
        n = np.linalg.norm(u)
        if n < 1e-10:
            continue
        out.append(u / n)
    return out


def coverage_in(v, gs_basis):
    v_residual = v.copy()
    for u_gs in gs_basis:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


def residualize(axis, against):
    return unit(axis - (axis @ against) * against)


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)


print("Building axes...")
VALENCE = build_axis(wv, VALENCE_PAIRS)
AROUSAL = build_axis(wv, AROUSAL_PAIRS)
UD = build_axis(wv, UP_DOWN_MML)

print(f"\ncos(VALENCE, UD)      = {cos(VALENCE, UD):+.4f}")
print(f"cos(VALENCE, AROUSAL) = {cos(VALENCE, AROUSAL):+.4f}")
print(f"cos(UD, AROUSAL)      = {cos(UD, AROUSAL):+.4f}")

# Cog-13 basis for reference
C = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
W = build_axis(wv, TARGET_WEIGHT_PAIRS)
ATT = build_axis(wv, ATTENTION_CLEAN_PAIRS)
INT = build_axis(wv, INTENTION_CLEAN_PAIRS)
R = build_R(wv)
D = build_axis(wv, TARGET_SURPRISAL_PAIRS)
IO = build_axis(wv, IN_OUT_MML_CLEAN)
DV = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
MB = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
EV = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)
ABS = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
REAL_IMAG = build_axis(wv, REAL_IMAGINARY_PAIRS)

basis_13b = [UD, residualize(C, UD), W, ATT, residualize(INT, UD), R, D, IO,
             residualize(DV, UD), MB, EV, ABS, residualize(REAL_IMAG, UD)]
gs_cog13 = gs_orthogonalize(basis_13b)


# Build minimal bases
gs_VAL_only = gs_orthogonalize([VALENCE])
gs_UD_only = gs_orthogonalize([UD])
gs_VAL_UD = gs_orthogonalize([VALENCE, UD])
gs_VAL_UD_AR = gs_orthogonalize([VALENCE, UD, AROUSAL])


# Random bases (multiple seeds, anisotropy-orthogonal)
def make_random_bases(k, n_seeds=20):
    bases = []
    np.random.seed(0)
    mu_unit = unit(mu)
    for s in range(n_seeds):
        dirs = np.random.randn(k, 300)
        dirs = dirs - (dirs @ mu_unit)[:, None] * mu_unit
        bases.append(gs_orthogonalize([dirs[i] for i in range(k)]))
    return bases


random_1 = make_random_bases(1)
random_2 = make_random_bases(2)
random_3 = make_random_bases(3)


def get_deanisotropized(word):
    if word not in wv.key_to_index:
        return None
    v = wv[word] - mu
    n = np.linalg.norm(v)
    if n < 1e-10:
        return None
    return v / n


test_categories = {
    "EMOTIONS_AFFECTIVE": [
        "happiness", "sadness", "anger", "envy", "jealousy", "pride", "humility",
        "contentment", "longing", "delight", "melancholy", "rage", "elation",
    ],
    "AGENTIVE_STATES": [
        "ambition", "determination", "resignation", "willpower", "discipline",
        "procrastination", "perseverance", "complacency", "vigilance",
    ],
    "EPISTEMIC_STATES": [
        "knowledge", "ignorance", "belief", "doubt", "uncertainty",
        "conviction", "skepticism", "confidence", "speculation",
    ],
    "CONCRETE_NOUNS": [
        "chair", "table", "dog", "stone", "tree", "river", "mountain",
        "hammer", "rope", "lamp",
    ],
    "ABSTRACT_FORMAL": [
        "theorem", "philosophy", "ontology", "epistemology", "axiom",
        "principle", "framework", "paradigm",
    ],
    "MODAL_HYPOTHETICAL": [
        "hypothetical", "imaginary", "fictional", "speculative", "conjectural",
        "perhaps", "supposedly",
    ],
    "PHYSICAL_PROPERTIES": [
        "heavy", "light", "warm", "cool", "loud", "quiet", "smooth", "rough",
        "fast", "slow", "wet", "dry", "hard", "soft",
    ],
}


def avg_coverage(gs_basis, words):
    cs = []
    for w in words:
        v = get_deanisotropized(w)
        if v is None:
            continue
        cs.append(coverage_in(v, gs_basis))
    return np.mean(cs) if cs else None


def avg_coverage_random(random_bases, words):
    means = []
    for rb in random_bases:
        m = avg_coverage(rb, words)
        if m is not None:
            means.append(m)
    return (np.mean(means), np.std(means)) if means else (None, None)


# ============================================================================
# Main coverage table
# ============================================================================
print("\n" + "=" * 100)
print("COVERAGE COMPARISON — minimal bases vs Cog-13")
print("=" * 100)

bases_for_test = [
    ("VAL alone (1)", gs_VAL_only, None),
    ("UD alone (1)", gs_UD_only, None),
    ("VAL + UD (2)", gs_VAL_UD, None),
    ("VAL + UD + AR (3)", gs_VAL_UD_AR, None),
    ("Cog-13", gs_cog13, None),
]
random_bases_by_size = {1: random_1, 2: random_2, 3: random_3}

print(f"\n{'category':<26} " + " ".join(f"{n[:14]:>14}" for n, _, _ in bases_for_test) +
      f" {'Rand-1':>9} {'Rand-2':>9} {'Rand-3':>9}")
print("-" * 130)

all_results = {name: [] for name, _, _ in bases_for_test}
all_random = {1: [], 2: [], 3: []}

for cat, words in test_categories.items():
    row = f"  {cat:<24}"
    for name, basis, _ in bases_for_test:
        m = avg_coverage(basis, words)
        if m is not None:
            all_results[name].append(m * len([w for w in words if w in wv.key_to_index]))
            # accumulate mean weighted
            row += f"  {m * 100:>11.1f}%"
        else:
            row += f"  {'N/A':>13}"
    for k in [1, 2, 3]:
        rm, rs = avg_coverage_random(random_bases_by_size[k], words)
        if rm is not None:
            all_random[k].append((rm, rs))
            row += f"  {rm * 100:>6.1f}±{rs * 100:.1f}"
    print(row)


# Compute overall means
print("-" * 130)
overall_row = f"  {'OVERALL MEAN':<24}"
for name, basis, _ in bases_for_test:
    # recompute as straight mean
    cs_all = []
    for cat, words in test_categories.items():
        for w in words:
            v = get_deanisotropized(w)
            if v is None:
                continue
            cs_all.append(coverage_in(v, basis))
    overall_row += f"  {np.mean(cs_all) * 100:>11.2f}%"
for k in [1, 2, 3]:
    cs_random = []
    for rb in random_bases_by_size[k]:
        for cat, words in test_categories.items():
            for w in words:
                v = get_deanisotropized(w)
                if v is None:
                    continue
                cs_random.append(coverage_in(v, rb))
    overall_row += f"  {np.mean(cs_random) * 100:>7.1f}%"
print(overall_row)


# ============================================================================
# What does VALENCE+UD MISS that the other 11 axes capture?
# ============================================================================
print("\n" + "=" * 100)
print("Marginal coverage: each axis's content NOT in VALENCE+UD subspace")
print("=" * 100)

# For each of the other 11 cognitive axes, how much is in (VALENCE+UD) subspace?
print(f"\n{'axis':<10}  {'cos with VAL':>14}  {'cos with UD':>13}  "
      f"{'frac in (V+UD)':>16}  {'frac orthogonal':>16}")
print("-" * 78)

other_axes = [("C", C), ("W", W), ("ATT", ATT), ("INT", INT), ("R", R),
              ("D", D), ("IO", IO), ("DV", DV), ("MB", MB), ("EV", EV),
              ("ABS", ABS), ("REAL_IMAG", REAL_IMAG)]

for name, axis in other_axes:
    # Residual after projecting out gs_VAL_UD
    a = axis.copy()
    for u in gs_VAL_UD:
        a = a - (a @ u) * u
    in_subspace = 1 - (a @ a) / (axis @ axis)
    orthog = 1 - in_subspace
    print(f"  {name:<8}  {cos(axis, VALENCE):>+14.4f}  "
          f"{cos(axis, UD):>+13.4f}  "
          f"{in_subspace * 100:>14.1f}%  {orthog * 100:>14.1f}%")

print("\n(axes with high 'frac orthogonal' add content beyond VALENCE+UD;")
print(" axes with low 'frac orthogonal' are largely captured by VALENCE+UD already)")


np.savez("/Users/macn/Documents/embeddingexp/exp87_results.npz",
         VALENCE=VALENCE, UD=UD, AROUSAL=AROUSAL,
         cos_VAL_UD=cos(VALENCE, UD))
print("\nSaved exp87_results.npz")
