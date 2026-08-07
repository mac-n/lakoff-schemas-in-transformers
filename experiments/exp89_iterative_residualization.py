"""
exp89 — Iterative greedy residualization.

Niamh's idea: UD is more explanatory than VALENCE (18.3% vs 11.1%). What
falls out if we subtract VALENCE from UD — i.e., what's UD's content that
isn't captured by VALENCE? Then iterate: at each step, find the most-
explanatory single axis of what's left.

This is essentially interpretable-greedy-PCA: instead of finding variance-
maximizing principal components, find the most-explanatory NAMED axis from
a pool of cognitively-meaningful candidates, residualize against it,
repeat.

Two-part experiment:

PART A — direct test of UD−VAL:
  Build UD_resid_VAL = residualize(UD, VALENCE). Check coverage, pole vocab,
  cos with cognitive axes.

PART B — iterative greedy basis-building:
  Start with empty basis. At each step:
    1. Project every candidate axis onto the orthogonal complement of current basis
    2. Compute residual-coverage of each
    3. Pick the highest, add to basis, residualize remaining candidates
  Repeat ~13 times. See which axes get selected and in what order.
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
from lakoff_canonical_vocabulary import (
    IN_OUT_MML_CLEAN, UP_DOWN_MML, FORWARD_BACK_MML, LIGHT_DARK_MML,
    PATH_MOTION_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML,
    DIFFICULTY_BURDEN_MML,
)
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
)


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


def residualize(axis, against):
    return unit(axis - (axis @ against) * against)


def coverage_in(v, gs_basis):
    v_residual = v.copy()
    for u_gs in gs_basis:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)


def get_deanisotropized(word):
    if word not in wv.key_to_index:
        return None
    v = wv[word] - mu
    n = np.linalg.norm(v)
    if n < 1e-10:
        return None
    return v / n


print("Building all candidate axes...")
# Cognitive (our basis)
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

# Lakoff (some not in current basis)
UD = build_axis(wv, UP_DOWN_MML)
FB = build_axis(wv, FORWARD_BACK_MML)
LD = build_axis(wv, LIGHT_DARK_MML)
PATH = build_axis(wv, PATH_MOTION_MML)
EXIST = build_axis(wv, EXISTENCE_MML)
FORCE = build_axis(wv, FORCE_MML)
BAL = build_axis(wv, BALANCE_MML)
DIFF = build_axis(wv, DIFFICULTY_BURDEN_MML)

# Cluster
VALENCE = build_axis(wv, VALENCE_PAIRS)
AROUSAL = build_axis(wv, AROUSAL_PAIRS)
COH = build_axis(wv, COHERENCE_PAIRS)
SUC = build_axis(wv, SUCCESS_FAILURE_PAIRS)
LOSS = build_axis(wv, LOSS_PAIRS)


# ============================================================================
# PART A — UD−VAL test
# ============================================================================
print("\n" + "=" * 78)
print("PART A — UD residualized against VALENCE")
print("=" * 78)

UD_minus_VAL = residualize(UD, VALENCE)
print(f"\ncos(UD_resid_VAL, VALENCE) = {cos(UD_minus_VAL, VALENCE):+.4f}  (should be ~0)")
print(f"cos(UD_resid_VAL, UD)      = {cos(UD_minus_VAL, UD):+.4f}  (how much of UD remains)")

print("\nUD_resid_VAL positive pole (top 15):")
for w, s in wv.similar_by_vector(UD_minus_VAL.astype(np.float32), topn=15):
    print(f"  {w:25s}  {s:+.4f}")
print("\nUD_resid_VAL negative pole (top 15):")
for w, s in wv.similar_by_vector((-UD_minus_VAL).astype(np.float32), topn=15):
    print(f"  {w:25s}  {s:+.4f}")

print("\ncos(UD_resid_VAL, all cognitive axes):")
for name, axis in [("C", C), ("W", W), ("ATT", ATT), ("INT", INT), ("R", R),
                    ("D", D), ("IO", IO), ("DV", DV), ("MB", MB), ("EV", EV),
                    ("ABS", ABS), ("REAL_IMAG", REAL_IMAG), ("UD", UD),
                    ("AROUSAL", AROUSAL)]:
    print(f"  {name:<10}  {cos(UD_minus_VAL, axis):+.4f}")


# ============================================================================
# PART B — Iterative greedy residualization
# ============================================================================
print("\n" + "=" * 78)
print("PART B — Iterative greedy basis-building")
print("=" * 78)
print("""
At each step:
  1. For every candidate axis, residualize against current basis
  2. Compute coverage of each residualized candidate on a test corpus
  3. Pick the highest, add to basis
  4. Repeat
""")

# Candidate pool (all axes, named)
candidates = {
    "C": C, "W": W, "ATT": ATT, "INT": INT, "R": R, "D": D, "IO": IO,
    "DV": DV, "MB": MB, "EV": EV, "ABS": ABS, "REAL_IMAG": REAL_IMAG,
    "UD": UD, "FB": FB, "LD": LD, "PATH": PATH, "EXIST": EXIST,
    "FORCE": FORCE, "BAL": BAL, "DIFF": DIFF,
    "VALENCE": VALENCE, "AROUSAL": AROUSAL, "COH": COH, "SUC": SUC, "LOSS": LOSS,
}

# Test corpus: deanisotropized vocab sample
np.random.seed(42)
sample_idx = np.random.choice(len(wv.vectors), 30000, replace=False)
sample = wv.vectors[sample_idx] - mu
norms = np.linalg.norm(sample, axis=1, keepdims=True)
norms[norms < 1e-10] = 1
sample = sample / norms  # (30000, 300) deanisotropized unit-normed


def basis_coverage_on_sample(gs_basis):
    """Mean coverage of basis on the 30K sample."""
    if not gs_basis:
        return 0.0
    B = np.stack(gs_basis)  # (k, 300)
    projections = sample @ B.T  # (N, k)
    captured = (projections ** 2).sum(axis=1)  # (N,) — variance captured per word
    explained = np.sqrt(np.maximum(0, captured))  # since vecs are unit-normed
    return float(explained.mean())


# Greedy selection
selected_names = []
selected_axes = []
selected_orthonormal = []  # GS-orthonormalized
remaining = dict(candidates)

print(f"\n{'step':<5} {'selected':<14} {'cov added':>10} {'cum cov':>9}  next-3-candidates")
print("-" * 75)

prev_coverage = 0.0
max_steps = 15

for step in range(1, max_steps + 1):
    if not remaining:
        break

    # For each remaining candidate, compute coverage if it were added
    candidate_results = []
    for name, axis in remaining.items():
        # Residualize against current orthonormal basis
        res = axis.copy()
        for u in selected_orthonormal:
            res = res - (res @ u) * u
        n = np.linalg.norm(res)
        if n < 1e-6:
            continue  # axis already in basis subspace
        res = res / n
        # Coverage of trial-basis (current + this candidate)
        trial_basis = selected_orthonormal + [res]
        cov = basis_coverage_on_sample(trial_basis)
        candidate_results.append((name, axis, res, cov))

    # Sort by coverage
    candidate_results.sort(key=lambda x: -x[3])

    if not candidate_results:
        break

    best_name, best_axis, best_res, best_cov = candidate_results[0]
    added = best_cov - prev_coverage

    # Show top 3 candidates
    top3 = ", ".join(f"{n} ({c * 100:.1f}%)"
                     for n, _, _, c in candidate_results[1:4])

    print(f"  {step:<3}  {best_name:<14}  +{added * 100:>6.2f}pp "
          f"{best_cov * 100:>7.2f}%  ← " + top3)

    selected_names.append(best_name)
    selected_axes.append(best_axis)
    selected_orthonormal.append(best_res)
    del remaining[best_name]
    prev_coverage = best_cov

    # Stopping criterion: added coverage < 1pp
    if added < 0.01 and step >= 5:
        print(f"  (added coverage dropped below 1pp at step {step}; stopping demo)")
        break


print(f"\nFinal greedy basis ({len(selected_names)} axes): {selected_names}")
print(f"Final coverage: {prev_coverage * 100:.2f}%")

# Comparison to our hand-curated 13-axis basis
basis_13b_axes = [
    UD,
    residualize(C, UD),
    W, ATT,
    residualize(INT, UD),
    R, D, IO,
    residualize(DV, UD),
    MB, EV, ABS,
    residualize(REAL_IMAG, UD),
]
gs_13b = gs_orthogonalize(basis_13b_axes)
print(f"\nReference: Cog-13b coverage on same sample = "
      f"{basis_coverage_on_sample(gs_13b) * 100:.2f}%")


np.savez("/Users/macn/Documents/embeddingexp/exp89_results.npz",
         UD_minus_VAL=UD_minus_VAL,
         selected_names=np.array(selected_names))
print("\nSaved exp89_results.npz")
