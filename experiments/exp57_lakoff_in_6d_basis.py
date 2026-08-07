"""
exp57: Lakoff schemas + concept words projected onto a 6D (non-orthogonal)
cognitive-primitive basis.

The six axes (from exp52-56):
  A  = A_RUSSELL_DIAGONAL    — affect quality (Russell V x A diagonal)
  C  = C_REWARD_COMPOSITE    — integrated wellbeing / expected value of state
  D  = D_SURPRISAL           — compressibility / predictability
  G  = G_GOAL_DIRECTED       — policy precision (commitment-to-policy)
  R  = R_EQ_RUN_residual_VA  — perceptual precision (V+A-stripped regulability)
  EE = EE_EXPLOIT_EXPLORE    — exploit-vs-explore decision

Caveat: the basis is non-Euclidean. G is correlated with A at +0.56;
all other pairwise |cos| < 0.2. We report both raw cosines (interpretable
per-axis) and Gram-Schmidt-orthogonalized coordinates (interpretable as
a clean projection-decomposition).

For each Lakoff schema and each concept word, report:
  - Raw cos with each of 6 basis axes
  - Gram-Schmidt coordinates (orthogonalized in order: C, A, D, G, R, EE)
  - Basis-explained fraction (how much of the vector lives in the 6D subspace)

Question: do specific Lakoff schemas have distinctive signatures across all
six primitives? Do specific concept words reveal active-inference predictions
(psychosis low everywhere, ritual on D, hope on G+, etc.)?
"""
import numpy as np
import gensim.downloader as api
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
    TARGET_SALIENCE_PAIRS, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS,
)
from exp53_residual_and_goal_directed import TARGET_GOAL_DIRECTED_PAIRS
from exp54_pc1_comparator import TARGET_REWARD_COMPOSITE_PAIRS, TARGET_SURPRISAL_PAIRS
from exp56_explore_exploit import TARGET_EXPLOIT_EXPLORE_PAIRS


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


def build_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


def project_out(v, u):
    u_unit = u / np.linalg.norm(u)
    return v - (v @ u_unit) * u_unit


# ============================================================
# Build 6D basis
# ============================================================
A_axis = build_axis(TARGET_SALIENCE_PAIRS)
C_axis = build_axis(TARGET_REWARD_COMPOSITE_PAIRS)
D_axis = build_axis(TARGET_SURPRISAL_PAIRS)
G_axis = build_axis(TARGET_GOAL_DIRECTED_PAIRS)
EE_axis = build_axis(TARGET_EXPLOIT_EXPLORE_PAIRS)

# R = perceptual precision = EQ_RUN with V+A removed
VALENCE = build_axis(VALENCE_PAIRS)
AROUSAL = build_axis(AROUSAL_PAIRS)
EQ_raw = build_axis(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
R_axis_pre = project_out(EQ_raw, VALENCE)
R_axis = project_out(R_axis_pre, AROUSAL)
R_axis = R_axis / np.linalg.norm(R_axis)

basis_raw = {
    "C_rew":   C_axis,
    "A_aff":   A_axis,
    "D_cmp":   D_axis,
    "G_pol":   G_axis,
    "R_per":   R_axis,
    "EE_x":    EE_axis,
}

# Gram-Schmidt order: most independent first (C, then A, D, R, EE), G last
# because it's the most entangled
gs_order = ["C_rew", "A_aff", "D_cmp", "R_per", "EE_x", "G_pol"]
gs_basis = []
for name in gs_order:
    u = basis_raw[name].copy()
    for prev in gs_basis:
        u = u - (u @ prev) * prev
    norm = np.linalg.norm(u)
    if norm > 1e-8:
        u = u / norm
    gs_basis.append(u)


# ============================================================
# Show inter-basis cosines (raw + post-GS)
# ============================================================
print("\n" + "="*72)
print("6D basis inter-axis cosines (RAW)")
print("="*72)
print()
names = list(basis_raw.keys())
print(f"{'':<10}" + "".join(f"{n:<10}" for n in names))
for n1 in names:
    row = f"{n1:<10}"
    for n2 in names:
        if n1 == n2:
            row += f"{'  1.00':<10}"
        else:
            c = float(basis_raw[n1] @ basis_raw[n2])
            row += f"{c:>+6.3f}    "
    print(row)


# ============================================================
# Project Lakoff schemas
# ============================================================
schemas = {
    "UD":       build_axis(UP_DOWN_MML),
    "FB":       build_axis(FORWARD_BACK_MML),
    "LD":       build_axis(LIGHT_DARK_MML),
    "IO_CLEAN": build_axis(IN_OUT_MML_CLEAN),
    "PATH":     build_axis(PATH_MOTION_MML),
    "EXIST":    build_axis(EXISTENCE_MML),
    "FORCE":    build_axis(FORCE_MML),
    "BAL":      build_axis(BALANCE_MML),
    "DIFF":     build_axis(DIFFICULTY_BURDEN_MML),
    "COH":      build_axis(COHERENCE_PAIRS),
    "SUC":      build_axis(SUCCESS_FAILURE_PAIRS),
    "LOSS":     build_axis(LOSS_PAIRS),
    "VALENCE":  VALENCE,
    "AROUSAL":  AROUSAL,
}


def raw_cosines(v):
    return {n: float(v @ basis_raw[n]) for n in names}


def gs_projection(v):
    """Returns dict of GS coordinates (named) + residual norm + explained fraction."""
    coords = {}
    v_residual = v.copy()
    for name, u_gs in zip(gs_order, gs_basis):
        c = float(v_residual @ u_gs)
        coords[name] = c
        v_residual = v_residual - c * u_gs
    residual_norm = float(np.linalg.norm(v_residual))
    explained = float(np.sqrt(max(0, 1 - residual_norm**2)))
    return coords, residual_norm, explained


# ============================================================
# Lakoff schemas: raw cosines + GS coordinates
# ============================================================
print("\n" + "="*72)
print("LAKOFF SCHEMAS — RAW cosines with each basis axis")
print("="*72)
print()
print(f"{'schema':<10} " + " ".join(f"{n:>7}" for n in names))
print("-"*72)
for sname, svec in schemas.items():
    raw = raw_cosines(svec)
    row = f"{sname:<10}"
    for n in names:
        row += f" {raw[n]:>+7.3f}"
    print(row)

print("\n" + "="*72)
print("LAKOFF SCHEMAS — GRAM-SCHMIDT coordinates (C, A, D, R, EE, G order)")
print("="*72)
print()
print(f"{'schema':<10} " + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
print("-"*72)
for sname, svec in schemas.items():
    coords, _, expl = gs_projection(svec)
    row = f"{sname:<10}"
    for n in gs_order:
        row += f" {coords[n]:>+7.3f}"
    row += f" | {expl*100:>4.1f}%"
    print(row)


# ============================================================
# Concept words
# ============================================================
print("\n" + "="*72)
print("CONCEPT WORDS — RAW cosines")
print("="*72)
print()

concept_words = [
    "hope", "freedom", "betrayal", "growth", "psychosis", "regulation",
    "agency", "flow", "depression", "trauma", "creativity", "play",
    "ritual", "meditation", "grief", "love",
    # additional concepts probing the basis
    "boredom", "curiosity", "obsession", "addiction", "wisdom",
    "rage", "compassion", "fear", "joy",
]

print(f"{'word':<14} " + " ".join(f"{n:>7}" for n in names))
print("-"*78)
for w in concept_words:
    if w not in wv.key_to_index:
        print(f"  {w:<14} not in GloVe")
        continue
    v = wv[w] / np.linalg.norm(wv[w])
    raw = raw_cosines(v)
    row = f"  {w:<14}"
    for n in names:
        row += f" {raw[n]:>+7.3f}"
    print(row)


print("\n" + "="*72)
print("CONCEPT WORDS — GRAM-SCHMIDT coordinates")
print("="*72)
print()
print(f"{'word':<14} " + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
print("-"*82)
for w in concept_words:
    if w not in wv.key_to_index:
        continue
    v = wv[w] / np.linalg.norm(wv[w])
    coords, _, expl = gs_projection(v)
    row = f"  {w:<14}"
    for n in gs_order:
        row += f" {coords[n]:>+7.3f}"
    row += f" | {expl*100:>4.1f}%"
    print(row)


# ============================================================
# Save
# ============================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp57_results.npz",
    basis_raw={n: v for n, v in basis_raw.items()},
    gs_basis=np.stack(gs_basis),
    gs_order=np.array(gs_order),
)
print("\nSaved: exp57_results.npz")
