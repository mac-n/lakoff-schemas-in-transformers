"""
exp55: Lakoff schemas projected into 5D active-inference + multi-loss space.

The inversion experiment. Instead of "PCA on Lakoff axes → discover that PCs
look like AI primitives," we now go the other way: build a basis from
theory-led axes (active-inference primitives + Niamh's multi-loss-function
hypothesis), project Lakoff schemas onto it, see what they look like in
those coordinates.

The 5D AI-plus basis:
  - A: A_RUSSELL_DIAGONAL    — affect quality (pleasant-calm vs unpleasant-aroused)
  - C: C_REWARD_COMPOSITE    — integrated reward / expected value (winner of exp54
                               PC1-comparator at cos=+0.544 with PC1)
  - D: D_SURPRISAL           — compressibility / predictability
  - G: target_GOAL_DIRECTED  — policy precision (validated at cos=-0.30 with PC2)
  - R: target_EQ_RUN_resVA   — perceptual precision (V+A-stripped EQ-vs-RUN axis)

Niamh's multi-loss hypothesis: A, C, D are three orthogonal candidate loss
functions the brain tracks (affect-regulation, reward-maximization, prediction-
error-minimization / compression). The empirical inter-target orthogonality
from exp54 (cos(A,C)=+0.008, cos(A,D)=+0.023, cos(C,D)=+0.183) supports the
multi-loss reading.

The inversion test:
  For each Lakoff schema, compute coordinates (a, c, d, g, r) by projecting
  onto the 5D basis. Then compute residual norm — how much of each schema's
  axis lives OUTSIDE the AI-plus subspace.

Predictions:
  - Tier-1 cluster schemas (UD, FB, LD, EXIST, SUC, LOSS, COH, BAL, PATH)
    should project substantially onto the basis. Each should have a
    distinctive signature.
  - Tier-2 independent schemas (IO_CLEAN, FORCE, DIFF) should project
    weakly onto the basis. Large residuals → they're not in AI-plus space.

Stretch goal: project concept words (HOPE, FREEDOM, BETRAYAL, GROWTH,
PSYCHOSIS, REGULATION, AGENCY, FLOW, DEPRESSION) onto AI-plus space
to see what their coordinates look like.
"""
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
    TARGET_SALIENCE_PAIRS,  # = A: Russell diagonal
    TARGET_EQUILIBRIUM_RUNAWAY_PAIRS,
)
from exp53_residual_and_goal_directed import (
    TARGET_GOAL_DIRECTED_PAIRS,
)
from exp54_pc1_comparator import (
    TARGET_VALUE_PURE_PAIRS,
    TARGET_REWARD_COMPOSITE_PAIRS,
    TARGET_SURPRISAL_PAIRS,
)


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


def build_axis(pairs, label=""):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    if not offs:
        raise ValueError(f"No pairs survived OOV for {label}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


def project_out(v, u):
    u_unit = u / np.linalg.norm(u)
    return v - (v @ u_unit) * u_unit


# ============================================================
# Build the 5 AI-plus basis axes
# ============================================================

print("\nBuilding 5D AI-plus basis...")
A = build_axis(TARGET_SALIENCE_PAIRS,            "A_affect")
C = build_axis(TARGET_REWARD_COMPOSITE_PAIRS,    "C_reward")
D = build_axis(TARGET_SURPRISAL_PAIRS,           "D_compress")
G = build_axis(TARGET_GOAL_DIRECTED_PAIRS,       "G_policy_prec")
EQ_raw = build_axis(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, "EQ_RUN_raw")

# Build V and A_AROUSAL for residualizing EQ_RUN against (V+A)
VALENCE = build_axis(VALENCE_PAIRS, "VALENCE")
AROUSAL = build_axis(AROUSAL_PAIRS, "AROUSAL")

EQ_resV  = project_out(EQ_raw, VALENCE)
EQ_resVA = project_out(EQ_resV, AROUSAL)
R = EQ_resVA / np.linalg.norm(EQ_resVA)

basis = {
    "A_affect":      A,
    "C_reward":      C,
    "D_compress":    D,
    "G_policy_prec": G,
    "R_percept_prec": R,
}

# ============================================================
# Check inter-basis orthogonality
# ============================================================
print("\n" + "="*72)
print("Inter-basis orthogonality (5x5)")
print("="*72)
print(f"\n{'':<16}" + "".join(f"{n:<16}" for n in basis.keys()))
B_names = list(basis.keys())
for n1 in B_names:
    row = f"{n1:<16}"
    for n2 in B_names:
        if n1 == n2:
            row += f"{'  1.000':<16}"
        else:
            c = float(basis[n1] @ basis[n2])
            row += f"{c:>+8.3f}        "
    print(row)


# ============================================================
# Build Lakoff schema axes
# ============================================================
print("\n\nBuilding Lakoff schema axes...")
schemas = {
    "UD":        build_axis(UP_DOWN_MML, "UD"),
    "FB":        build_axis(FORWARD_BACK_MML, "FB"),
    "LD":        build_axis(LIGHT_DARK_MML, "LD"),
    "IO_CLEAN":  build_axis(IN_OUT_MML_CLEAN, "IO_CLEAN"),
    "PATH":      build_axis(PATH_MOTION_MML, "PATH"),
    "EXIST":     build_axis(EXISTENCE_MML, "EXIST"),
    "FORCE":     build_axis(FORCE_MML, "FORCE"),
    "BAL":       build_axis(BALANCE_MML, "BAL"),
    "DIFF":      build_axis(DIFFICULTY_BURDEN_MML, "DIFF"),
    "COH":       build_axis(COHERENCE_PAIRS, "COH"),
    "SUC":       build_axis(SUCCESS_FAILURE_PAIRS, "SUC"),
    "LOSS":      build_axis(LOSS_PAIRS, "LOSS"),
    "VALENCE":   VALENCE,
    "AROUSAL":   AROUSAL,
}


# ============================================================
# Project schemas onto basis
# ============================================================

def project_onto_basis(v, basis_axes):
    """
    Project vector v onto the (non-orthogonal) basis using each basis axis
    individually (each coordinate = cos(v, basis_axis)). Then compute the
    residual: subtract the component along each basis axis sequentially
    (which approximates Gram-Schmidt for nearly-orthogonal bases).

    Note: since basis is approximately orthogonal, the individual cosines
    give the dominant decomposition. We also report Gram-Schmidt-corrected
    residual norm for precision.
    """
    coords = {}
    for name, ax in basis_axes.items():
        coords[name] = float(v @ ax)  # cosine (since v and ax are unit-norm)

    # Compute Gram-Schmidt-orthogonalized residual
    v_residual = v.copy()
    # Orthogonalize basis first (Gram-Schmidt order: A, C, D, G, R)
    ordered_axes = [basis_axes[n] for n in basis_axes.keys()]
    gs_basis = []
    for u in ordered_axes:
        u_gs = u.copy()
        for prev in gs_basis:
            u_gs = u_gs - (u_gs @ prev) * prev
        norm = np.linalg.norm(u_gs)
        if norm > 1e-8:
            u_gs = u_gs / norm
        gs_basis.append(u_gs)
    # Project v onto orthogonalized basis
    for u_gs in gs_basis:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    residual_norm = float(np.linalg.norm(v_residual))
    explained = float(np.sqrt(max(0, 1 - residual_norm**2)))

    return coords, residual_norm, explained


print("\n" + "="*72)
print("LAKOFF SCHEMAS IN AI-PLUS SPACE")
print("="*72)
print()
print(f"{'schema':<10} {'A_aff':>8} {'C_rew':>8} {'D_cmp':>8} "
      f"{'G_pol':>8} {'R_per':>8} | {'residual':>9} {'explained':>10}")
print("-"*82)

projections = {}
for sname, svec in schemas.items():
    coords, resnorm, expl = project_onto_basis(svec, basis)
    projections[sname] = {"coords": coords, "residual": resnorm, "explained": expl}
    row = f"{sname:<10}"
    for bname in basis.keys():
        row += f" {coords[bname]:>+7.3f} "
    row += f" |  {resnorm:>+.3f}    {expl:>+.3f}"
    print(row)


# ============================================================
# Interpretation summary
# ============================================================
print("\n" + "="*72)
print("INTERPRETATION SUMMARY")
print("="*72)
print()
print("For each schema, the dominant coordinate (largest |cos|) and verdict:")
print()
for sname, sdata in projections.items():
    cmax = max(sdata["coords"].items(), key=lambda kv: abs(kv[1]))
    expl_pct = sdata["explained"] * 100
    tier = ("cluster" if sname in {"UD","FB","LD","EXIST","SUC","LOSS","COH","BAL","PATH",
                                    "VALENCE","AROUSAL"}
            else "independent (Tier-2)")
    print(f"  {sname:<10} [{tier:<22}]  dominant: {cmax[0]:<14} ({cmax[1]:+.3f}), "
          f"basis-explained: {expl_pct:>5.1f}%")


# ============================================================
# Concept-word stretch goal
# ============================================================
print("\n" + "="*72)
print("CONCEPT-WORD STRETCH GOAL")
print("="*72)
print()
print("Project specific concept words (not anchor-built axes) onto AI-plus space.")
print("Each word vector is unit-normalized first.")
print()

# Concept words: project word vectors directly (no anchor pairs)
concept_words = [
    "hope", "freedom", "betrayal", "growth", "psychosis", "regulation",
    "agency", "flow", "depression", "trauma", "creativity", "play",
    "ritual", "meditation", "grief", "love",
]

print(f"{'word':<14} {'A_aff':>8} {'C_rew':>8} {'D_cmp':>8} "
      f"{'G_pol':>8} {'R_per':>8} | {'explained':>10}")
print("-"*78)

for w in concept_words:
    if w not in wv.key_to_index:
        print(f"  {w:<14} not in GloVe")
        continue
    v = wv[w]
    v_unit = v / np.linalg.norm(v)
    coords, resnorm, expl = project_onto_basis(v_unit, basis)
    row = f"  {w:<14}"
    for bname in basis.keys():
        row += f" {coords[bname]:>+7.3f} "
    row += f" |   {expl*100:>5.1f}%"
    print(row)


# ============================================================
# Save
# ============================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp55_results.npz",
    basis_axes={k: v for k, v in basis.items()},
    schema_projections={sname: sdata for sname, sdata in projections.items()},
)
print("\nSaved: exp55_results.npz")
