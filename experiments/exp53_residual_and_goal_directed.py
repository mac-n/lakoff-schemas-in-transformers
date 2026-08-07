"""
exp53: Two follow-up tests after exp52.

(1) Residualize target_EQ_RUN against PC1 (the V x A diagonal) and re-check
    PC3 alignment. The exp52 result showed cos(target_EQ_RUN, PC1) = +0.29
    and cos(target_EQ_RUN, PC3) = +0.21. Need to know: after stripping out
    the V x A diagonal contribution (regulation = good = calm, runaway =
    bad = aroused), does PC3 alignment SURVIVE or COLLAPSE?

    If the PC3 cosine of the residualized target is still moderate (>0.3),
    the equilibrium-vs-runaway / regulability axis is a real primitive
    distinct from affect.

    If it collapses to noise, "regulability" was just valence-with-arousal
    in different clothing.

(2) Test Niamh's reframe of PC2 from MOTION to GOAL-DIRECTEDNESS.

    exp52 found target_MOTION (locomotion vs body-position) does NOT
    recover PC2 (max cos +0.05). But Niamh noticed that PC2's negative
    pole nearest neighbors in exp40 included *must, ensure, unable, failure*
    — modal/constraint/policy vocabulary, NOT stasis vocabulary.

    Goal-directedness — purposeful action toward an aim vs maintenance /
    constraint / inability — fits PC2's actual semantic shape better than
    literal motion does. Active-inference reframe: this is the policy axis.

    target_GOAL_DIRECTED anchors are valence-balanced (purposeful action
    can be good or sinister; passivity can be relief or failure).
"""
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)

# Reuse the anchor sets from exp52 (reimport rather than re-state for brevity)
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
    TARGET_SALIENCE_PAIRS, TARGET_MOTION_PAIRS,
    TARGET_EQUILIBRIUM_RUNAWAY_PAIRS,
)


# ============================================================
# NEW: target_GOAL_DIRECTED anchors
# Valence-balanced as far as possible. Purposeful action can be good or
# sinister; passivity can be peaceful relief or thwarted failure.
# Avoids all PATH / spatial-direction vocabulary, all V/A/SUC/LOSS/COH/EXIST
# anchors. Tests whether PC2 is operationalizing goal-pursuit-vs-maintenance
# rather than literal motion-vs-stasis.
# ============================================================
TARGET_GOAL_DIRECTED_PAIRS = [
    ("pursuing",       "idling"),
    ("aiming",         "wandering"),
    ("purposeful",     "aimless"),
    ("deliberate",     "accidental"),
    ("motivated",      "unmotivated"),
    ("intentional",    "unintentional"),
    ("resolute",       "hesitant"),
    ("committed",      "uncommitted"),
    ("driven",         "becalmed"),
    ("oriented",       "disoriented"),
    ("targeted",       "untargeted"),
    ("decided",        "undecided"),
    ("chasing",        "dawdling"),
    ("ambitious",      "complacent"),
]


# ============================================================
# Load embeddings + build axes
# ============================================================

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


def build_axis(pairs, label=""):
    offs = []
    missing = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
        else:
            for w in (a, c):
                if w not in wv.key_to_index:
                    missing.append(w)
    if missing:
        print(f"  [{label}] missing words ({len(missing)}): {sorted(set(missing))}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw), len(offs)


# Build the 11 cluster axes (PCA input set, identical to exp40 / exp52)
print("\nBuilding PCA input axes (cluster set)...")
axes = {}
axes["VALENCE"],   _ = build_axis(VALENCE_PAIRS, "VALENCE")
axes["AROUSAL"],   _ = build_axis(AROUSAL_PAIRS, "AROUSAL")
axes["UD"],        _ = build_axis(UP_DOWN_MML, "UD")
axes["FB"],        _ = build_axis(FORWARD_BACK_MML, "FB")
axes["LD"],        _ = build_axis(LIGHT_DARK_MML, "LD")
axes["PATH"],      _ = build_axis(PATH_MOTION_MML, "PATH")
axes["EXIST"],     _ = build_axis(EXISTENCE_MML, "EXIST")
axes["BAL"],       _ = build_axis(BALANCE_MML, "BAL")
axes["COHERENCE"], _ = build_axis(COHERENCE_PAIRS, "COHERENCE")
axes["SUCCESS"],   _ = build_axis(SUCCESS_FAILURE_PAIRS, "SUCCESS")
axes["LOSS"],      _ = build_axis(LOSS_PAIRS, "LOSS")
axis_names = list(axes.keys())
M = np.stack([axes[n] for n in axis_names])


# Build target axes
print("\nBuilding target axes...")
target_sal, _ = build_axis(TARGET_SALIENCE_PAIRS, "target_SALIENCE")
target_mot, _ = build_axis(TARGET_MOTION_PAIRS, "target_MOTION")
target_eqr, _ = build_axis(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, "target_EQ_RUN")
target_gld, n_gld = build_axis(TARGET_GOAL_DIRECTED_PAIRS, "target_GOAL_DIRECTED")
print(f"  target_GOAL_DIRECTED built from {n_gld}/{len(TARGET_GOAL_DIRECTED_PAIRS)} pairs")


# ============================================================
# Run PCA (same as exp40 / exp52)
# ============================================================
pca = PCA(n_components=min(len(axis_names), 10))
pca.fit(M)


def cos(a, b):
    """cosine between two vectors (no need to renormalize if already unit-norm)"""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def project_out(v, u):
    """remove the u-component from v. assumes u is unit-norm."""
    u_unit = u / np.linalg.norm(u)
    return v - (v @ u_unit) * u_unit


# ============================================================
# TEST 1: residualize target_EQ_RUN against PC1, re-test PC3
# ============================================================
print("\n" + "="*72)
print("TEST 1: target_EQ_RUN residualized against PC1")
print("="*72)
print()
print("Question: after stripping out the V x A diagonal contribution, does")
print("the EQUILIBRIUM-vs-RUNAWAY signal survive on PC3?")
print()

pc1 = pca.components_[0]
pc2 = pca.components_[1]
pc3 = pca.components_[2]

# Before residualization (replicating exp52)
print("Before residualization (exp52 baseline):")
print(f"  cos(target_EQ_RUN, PC1) = {cos(target_eqr, pc1):>+.4f}")
print(f"  cos(target_EQ_RUN, PC2) = {cos(target_eqr, pc2):>+.4f}")
print(f"  cos(target_EQ_RUN, PC3) = {cos(target_eqr, pc3):>+.4f}")

# Residualize: project out PC1
target_eqr_res = project_out(target_eqr, pc1)
norm_before = np.linalg.norm(target_eqr)
norm_after = np.linalg.norm(target_eqr_res)
target_eqr_res_unit = target_eqr_res / norm_after

print(f"\nAfter projecting out PC1:")
print(f"  Magnitude retained: {norm_after / norm_before * 100:.1f}% "
      f"(stripped {(1 - norm_after/norm_before) * 100:.1f}%)")
print(f"  cos(target_EQ_RUN_res_unit, PC1) = {cos(target_eqr_res_unit, pc1):>+.4f}   (should be ~0)")
print(f"  cos(target_EQ_RUN_res_unit, PC2) = {cos(target_eqr_res_unit, pc2):>+.4f}")
print(f"  cos(target_EQ_RUN_res_unit, PC3) = {cos(target_eqr_res_unit, pc3):>+.4f}")

# Additionally residualize against VALENCE specifically (input axis), not just PC1
target_eqr_resV = project_out(target_eqr, axes["VALENCE"])
target_eqr_resVA = project_out(target_eqr_resV, axes["AROUSAL"])
target_eqr_resVA_unit = target_eqr_resVA / np.linalg.norm(target_eqr_resVA)
print(f"\nAlternative residualization: project out VALENCE then AROUSAL (input axes):")
print(f"  Magnitude retained: {np.linalg.norm(target_eqr_resVA) / norm_before * 100:.1f}%")
print(f"  cos(EQ_RUN_resVA, PC1) = {cos(target_eqr_resVA_unit, pc1):>+.4f}")
print(f"  cos(EQ_RUN_resVA, PC2) = {cos(target_eqr_resVA_unit, pc2):>+.4f}")
print(f"  cos(EQ_RUN_resVA, PC3) = {cos(target_eqr_resVA_unit, pc3):>+.4f}")

# Verdict
pc3_residual = cos(target_eqr_res_unit, pc3)
pc3_resVA = cos(target_eqr_resVA_unit, pc3)
print(f"\nVerdict:")
print(f"  PC3 alignment after PC1-residualization: {pc3_residual:+.3f}")
print(f"  PC3 alignment after V+A-residualization: {pc3_resVA:+.3f}")
if abs(pc3_residual) > 0.30:
    print(f"  → Regulability hypothesis SURVIVES: PC3 alignment moderate-to-strong after"
          f" stripping affect-diagonal load")
elif abs(pc3_residual) > 0.15:
    print(f"  → Regulability hypothesis PARTIAL: PC3 alignment present but not strong"
          f" after residualization")
else:
    print(f"  → Regulability hypothesis DOES NOT survive: PC3 alignment collapses"
          f" after stripping affect-diagonal load — EQ-vs-RUN was mostly V x A")


# ============================================================
# TEST 2: target_GOAL_DIRECTED on PC2 (and other PCs)
# ============================================================
print("\n" + "="*72)
print("TEST 2: target_GOAL_DIRECTED against PCs")
print("="*72)
print()
print("Question: does PC2 operationalize goal-pursuit-vs-maintenance / policy")
print("(Niamh's reframe), rather than literal locomotion-vs-stasis?")
print()

print(f"Cosines:")
for i in range(6):
    pc_i = pca.components_[i]
    c = cos(target_gld, pc_i)
    print(f"  cos(target_GOAL_DIRECTED, PC{i+1} ({pca.explained_variance_ratio_[i]*100:>4.1f}% var)) = {c:>+.4f}")

# Compare to target_MOTION on the same PCs
print(f"\nReminder (target_MOTION from exp52):")
for i in range(6):
    pc_i = pca.components_[i]
    c = cos(target_mot, pc_i)
    print(f"  cos(target_MOTION, PC{i+1} ({pca.explained_variance_ratio_[i]*100:>4.1f}% var)) = {c:>+.4f}")

# Sanity: cosines with input axes
print(f"\ntarget_GOAL_DIRECTED cosines with INPUT axes:")
input_cos = [(n, cos(target_gld, axes[n])) for n in axis_names]
input_cos.sort(key=lambda x: abs(x[1]), reverse=True)
for n, c in input_cos[:6]:
    print(f"  {n:>10}: {c:>+.4f}")

# Nearest-neighbor words
print(f"\ntarget_GOAL_DIRECTED positive pole words:")
for w, s in wv.similar_by_vector(target_gld, topn=12):
    print(f"  {s:>+.4f}  {w}")
print(f"\ntarget_GOAL_DIRECTED negative pole words:")
for w, s in wv.similar_by_vector(-target_gld, topn=12):
    print(f"  {s:>+.4f}  {w}")


# ============================================================
# TEST 3: target_GOAL_DIRECTED residualized against PC1 (parallel to test 1)
# ============================================================
print("\n" + "="*72)
print("TEST 3: target_GOAL_DIRECTED residualized against PC1")
print("="*72)
print()
print("Sanity check: goal-directedness might also load on PC1 because")
print("'driven, committed, motivated, deliberate' lean positive-valence")
print("while 'idling, dawdling, hesitant' lean negative-valence.")

target_gld_res = project_out(target_gld, pc1)
target_gld_res_unit = target_gld_res / np.linalg.norm(target_gld_res)
gld_norm_before = np.linalg.norm(target_gld)
gld_norm_after = np.linalg.norm(target_gld_res)
print(f"\nMagnitude retained after PC1-residualization: "
      f"{gld_norm_after / gld_norm_before * 100:.1f}%")
print(f"  cos(GOAL_DIRECTED_res, PC1) = {cos(target_gld_res_unit, pc1):>+.4f}   (~0)")
print(f"  cos(GOAL_DIRECTED_res, PC2) = {cos(target_gld_res_unit, pc2):>+.4f}")
print(f"  cos(GOAL_DIRECTED_res, PC3) = {cos(target_gld_res_unit, pc3):>+.4f}")


# ============================================================
# Synthesis table — all target axes vs all PCs, raw + PC1-residualized
# ============================================================
print("\n" + "="*72)
print("SYNTHESIS TABLE: target axes × PCs, RAW vs PC1-RESIDUALIZED")
print("="*72)

targets_raw = {
    "target_SALIENCE":      target_sal,
    "target_MOTION":        target_mot,
    "target_GOAL_DIRECTED": target_gld,
    "target_EQ_RUN":        target_eqr,
}

print(f"\n{'':<24} {'PC1':>8} {'PC2':>8} {'PC3':>8} {'PC4':>8} {'PC5':>8} {'PC6':>8}")
print(f"{'  RAW cosines:':<24}")
for tname, tvec in targets_raw.items():
    row = f"  {tname:<22}"
    for i in range(6):
        row += f" {cos(tvec, pca.components_[i]):>+7.3f}"
    print(row)

print(f"\n{'  PC1-RESIDUALIZED:':<24}")
for tname, tvec in targets_raw.items():
    tvec_res = project_out(tvec, pc1)
    tvec_res = tvec_res / np.linalg.norm(tvec_res)
    row = f"  {tname:<22}"
    for i in range(6):
        row += f" {cos(tvec_res, pca.components_[i]):>+7.3f}"
    print(row)


# ============================================================
# Save
# ============================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp53_results.npz",
    target_goal_directed=target_gld,
    target_eq_run_residual_pc1=target_eqr_res_unit,
    target_eq_run_residual_va=target_eqr_resVA_unit,
    target_goal_directed_residual_pc1=target_gld_res_unit,
    pca_components=pca.components_,
    pca_variance_ratio=pca.explained_variance_ratio_,
)
print("\nSaved: exp53_results.npz")
