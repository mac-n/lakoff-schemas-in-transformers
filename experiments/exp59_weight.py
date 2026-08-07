"""
exp59: target_WEIGHT — Niamh's hypothesis that PC2's missing piece is weight.

The intuition: PC2 currently has G (goal-directedness) + affect-content, but
neither EE nor pure-G fully describe it. Lakoff's DIFFICULTIES ARE BURDENS
primary metaphor encodes weight as the somatic substrate for difficulty.
Phenomenologically: hope/freedom/agency feel LIGHT; depression/grief/anxiety
feel HEAVY. Weight in language tracks *resistance to action* — the inertial
component that pure goal-directedness misses.

Tests:
  1. cos(target_WEIGHT, PC1-6) — does it land on PC2 specifically?
  2. cos(target_WEIGHT, basis axes A/C/D/G/R/EE) — orthogonal to A?
     (Cleaner than G's +0.56 entanglement with A?)
  3. cos(target_WEIGHT, each Lakoff schema) — does it capture DIFF
     particularly?
  4. Rerun concept-word projection in 7D basis (adding W) — does adding
     weight bring DIFF and the pathological-state explained-fractions up?
  5. Predictions:
     - Healthy capacity-words (hope, freedom, agency, play) — negative W (light)
     - Pathological states (depression, grief, anxiety, trauma) — positive W (heavy)
     - DIFF schema heavily on W
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
    TARGET_SALIENCE_PAIRS, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS,
)
from exp53_residual_and_goal_directed import TARGET_GOAL_DIRECTED_PAIRS
from exp54_pc1_comparator import TARGET_REWARD_COMPOSITE_PAIRS, TARGET_SURPRISAL_PAIRS
from exp56_explore_exploit import TARGET_EXPLOIT_EXPLORE_PAIRS

from sklearn.decomposition import PCA


# Build target_WEIGHT from heavy/light vocabulary.
# Note: must avoid LIGHT_DARK overlap. LD_PAIRS uses "light, bright" etc. —
# these refer to illumination, not weight. To disambiguate, use specifically
# weight-domain vocabulary. Avoid bare "light" since it's polysemous and
# already in LD anchors.
TARGET_WEIGHT_PAIRS = [
    ("heavy",        "weightless"),
    ("weighty",      "airy"),
    ("ponderous",    "buoyant"),
    ("burdensome",   "effortless"),
    ("laden",        "unburdened"),
    ("cumbersome",   "nimble"),
    ("leaden",       "feathery"),
    ("dense",        "wispy"),
    ("encumbered",   "unencumbered"),
    ("heavyweight",  "featherweight"),
    ("massive",      "delicate"),
    ("oppressive",   "lighthearted"),
]


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
        print(f"  [{label}] missing: {sorted(set(missing))}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


def project_out(v, u):
    u_unit = u / np.linalg.norm(u)
    return v - (v @ u_unit) * u_unit


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# Build the 11 cluster axes for the PCA
axes = {}
axes["VALENCE"]   = build_axis(VALENCE_PAIRS, "VALENCE")
axes["AROUSAL"]   = build_axis(AROUSAL_PAIRS, "AROUSAL")
axes["UD"]        = build_axis(UP_DOWN_MML, "UD")
axes["FB"]        = build_axis(FORWARD_BACK_MML, "FB")
axes["LD"]        = build_axis(LIGHT_DARK_MML, "LD")
axes["PATH"]      = build_axis(PATH_MOTION_MML, "PATH")
axes["EXIST"]     = build_axis(EXISTENCE_MML, "EXIST")
axes["BAL"]       = build_axis(BALANCE_MML, "BAL")
axes["COHERENCE"] = build_axis(COHERENCE_PAIRS, "COHERENCE")
axes["SUCCESS"]   = build_axis(SUCCESS_FAILURE_PAIRS, "SUCCESS")
axes["LOSS"]      = build_axis(LOSS_PAIRS, "LOSS")
axis_names = list(axes.keys())
M = np.stack([axes[n] for n in axis_names])
pca = PCA(n_components=10)
pca.fit(M)


# Build the 6D AI-plus basis + W
A_axis = build_axis(TARGET_SALIENCE_PAIRS, "A")
C_axis = build_axis(TARGET_REWARD_COMPOSITE_PAIRS, "C")
D_axis = build_axis(TARGET_SURPRISAL_PAIRS, "D")
G_axis = build_axis(TARGET_GOAL_DIRECTED_PAIRS, "G")
EE_axis = build_axis(TARGET_EXPLOIT_EXPLORE_PAIRS, "EE")
EQ_raw = build_axis(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, "EQ_raw")
R_axis = project_out(EQ_raw, axes["VALENCE"])
R_axis = project_out(R_axis, axes["AROUSAL"])
R_axis = R_axis / np.linalg.norm(R_axis)
W_axis = build_axis(TARGET_WEIGHT_PAIRS, "W")


# ============================================================
# TEST 1: W vs PCs
# ============================================================
print("\n" + "="*72)
print("TEST 1: cos(target_WEIGHT, PC1-6)")
print("="*72)
print(f"\n{'':<10}" + "".join(f"{n:>10}" for n in [f'PC{i+1}' for i in range(6)]))
print(f"{'WEIGHT':<10}" + "".join(f"{cos(W_axis, pca.components_[i]):>+10.3f}" for i in range(6)))
print(f"\nReminder (from exp52):")
print(f"  cos(G_GOAL_DIRECTED, PC2) = -0.303")
print(f"  cos(target_MOTION,    PC2) = +0.051")
print(f"  cos(EE_EXPLOIT_EXPLORE, PC2) = +0.064")


# ============================================================
# TEST 2: W orthogonality with the other 6D basis axes
# ============================================================
print("\n" + "="*72)
print("TEST 2: W's orthogonality to the 6D basis axes")
print("="*72)
print(f"\n  cos(W, A_RUSSELL)      = {cos(W_axis, A_axis):>+.4f}")
print(f"  cos(W, C_REWARD)       = {cos(W_axis, C_axis):>+.4f}")
print(f"  cos(W, D_SURPRISAL)    = {cos(W_axis, D_axis):>+.4f}")
print(f"  cos(W, G_GOAL_DIR)     = {cos(W_axis, G_axis):>+.4f}")
print(f"  cos(W, R_PERCEPT_PREC) = {cos(W_axis, R_axis):>+.4f}")
print(f"  cos(W, EE_EXP_EXP)     = {cos(W_axis, EE_axis):>+.4f}")


# ============================================================
# TEST 3: W's input-axis loadings and nearest neighbors
# ============================================================
print("\n" + "="*72)
print("TEST 3: W's loading on input axes + nearest-neighbor words")
print("="*72)
print(f"\n  Input-axis loadings:")
ipt = [(n, cos(W_axis, axes[n])) for n in axis_names]
ipt.sort(key=lambda x: abs(x[1]), reverse=True)
for n, c in ipt[:6]:
    print(f"    {n:>10}: {c:>+.4f}")

print(f"\n  Heavy pole (positive):")
for w, s in wv.similar_by_vector(W_axis, topn=10):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Light pole (negative):")
for w, s in wv.similar_by_vector(-W_axis, topn=10):
    print(f"    {s:>+.4f}  {w}")


# ============================================================
# TEST 4: cos(W, each Lakoff schema) — does DIFF stand out?
# ============================================================
print("\n" + "="*72)
print("TEST 4: W's cosine with each Lakoff schema")
print("="*72)
schemas = {
    "UD":        build_axis(UP_DOWN_MML),
    "FB":        build_axis(FORWARD_BACK_MML),
    "LD":        build_axis(LIGHT_DARK_MML),
    "IO_CLEAN":  build_axis(IN_OUT_MML_CLEAN),
    "PATH":      build_axis(PATH_MOTION_MML),
    "EXIST":     build_axis(EXISTENCE_MML),
    "FORCE":     build_axis(FORCE_MML),
    "BAL":       build_axis(BALANCE_MML),
    "DIFF":      build_axis(DIFFICULTY_BURDEN_MML),
    "COH":       build_axis(COHERENCE_PAIRS),
    "SUC":       build_axis(SUCCESS_FAILURE_PAIRS),
    "LOSS":      build_axis(LOSS_PAIRS),
}
print()
schema_cos = [(n, cos(W_axis, v)) for n, v in schemas.items()]
schema_cos.sort(key=lambda x: abs(x[1]), reverse=True)
for n, c in schema_cos:
    print(f"  cos(W, {n:<10}) = {c:>+.4f}")


# ============================================================
# TEST 5: project concept words onto 7D basis (adding W)
# ============================================================
print("\n" + "="*72)
print("TEST 5: concept words in 7D basis (W added)")
print("="*72)

# Gram-Schmidt order: most-independent first, G last
basis_raw = {
    "C_rew": C_axis, "A_aff": A_axis, "D_cmp": D_axis,
    "W_wgt": W_axis, "R_per": R_axis, "EE_x": EE_axis, "G_pol": G_axis,
}
gs_order = ["C_rew", "A_aff", "D_cmp", "W_wgt", "R_per", "EE_x", "G_pol"]
gs_basis = []
for name in gs_order:
    u = basis_raw[name].copy()
    for prev in gs_basis:
        u = u - (u @ prev) * prev
    u = u / np.linalg.norm(u)
    gs_basis.append(u)


def project_gs(v):
    coords = {}
    v_residual = v.copy()
    for name, u_gs in zip(gs_order, gs_basis):
        c = float(v_residual @ u_gs)
        coords[name] = c
        v_residual = v_residual - c * u_gs
    residual_norm = float(np.linalg.norm(v_residual))
    explained = float(np.sqrt(max(0, 1 - residual_norm**2)))
    return coords, explained


# Concept words — same set as exp57/58
concept_words = [
    # capacity / agency
    "hope", "freedom", "agency", "growth", "play", "love", "wisdom",
    # clinical / pathological
    "psychosis", "trauma", "depression", "grief", "anxiety", "fear",
    "panic", "shame", "guilt", "burnout",
    # affective
    "joy", "rage", "compassion", "boredom", "curiosity",
    # contemplative
    "awe", "sublime", "transcendence", "ineffable", "presence",
    # other concepts
    "ritual", "meditation", "creativity", "flow", "regulation",
    "betrayal", "obsession", "addiction",
]

print(f"\n{'word':<14} " + " ".join(f"{n[:6]:>7}" for n in gs_order) + f" | {'expl':>5}")
print("-"*92)
for w in concept_words:
    if w not in wv.key_to_index:
        print(f"  {w:<14}  (not in GloVe)")
        continue
    v = wv[w] / np.linalg.norm(wv[w])
    coords, expl = project_gs(v)
    row = f"  {w:<14}"
    for n in gs_order:
        row += f" {coords[n]:>+7.3f}"
    row += f" | {expl*100:>4.1f}%"
    print(row)


# ============================================================
# TEST 6: DIFF specifically — does adding W bring its explained up?
# ============================================================
print("\n" + "="*72)
print("TEST 6: DIFF before vs after adding W to basis")
print("="*72)

# Project DIFF in 6D basis (no W) and 7D basis (with W)
gs_order_6d = ["C_rew", "A_aff", "D_cmp", "R_per", "EE_x", "G_pol"]
gs_basis_6d = []
for name in gs_order_6d:
    u = basis_raw[name].copy()
    for prev in gs_basis_6d:
        u = u - (u @ prev) * prev
    u = u / np.linalg.norm(u)
    gs_basis_6d.append(u)

def project_gs_basis(v, basis_list, names):
    coords = {}
    v_residual = v.copy()
    for name, u_gs in zip(names, basis_list):
        c = float(v_residual @ u_gs)
        coords[name] = c
        v_residual = v_residual - c * u_gs
    residual_norm = float(np.linalg.norm(v_residual))
    explained = float(np.sqrt(max(0, 1 - residual_norm**2)))
    return coords, explained


print()
print(f"DIFF schema:")
coords_6d, expl_6d = project_gs_basis(schemas["DIFF"], gs_basis_6d, gs_order_6d)
coords_7d, expl_7d = project_gs_basis(schemas["DIFF"], gs_basis, gs_order)
print(f"  6D basis (no W) — explained: {expl_6d*100:.1f}%")
for n in gs_order_6d:
    print(f"    {n}: {coords_6d[n]:>+.4f}")
print(f"  7D basis (+W) — explained: {expl_7d*100:.1f}%")
for n in gs_order:
    print(f"    {n}: {coords_7d[n]:>+.4f}")

# Same for all schemas
print(f"\nAll schemas: 6D explained vs 7D explained (with W)")
print(f"{'schema':<12} {'6D expl':>8} {'7D expl':>8} {'Δ':>6}")
for sname, svec in schemas.items():
    _, e6 = project_gs_basis(svec, gs_basis_6d, gs_order_6d)
    _, e7 = project_gs_basis(svec, gs_basis, gs_order)
    d = e7 - e6
    print(f"  {sname:<10}  {e6*100:>6.1f}%  {e7*100:>6.1f}%  {d*100:>+5.1f}%")
