"""
exp60: Final 7-axis basis — drop EE (didn't pan out), add W and IO_CLEAN.

The cleaned basis after exp52-59:
  C  = integrated reward / expected value of state
  W  = weight / effort-cost / felt-burden
  A  = affect quality (Russell V x A diagonal)
  G  = policy precision / goal-directedness
  R  = perceptual precision / regulability
  D  = compression / surprisal / predictability
  IO = Markov-blanket / self-vs-environment boundary

Active-inference reading:
  - C, W are the "value-vs-cost" pair on PC1
  - G, R are the two precision signals
  - A is somatic-felt-state (separate from PC1-as-reward)
  - D is the predictability/compression signal
  - IO is the SUBSTRATE primitive — what constitutes an agent at all,
    structurally prior to the content primitives. This explains why
    IO_CLEAN was robustly Tier-2-independent across exp21-57.

EE_EXPLOIT_EXPLORE dropped: cos(EE, PC2) = +0.06 (failed to capture PC2),
and EE's word-vector vocabulary was corporate-optimization vs scientific-
investigation register rather than a cognitive primitive. Niamh's call.

Tests:
  1. Inter-axis orthogonality of the 7-axis basis
  2. Lakoff schemas projected onto the new 7D basis
  3. Concept words projected, including new IO-probe words
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
from exp59_weight import TARGET_WEIGHT_PAIRS


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


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# Build the final 7-axis basis
C_axis  = build_axis(TARGET_REWARD_COMPOSITE_PAIRS)
W_axis  = build_axis(TARGET_WEIGHT_PAIRS)
A_axis  = build_axis(TARGET_SALIENCE_PAIRS)
G_axis  = build_axis(TARGET_GOAL_DIRECTED_PAIRS)
D_axis  = build_axis(TARGET_SURPRISAL_PAIRS)
EQ_raw  = build_axis(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
VALENCE = build_axis(VALENCE_PAIRS)
AROUSAL = build_axis(AROUSAL_PAIRS)
R_axis  = project_out(EQ_raw, VALENCE)
R_axis  = project_out(R_axis, AROUSAL)
R_axis  = R_axis / np.linalg.norm(R_axis)
IO_axis = build_axis(IN_OUT_MML_CLEAN)

basis_raw = {
    "C_rew":  C_axis,
    "W_wgt":  W_axis,
    "A_aff":  A_axis,
    "G_pol":  G_axis,
    "R_per":  R_axis,
    "D_cmp":  D_axis,
    "IO_blk": IO_axis,
}


# ============================================================
# TEST 1: inter-axis orthogonality of the 7-axis basis
# ============================================================
print("\n" + "="*84)
print("FINAL 7-AXIS BASIS — inter-axis cosines (no EE, with W and IO)")
print("="*84)
print()
names = list(basis_raw.keys())
print(f"{'':<10}" + "".join(f"{n:>10}" for n in names))
for n1 in names:
    row = f"{n1:<10}"
    for n2 in names:
        if n1 == n2:
            row += f"{'  1.00':<10}"
        else:
            row += f"{cos(basis_raw[n1], basis_raw[n2]):>+10.3f}"
    print(row)


# ============================================================
# Gram-Schmidt: order = most-independent first, G last
# (G has +0.56 entanglement with A; G last lets others retain their
#  independent content)
# ============================================================
gs_order = ["C_rew", "W_wgt", "IO_blk", "A_aff", "D_cmp", "R_per", "G_pol"]
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


# ============================================================
# TEST 2: Lakoff schemas in final 7-axis basis
# ============================================================
print("\n" + "="*84)
print("LAKOFF SCHEMAS in final 7-axis basis (Gram-Schmidt orthogonalized)")
print("="*84)
print()

schemas = {
    "UD":        build_axis(UP_DOWN_MML),
    "FB":        build_axis(FORWARD_BACK_MML),
    "LD":        build_axis(LIGHT_DARK_MML),
    "IO_CLEAN":  IO_axis,  # is a basis vector now, will be 100%
    "PATH":     build_axis(PATH_MOTION_MML),
    "EXIST":     build_axis(EXISTENCE_MML),
    "FORCE":     build_axis(FORCE_MML),
    "BAL":       build_axis(BALANCE_MML),
    "DIFF":      build_axis(DIFFICULTY_BURDEN_MML),
    "COH":       build_axis(COHERENCE_PAIRS),
    "SUC":       build_axis(SUCCESS_FAILURE_PAIRS),
    "LOSS":      build_axis(LOSS_PAIRS),
    "VALENCE":   VALENCE,
    "AROUSAL":   AROUSAL,
}

print(f"{'schema':<10} " + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
print("-"*82)
for sname, svec in schemas.items():
    coords, expl = project_gs(svec)
    row = f"{sname:<10}"
    for n in gs_order:
        row += f" {coords[n]:>+7.3f}"
    row += f" | {expl*100:>4.1f}%"
    print(row)


# ============================================================
# TEST 3: concept words including IO-probes
# ============================================================
print("\n" + "="*84)
print("CONCEPT WORDS in final 7-axis basis")
print("="*84)

# Concept word groups
io_probes = ["self", "selfhood", "identity", "boundary", "intimacy",
             "belonging", "alienation", "isolation", "communion", "exile",
             "membership", "kinship", "loneliness", "togetherness"]

agency_and_clinical = [
    "hope", "freedom", "agency", "growth", "play", "love",
    "psychosis", "trauma", "depression", "grief", "anxiety", "fear",
    "panic", "shame", "guilt", "burnout", "dissociation",
]

affective_and_contemplative = [
    "joy", "rage", "compassion", "boredom", "curiosity",
    "awe", "sublime", "transcendence", "ineffable", "presence",
    "ritual", "meditation", "creativity", "flow",
]

random_nouns = ["sausage", "pyjamas", "marigold", "stapler", "pebble"]

groups = [
    ("IO PROBES (self/other/boundary content)", io_probes),
    ("AGENCY + CLINICAL", agency_and_clinical),
    ("AFFECTIVE + CONTEMPLATIVE", affective_and_contemplative),
    ("RANDOM NOUNS (control)", random_nouns),
]

for group_name, words in groups:
    print(f"\n--- {group_name} ---")
    print(f"{'word':<15} " + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
    print("-"*87)
    for w in words:
        if w not in wv.key_to_index:
            print(f"  {w:<15} (not in GloVe)")
            continue
        v = wv[w] / np.linalg.norm(wv[w])
        coords, expl = project_gs(v)
        row = f"  {w:<15}"
        for n in gs_order:
            row += f" {coords[n]:>+7.3f}"
        row += f" | {expl*100:>4.1f}%"
        print(row)


# ============================================================
# TEST 4: how does adding IO_CLEAN as basis affect other schemas?
# ============================================================
print("\n" + "="*84)
print("Schema 6D-vs-7D comparison (adding IO_CLEAN to basis)")
print("="*84)

# Build 6D basis (no IO_CLEAN)
gs_order_6d = ["C_rew", "W_wgt", "A_aff", "D_cmp", "R_per", "G_pol"]
gs_basis_6d = []
for name in gs_order_6d:
    u = basis_raw[name].copy()
    for prev in gs_basis_6d:
        u = u - (u @ prev) * prev
    u = u / np.linalg.norm(u)
    gs_basis_6d.append(u)


def project_basis(v, basis_list, names):
    v_residual = v.copy()
    for u_gs in basis_list:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual)**2)))


print(f"\n{'schema':<12} {'6D expl':>8} {'7D expl':>8} {'Δ':>6}")
for sname, svec in schemas.items():
    if sname == "IO_CLEAN":
        continue
    e6 = project_basis(svec, gs_basis_6d, gs_order_6d)
    e7 = project_basis(svec, gs_basis, gs_order)
    d = (e7 - e6) * 100
    print(f"  {sname:<10}  {e6*100:>6.1f}%  {e7*100:>6.1f}%  {d:>+5.1f}%")

print("\nSaved nothing — final basis arrays still in memory")
np.savez(
    "/Users/macn/Documents/embeddingexp/exp60_results.npz",
    basis_raw={n: v for n, v in basis_raw.items()},
    gs_basis=np.stack(gs_basis),
    gs_order=np.array(gs_order),
)
print("Saved: exp60_results.npz")
