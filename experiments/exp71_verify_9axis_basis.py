"""
exp71 — Verification reruns for the 9-axis basis (post-exp69).

Handoff §2 reruns, consolidated into one script:
  TEST 1 — 9x9 inter-axis cosine matrix (the new basis's mutual orthogonality)
  TEST 2 — Lakoff schema decomposition over the 9-axis basis
           (2a: raw cos(schema, axis); 2b: Gram-Schmidt 9D; 2c: 7D-vs-9D explained-variance)
  TEST 3 — Concept-word projections in 9D (agency/clinical, affective, IO/MB probes)
  TEST 4 — fear/anxiety/panic distinction under new basis (does it survive G→INT swap?)

Basis axes loaded:
  C, W, R, D, IO       — from exp60_results.npz (unchanged from original 7-axis basis)
  ATT_CLEAN, INT_CLEAN — from exp69_results.npz
  DV (= GATING),  MB   — from exp64_results.npz
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
)


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return unit(raw)


# ----------------------------------------------------------------------------
# Load saved axes
# ----------------------------------------------------------------------------
print("Loading exp60 + exp64 + exp69 saved arrays...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
exp64 = np.load("exp64_results.npz", allow_pickle=True)
exp69 = np.load("exp69_results.npz", allow_pickle=True)

C   = unit(basis_raw["C_rew"])
W   = unit(basis_raw["W_wgt"])
R   = unit(basis_raw["R_per"])
D   = unit(basis_raw["D_cmp"])
IO  = unit(basis_raw["IO_blk"])
ATT = unit(exp69["ATTENTION_CLEAN"])
INT = unit(exp69["INTENTION_CLEAN"])
DV  = unit(exp64["GATING"])           # renamed SELECTION → GATING → DV in exp64 A3
MB  = unit(exp64["MARKOV_BLANKET"])

# Deprecated A and G — needed for 7D-vs-9D comparison only
A_orig = unit(basis_raw["A_aff"])
G_orig = unit(basis_raw["G_pol"])

axis_names = ["C", "W", "ATT", "INT", "R", "D", "IO", "DV", "MB"]
axes       = [ C,   W,   ATT,   INT,   R,   D,   IO,   DV,   MB ]


# ============================================================================
# TEST 1 — 9x9 inter-axis cosine matrix
# ============================================================================
print("\n" + "=" * 84)
print("TEST 1 — 9-axis inter-axis cosines (post-exp69)")
print("=" * 84)
print(f"\n{'':<6}" + "".join(f"{n:>8}" for n in axis_names))

M = np.zeros((9, 9))
for i, a in enumerate(axes):
    row = f"{axis_names[i]:<6}"
    for j, b in enumerate(axes):
        if i == j:
            M[i, j] = 1.0
            row += f"   1.000"
        else:
            c = cos(a, b)
            M[i, j] = c
            row += f"  {c:+6.3f}"
    print(row)

abs_off = np.abs(M - np.eye(9))
flat = abs_off[np.triu_indices(9, k=1)]
print(f"\nOff-diagonal magnitude:  max={flat.max():.3f}  "
      f"mean={flat.mean():.3f}  median={np.median(flat):.3f}")
print(f"Pairs with |cos| > 0.20: {(flat > 0.20).sum()}/{len(flat)}")
print(f"Pairs with |cos| > 0.35: {(flat > 0.35).sum()}/{len(flat)}  "
      "(handoff target: 0)")

print("\nSpecific pairs of interest:")
print(f"  cos(ATT, INT) = {cos(ATT, INT):+.4f}  "
      "(was +0.561 with A_orig, G_orig — the deprecated entanglement)")
print(f"  cos(C, W)     = {cos(C, W):+.4f}  "
      "(structural value-cost anti-correlation)")
print(f"  cos(MB, IO)   = {cos(MB, IO):+.4f}  "
      "(the two Markov-blanket-related axes)")
print(f"  cos(MB, DV)   = {cos(MB, DV):+.4f}  "
      "(the two added axes)")


# ============================================================================
# TEST 2 — Lakoff schema decomposition over 9-axis basis
# ============================================================================
schemas = {
    "UD":       build_axis(UP_DOWN_MML),
    "FB":       build_axis(FORWARD_BACK_MML),
    "LD":       build_axis(LIGHT_DARK_MML),
    "IO_CLEAN": IO,
    "PATH":     build_axis(PATH_MOTION_MML),
    "EXIST":    build_axis(EXISTENCE_MML),
    "FORCE":    build_axis(FORCE_MML),
    "BAL":      build_axis(BALANCE_MML),
    "DIFF":     build_axis(DIFFICULTY_BURDEN_MML),
    "COH":      build_axis(COHERENCE_PAIRS),
    "SUC":      build_axis(SUCCESS_FAILURE_PAIRS),
    "LOSS":     build_axis(LOSS_PAIRS),
    "VALENCE":  build_axis(VALENCE_PAIRS),
    "AROUSAL":  build_axis(AROUSAL_PAIRS),
}

# ----------------------------------------------------------------------------
# TEST 2a — raw cos(schema, new basis axis)
# ----------------------------------------------------------------------------
print("\n" + "=" * 92)
print("TEST 2a — cos(Lakoff schema, new basis axis), dominant-axis assignment")
print("=" * 92)
print(f"\n{'schema':<10}" + " ".join(f"{n:>7}" for n in axis_names) + " | dominant")
print("-" * 92)
for sname, svec in schemas.items():
    cs = [cos(svec, a) for a in axes]
    abs_cs = [abs(c) for c in cs]
    dom_idx = int(np.argmax(abs_cs))
    row = f"{sname:<10}"
    for c in cs:
        row += f" {c:+7.3f}"
    row += f" | {axis_names[dom_idx]} ({cs[dom_idx]:+.3f})"
    print(row)

# Also report old basis dominant-axis for comparison
print(f"\n{'schema':<10}  new-basis dominant   old-basis dominant   shift?")
print("-" * 70)
old_axis_names = ["C", "W", "A_orig", "G_orig", "R", "D", "IO"]
old_axes_list  = [ C,   W,   A_orig,   G_orig,   R,   D,   IO ]
for sname, svec in schemas.items():
    new_cs = [cos(svec, a) for a in axes]
    old_cs = [cos(svec, a) for a in old_axes_list]
    new_dom = axis_names[int(np.argmax(np.abs(new_cs)))]
    old_dom = old_axis_names[int(np.argmax(np.abs(old_cs)))]
    shift = "SHIFT" if new_dom != old_dom else ""
    # Map: A_orig → ATT, G_orig → INT for "non-shift" interpretation
    if old_dom == "A_orig" and new_dom == "ATT": shift = "(A→ATT, same primitive)"
    if old_dom == "G_orig" and new_dom == "INT": shift = "(G→INT, same primitive)"
    print(f"  {sname:<10}  {new_dom:<18}  {old_dom:<18}  {shift}")


# ----------------------------------------------------------------------------
# TEST 2b — Gram-Schmidt 9D, schemas projected
# ----------------------------------------------------------------------------
# GS order: most independent first, most entangled last.
# From exp69's data + 7-axis matrix:
#   MB has max |cos| ≈ 0.137 (cleanest)
#   C, W are the structural pair (cos = -0.344)
#   ATT and INT still have +0.18 (with C) and +0.21 (with IO) respectively
# So: C → W → MB → D → IO → R → DV → ATT → INT
gs_order = ["C", "W", "MB", "D", "IO", "R", "DV", "ATT", "INT"]
gs_idx = [axis_names.index(n) for n in gs_order]
gs_basis = []
for i in gs_idx:
    u = axes[i].copy()
    for prev in gs_basis:
        u = u - (u @ prev) * prev
    u = unit(u)
    gs_basis.append(u)


def project_gs(v):
    coords = {}
    v_residual = v.copy()
    for n, u_gs in zip(gs_order, gs_basis):
        c = float(v_residual @ u_gs)
        coords[n] = c
        v_residual = v_residual - c * u_gs
    residual_norm = float(np.linalg.norm(v_residual))
    explained = float(np.sqrt(max(0, 1 - residual_norm ** 2)))
    return coords, explained


print("\n" + "=" * 100)
print(f"TEST 2b — Lakoff schemas in Gram-Schmidt 9D")
print(f"  GS order: {' → '.join(gs_order)}")
print("=" * 100)
print(f"\n{'schema':<10}" + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
print("-" * 96)
for sname, svec in schemas.items():
    coords, expl = project_gs(svec)
    row = f"{sname:<10}"
    for n in gs_order:
        row += f" {coords[n]:>+7.3f}"
    row += f" | {expl * 100:>4.1f}%"
    print(row)


# ----------------------------------------------------------------------------
# TEST 2c — 7D (old basis) vs 9D (new basis) explained-variance
# ----------------------------------------------------------------------------
gs_order_7d = ["C", "W", "IO", "A_orig", "D", "R", "G_orig"]
old_axes_dict = {"C": C, "W": W, "IO": IO, "A_orig": A_orig,
                 "D": D, "R": R, "G_orig": G_orig}
gs_basis_7d = []
for n in gs_order_7d:
    u = old_axes_dict[n].copy()
    for prev in gs_basis_7d:
        u = u - (u @ prev) * prev
    u = unit(u)
    gs_basis_7d.append(u)


def explained_in_basis(v, gs_b):
    v_residual = v.copy()
    for u_gs in gs_b:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


print("\n" + "=" * 84)
print("TEST 2c — Schema explained-variance: 7D (old basis) vs 9D (new basis)")
print("=" * 84)
print(f"\n{'schema':<10} {'7D expl':>8} {'9D expl':>8} {'Δ':>6}")
print("-" * 38)
for sname, svec in schemas.items():
    e7 = explained_in_basis(svec, gs_basis_7d)
    e9 = explained_in_basis(svec, gs_basis)
    d = (e9 - e7) * 100
    print(f"  {sname:<10}  {e7 * 100:>6.1f}%  {e9 * 100:>6.1f}%  {d:>+5.1f}%")


# ============================================================================
# TEST 3 — Concept word projections (exp58 style)
# ============================================================================
print("\n" + "=" * 100)
print("TEST 3 — Concept words in 9D Gram-Schmidt basis")
print("=" * 100)

groups = [
    ("AGENCY + CLINICAL", [
        "hope", "freedom", "agency", "growth", "play", "love",
        "psychosis", "trauma", "depression", "grief", "anxiety", "fear",
        "panic", "shame", "guilt", "burnout", "dissociation",
    ]),
    ("AFFECTIVE + CONTEMPLATIVE", [
        "joy", "rage", "compassion", "boredom", "curiosity",
        "awe", "sublime", "transcendence", "ineffable", "presence",
        "ritual", "meditation", "creativity", "flow",
    ]),
    ("IO/MB PROBES (self/other content)", [
        "self", "selfhood", "identity", "boundary", "intimacy",
        "belonging", "alienation", "isolation", "communion", "exile",
        "membership", "kinship", "loneliness", "togetherness",
        "ego", "soul", "individuality", "autonomy", "sovereignty",
        "interiority", "subjectivity",
    ]),
    ("RANDOM NOUNS (control)", ["sausage", "pyjamas", "marigold", "stapler", "pebble"]),
]

for group_name, words in groups:
    print(f"\n--- {group_name} ---")
    print(f"{'word':<15} " + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
    print("-" * 100)
    for w in words:
        if w not in wv.key_to_index:
            print(f"  {w:<15} (not in GloVe)")
            continue
        v = unit(wv[w])
        coords, expl = project_gs(v)
        row = f"  {w:<15}"
        for n in gs_order:
            row += f" {coords[n]:>+7.3f}"
        row += f" | {expl * 100:>4.1f}%"
        print(row)


# ============================================================================
# TEST 4 — fear/anxiety/panic distinction (specific check)
# ============================================================================
print("\n" + "=" * 84)
print("TEST 4 — fear/anxiety/panic distinction under new basis")
print("=" * 84)
print()
print("Under old basis (BASIS_REFERENCE §5, exp58 6D space):")
print("  fear had G_pol = +0.191 (highest among the clinical states)")
print("  anxiety had G_pol = +0.011 (near-zero — the 'no policy' signature)")
print("  panic had R_per = -0.207 (strongest precision collapse)")
print()
print("If the fear-has-policy reading is real, fear should now have high INT_CLEAN.")
print(f"\n{'word':<10}" + " ".join(f"{n:>8}" for n in ["C", "INT", "ATT", "R", "DV", "MB"]))
for w in ["fear", "anxiety", "panic", "trauma", "depression", "grief", "psychosis"]:
    if w not in wv.key_to_index:
        continue
    v = unit(wv[w])
    row = f"{w:<10}"
    for n in ["C", "INT", "ATT", "R", "DV", "MB"]:
        idx = axis_names.index(n)
        row += f" {cos(v, axes[idx]):>+8.4f}"
    print(row)


# ============================================================================
# Save
# ============================================================================
np.savez("/Users/macn/Documents/embeddingexp/exp71_results.npz",
         axis_names=np.array(axis_names),
         inter_axis_matrix=M,
         gs_order=np.array(gs_order))
print("\nSaved exp71_results.npz")
