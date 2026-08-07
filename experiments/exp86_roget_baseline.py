"""
exp86 — Roget's thesaurus 13-axis baseline.

Niamh's idea: if our cognitive 13-axis basis is doing real work (capturing
COGNITIVE primitive structure specifically), then 13 random antonym-pair axes
from Roget's thesaurus should explain LESS of cognitively-meaningful word
content while still doing decent on concrete/physical content.

If random Roget-13 explains COMPARABLE coverage to cognitive-13, then the
basis isn't capturing "cognitive primitives" specifically — it's just
capturing "any coherent antonym structure."

Builds 13 antonym-cluster axes from physical/sensory/perceptual contrasts that
are NOT obviously cognitive primitives. Each axis has 4-6 anchor pairs of
synonyms.
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
    n = len(offs)
    return unit(np.stack(offs).mean(axis=0)), n, len(pairs)


def build_R(wv):
    eq, _, _ = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    v, _, _ = build_axis(wv, VALENCE_PAIRS)
    a, _, _ = build_axis(wv, AROUSAL_PAIRS)
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


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)


# ============================================================================
# 13 Roget-thesaurus antonym-cluster axes
# Physical/sensory/perceptual contrasts that aren't obviously cognitive primitives
# ============================================================================
roget_axes_pairs = {
    "TEMPERATURE":  [("hot", "cold"), ("warm", "cool"), ("heated", "chilled"),
                     ("scorching", "freezing"), ("boiling", "frozen")],
    "MOISTURE":     [("wet", "dry"), ("damp", "arid"), ("soaked", "parched"),
                     ("moist", "dehydrated"), ("humid", "desiccated")],
    "SPEED":        [("fast", "slow"), ("quick", "sluggish"), ("rapid", "gradual"),
                     ("swift", "leisurely"), ("hasty", "deliberate")],
    "SIZE":         [("large", "small"), ("big", "tiny"), ("huge", "minuscule"),
                     ("vast", "minute"), ("enormous", "miniature")],
    "STRENGTH":     [("strong", "weak"), ("powerful", "feeble"), ("mighty", "fragile"),
                     ("robust", "frail"), ("sturdy", "delicate")],
    "TEXTURE":      [("smooth", "rough"), ("silky", "coarse"), ("polished", "jagged"),
                     ("sleek", "abrasive")],
    "TASTE":        [("sweet", "sour"), ("sugary", "bitter"), ("saccharine", "acrid"),
                     ("honeyed", "tart")],
    "AGE":          [("young", "old"), ("youthful", "elderly"), ("juvenile", "ancient"),
                     ("fresh", "aged")],
    "SOUND_VOLUME": [("loud", "quiet"), ("noisy", "silent"), ("booming", "hushed"),
                     ("deafening", "soundless"), ("clamorous", "muted")],
    "CLEANLINESS":  [("clean", "dirty"), ("pristine", "filthy"), ("spotless", "grimy"),
                     ("sanitary", "contaminated")],
    "HARDNESS":     [("hard", "soft"), ("firm", "mushy"), ("rigid", "pliable"),
                     ("solid", "flimsy")],
    "SHARPNESS":    [("sharp", "dull"), ("pointed", "blunt"), ("keen", "rounded"),
                     ("edged", "smooth")],
    "DENSITY":      [("dense", "sparse"), ("thick", "thin"), ("compact", "dispersed"),
                     ("crowded", "scattered")],
}


print("\nBuilding 13 Roget-thesaurus antonym axes...")
roget_axes = {}
for name, pairs in roget_axes_pairs.items():
    axis, used, total = build_axis(wv, pairs)
    roget_axes[name] = axis
    if used < total:
        missing = [(a, c) for a, c in pairs
                   if a not in wv.key_to_index or c not in wv.key_to_index]
        print(f"  {name}: {used}/{total} pairs used (missing: {missing})")
    else:
        print(f"  {name}: {used}/{total} pairs all in vocab")


# Also build our 13-axis cognitive basis for comparison
print("\nBuilding 13-axis cognitive basis (Basis-13b)...")


def residualize(axis, against):
    return unit(axis - (axis @ against) * against)


UD, _, _    = build_axis(wv, UP_DOWN_MML)
C, _, _     = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
W, _, _     = build_axis(wv, TARGET_WEIGHT_PAIRS)
ATT, _, _   = build_axis(wv, ATTENTION_CLEAN_PAIRS)
INT, _, _   = build_axis(wv, INTENTION_CLEAN_PAIRS)
R = build_R(wv)
D, _, _     = build_axis(wv, TARGET_SURPRISAL_PAIRS)
IO, _, _    = build_axis(wv, IN_OUT_MML_CLEAN)
DV, _, _    = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
MB, _, _    = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
EV, _, _    = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)
ABS, _, _   = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
REAL_IMAG, _, _ = build_axis(wv, REAL_IMAGINARY_PAIRS)

basis_13b = [UD, residualize(C, UD), W, ATT, residualize(INT, UD), R, D, IO,
             residualize(DV, UD), MB, EV, ABS, residualize(REAL_IMAG, UD)]

gs_cog = gs_orthogonalize(basis_13b)
gs_roget = gs_orthogonalize(list(roget_axes.values()))


# Random 13-axis baseline (multiple seeds, anisotropy-orthogonal)
random_bases = []
np.random.seed(0)
mu_unit = unit(mu)
for seed_i in range(20):
    rand_dirs = np.random.randn(13, 300)
    rand_dirs = rand_dirs - (rand_dirs @ mu_unit)[:, None] * mu_unit
    random_bases.append(gs_orthogonalize([rand_dirs[i] for i in range(13)]))


# ============================================================================
# Inter-axis cosines within Roget basis (sanity)
# ============================================================================
print("\n" + "=" * 78)
print("Roget-13 inter-axis cosines")
print("=" * 78)
names = list(roget_axes.keys())
M_roget = np.zeros((13, 13))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        M_roget[i, j] = cos(roget_axes[ni], roget_axes[nj]) if i != j else 1.0

print(f"\n{'':<14}" + "".join(f"{n[:6]:>7}" for n in names))
for i, ni in enumerate(names):
    row = f"{ni[:13]:<14}"
    for j in range(13):
        if i == j:
            row += f" 1.000 "
        else:
            row += f"{M_roget[i, j]:>+6.3f} "
    print(row)

abs_off = np.abs(M_roget - np.eye(13))
flat = abs_off[np.triu_indices(13, k=1)]
print(f"\nRoget-13 off-diagonal: max = {flat.max():.3f}, mean = {flat.mean():.3f}")
print(f"Pairs > 0.35: {(flat > 0.35).sum()}/{len(flat)}")

# Find biggest off-diagonal
i_m, j_m = np.unravel_index(np.argmax(abs_off), abs_off.shape)
print(f"Largest: {names[i_m]} ↔ {names[j_m]} = {M_roget[i_m, j_m]:+.4f}")


# ============================================================================
# Coverage comparison
# ============================================================================
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


print("\n" + "=" * 78)
print("COVERAGE COMPARISON")
print("=" * 78)
print(f"\n{'category':<28} {'Cog-13':>9} {'Roget-13':>10} {'Random-13':>11}")
print("-" * 62)

all_cog, all_roget = [], []
random_means = []

for cat, words in test_categories.items():
    cs_cog, cs_roget = [], []
    cs_random_per_basis = [[] for _ in random_bases]
    for w in words:
        v = get_deanisotropized(w)
        if v is None:
            continue
        cs_cog.append(coverage_in(v, gs_cog))
        cs_roget.append(coverage_in(v, gs_roget))
        for i, rb in enumerate(random_bases):
            cs_random_per_basis[i].append(coverage_in(v, rb))
    if not cs_cog:
        continue
    all_cog.extend(cs_cog)
    all_roget.extend(cs_roget)
    cog_m = np.mean(cs_cog)
    roget_m = np.mean(cs_roget)
    rand_means_for_cat = [np.mean(b) for b in cs_random_per_basis if b]
    rand_m = np.mean(rand_means_for_cat)
    rand_s = np.std(rand_means_for_cat)
    print(f"  {cat:<26} {cog_m * 100:>7.1f}% {roget_m * 100:>8.1f}% "
          f"{rand_m * 100:>6.1f}%±{rand_s * 100:.1f}")

print("-" * 62)
print(f"  {'OVERALL MEAN':<26} {np.mean(all_cog) * 100:>7.2f}% "
      f"{np.mean(all_roget) * 100:>8.2f}%")


# ============================================================================
# Cross-basis cosines: is Roget-13 orthogonal to Cog-13?
# ============================================================================
print("\n" + "=" * 78)
print("Cross-cosines between Roget-13 and Cog-13")
print("=" * 78)

cog_names = ["UD", "C_r", "W", "ATT", "INT_r", "R", "D", "IO",
             "DV_r", "MB", "EV", "ABS", "RI_r"]

print(f"\n{'':<14}" + "".join(f"{n:>7}" for n in cog_names))
for i, rn in enumerate(names):
    row = f"{rn[:13]:<14}"
    for j, cv in enumerate(basis_13b):
        c = cos(roget_axes[rn], cv)
        row += f"{c:>+7.3f}"
    print(row)


# Subspace overlap between Roget-13 and Cog-13
ROGET = np.stack(gs_roget)
COG = np.stack(gs_cog)
cross = ROGET @ COG.T
U_pa, sv, Vt_pa = np.linalg.svd(cross)
print(f"\nSubspace overlap (sum of squared principal-angle cosines): "
      f"{(sv ** 2).sum():.3f} out of 13 max")
print(f"  → fraction overlap: {(sv ** 2).sum() / 13 * 100:.1f}%")
print(f"\nPrincipal-angle cosines: {[f'{s:.3f}' for s in sv]}")
print(f"Number of principal angles with cos > 0.5: {(sv > 0.5).sum()}")
print(f"Number near-orthogonal (cos < 0.3): {(sv < 0.3).sum()}")


np.savez("/Users/macn/Documents/embeddingexp/exp86_results.npz",
         roget_axes_dict={n: v for n, v in roget_axes.items()},
         cross_cosines=cross,
         principal_angles=sv,
         mean_cog=float(np.mean(all_cog)),
         mean_roget=float(np.mean(all_roget)))
print("\nSaved exp86_results.npz")
