"""
exp90 — Greedy feature selection with cognitive-content objective + Roget in pool.

Fixes two issues from exp89:
  1. Includes 13 Roget physical-antonym axes in the candidate pool
  2. Uses cognitive-test-categories as the coverage objective (instead of 30K
     random vocab) — measures coverage of words that ARE cognitive content

Candidate pool (38 axes):
  12 cognitive: C, W, ATT, INT, R, D, IO, DV, MB, EV, ABS, REAL_IMAGINARY
  9 Lakoff:     UD, FB, LD, PATH, EXIST, FORCE, BAL, DIFF, COH
  4 cluster:    VALENCE, AROUSAL, SUC, LOSS  (COH is in Lakoff list above)
  13 Roget:     TEMPERATURE, MOISTURE, SPEED, SIZE, STRENGTH, TEXTURE,
                 TASTE, AGE, SOUND_VOLUME, CLEANLINESS, HARDNESS, SHARPNESS,
                 DENSITY

This is the same greedy procedure as exp89 but with the cognitive-coverage
objective the project actually cares about.
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


# ============================================================================
# Cognitive test categories — the objective
# ============================================================================
test_categories = {
    "EMOTIONS_AFFECTIVE": [
        "happiness", "sadness", "anger", "envy", "jealousy", "pride", "humility",
        "contentment", "longing", "delight", "melancholy", "rage", "elation",
        "despair", "serenity", "anguish", "disgust",
    ],
    "AGENTIVE_STATES": [
        "ambition", "determination", "resignation", "willpower", "discipline",
        "procrastination", "perseverance", "complacency", "vigilance",
        "diligence", "industriousness", "negligence", "carelessness",
    ],
    "SOCIAL_RELATIONAL": [
        "trust", "betrayal", "friendship", "enmity", "loyalty", "rivalry",
        "respect", "contempt", "admiration", "scorn", "gratitude", "resentment",
    ],
    "EPISTEMIC_STATES": [
        "knowledge", "ignorance", "belief", "doubt", "uncertainty",
        "conviction", "skepticism", "confidence", "hesitation",
        "speculation", "intuition", "memory", "forgetting",
    ],
    "CONCRETE_NOUNS": [
        "chair", "table", "dog", "stone", "tree", "river", "mountain",
        "hammer", "rope", "lamp", "cup", "window",
    ],
    "ABSTRACT_FORMAL": [
        "theorem", "philosophy", "ontology", "epistemology", "axiom",
        "principle", "framework", "paradigm", "schema", "abstraction",
    ],
    "MODAL_HYPOTHETICAL": [
        "hypothetical", "imaginary", "fictional", "speculative", "conjectural",
        "perhaps", "supposedly", "allegedly", "putative",
    ],
}

# Build the target sample: deanisotropized vectors of all test words
test_vectors = []
for cat, words in test_categories.items():
    for w in words:
        v = get_deanisotropized(w)
        if v is not None:
            test_vectors.append(v)
test_vectors = np.stack(test_vectors)  # (N_words, 300)
print(f"\nTest sample: {len(test_vectors)} cognitive words")


# ============================================================================
# Build all 38 candidate axes
# ============================================================================
print("\nBuilding candidate axes...")
candidates = {}

# Cognitive (12)
candidates["C"]   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
candidates["W"]   = build_axis(wv, TARGET_WEIGHT_PAIRS)
candidates["ATT"] = build_axis(wv, ATTENTION_CLEAN_PAIRS)
candidates["INT"] = build_axis(wv, INTENTION_CLEAN_PAIRS)
candidates["R"]   = build_R(wv)
candidates["D"]   = build_axis(wv, TARGET_SURPRISAL_PAIRS)
candidates["IO"]  = build_axis(wv, IN_OUT_MML_CLEAN)
candidates["DV"]  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
candidates["MB"]  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
candidates["EV"]  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)
candidates["ABS"] = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
candidates["REAL_IMAG"] = build_axis(wv, REAL_IMAGINARY_PAIRS)

# Lakoff image schemas (9)
candidates["UD"]    = build_axis(wv, UP_DOWN_MML)
candidates["FB"]    = build_axis(wv, FORWARD_BACK_MML)
candidates["LD"]    = build_axis(wv, LIGHT_DARK_MML)
candidates["PATH"]  = build_axis(wv, PATH_MOTION_MML)
candidates["EXIST"] = build_axis(wv, EXISTENCE_MML)
candidates["FORCE"] = build_axis(wv, FORCE_MML)
candidates["BAL"]   = build_axis(wv, BALANCE_MML)
candidates["DIFF"]  = build_axis(wv, DIFFICULTY_BURDEN_MML)
candidates["COH"]   = build_axis(wv, COHERENCE_PAIRS)

# Cluster (4 — COH already added)
candidates["VALENCE"] = build_axis(wv, VALENCE_PAIRS)
candidates["AROUSAL"] = build_axis(wv, AROUSAL_PAIRS)
candidates["SUC"] = build_axis(wv, SUCCESS_FAILURE_PAIRS)
candidates["LOSS"] = build_axis(wv, LOSS_PAIRS)

# Roget (13)
roget_pairs = {
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
for name, pairs in roget_pairs.items():
    candidates[name] = build_axis(wv, pairs)

print(f"  Total candidates: {len(candidates)}")


# ============================================================================
# Greedy selection on cognitive-test-vector sample
# ============================================================================
def basis_coverage_on_sample(gs_basis, sample):
    if not gs_basis:
        return 0.0
    B = np.stack(gs_basis)
    projections = sample @ B.T
    captured = (projections ** 2).sum(axis=1)
    explained = np.sqrt(np.maximum(0, captured))
    return float(explained.mean())


print("\n" + "=" * 88)
print(f"Greedy selection — objective: coverage of {len(test_vectors)} cognitive test words")
print("=" * 88)

selected_names = []
selected_orthonormal = []
remaining = dict(candidates)
prev_coverage = 0.0

print(f"\n{'step':<5} {'selected':<14} {'cov added':>10} {'cum cov':>9}  next-4-candidates")
print("-" * 100)

axis_origin = {}  # name → "cog" / "lakoff" / "cluster" / "roget"
cog_set = {"C", "W", "ATT", "INT", "R", "D", "IO", "DV", "MB", "EV", "ABS", "REAL_IMAG"}
lakoff_set = {"UD", "FB", "LD", "PATH", "EXIST", "FORCE", "BAL", "DIFF", "COH"}
cluster_set = {"VALENCE", "AROUSAL", "SUC", "LOSS"}
roget_set = set(roget_pairs.keys())
for n in candidates:
    if n in cog_set: axis_origin[n] = "cog"
    elif n in lakoff_set: axis_origin[n] = "lak"
    elif n in cluster_set: axis_origin[n] = "clu"
    elif n in roget_set: axis_origin[n] = "rog"

for step in range(1, 20):
    if not remaining:
        break

    candidate_results = []
    for name, axis in remaining.items():
        res = axis.copy()
        for u in selected_orthonormal:
            res = res - (res @ u) * u
        n = np.linalg.norm(res)
        if n < 1e-6:
            continue
        res = res / n
        trial = selected_orthonormal + [res]
        cov = basis_coverage_on_sample(trial, test_vectors)
        candidate_results.append((name, axis, res, cov))

    if not candidate_results:
        break

    candidate_results.sort(key=lambda x: -x[3])
    best_name, best_axis, best_res, best_cov = candidate_results[0]
    added = best_cov - prev_coverage

    top4_others = ", ".join(
        f"{n}({axis_origin[n][:3]}, {c * 100:.1f})"
        for n, _, _, c in candidate_results[1:5]
    )

    print(f"  {step:<3}  {best_name:<12} [{axis_origin[best_name]:<3}]  "
          f"+{added * 100:>6.2f}pp  {best_cov * 100:>6.2f}%   "
          f"{top4_others}")

    selected_names.append(best_name)
    selected_orthonormal.append(best_res)
    del remaining[best_name]
    prev_coverage = best_cov

    if added < 0.005 and step >= 8:
        print(f"  (added coverage dropped below 0.5pp at step {step})")
        break


print(f"\nFinal greedy basis ({len(selected_names)} axes):")
for i, name in enumerate(selected_names, 1):
    origin = axis_origin[name]
    print(f"  {i}. {name:<14} ({origin})")

# Composition breakdown
from collections import Counter
origins_picked = Counter(axis_origin[n] for n in selected_names)
print(f"\nComposition: {dict(origins_picked)}")
print(f"Final coverage: {prev_coverage * 100:.2f}%")

# Cog-13b comparison
basis_13b_axes = [
    candidates["UD"],
    residualize(candidates["C"], candidates["UD"]),
    candidates["W"], candidates["ATT"],
    residualize(candidates["INT"], candidates["UD"]),
    candidates["R"], candidates["D"], candidates["IO"],
    residualize(candidates["DV"], candidates["UD"]),
    candidates["MB"], candidates["EV"], candidates["ABS"],
    residualize(candidates["REAL_IMAG"], candidates["UD"]),
]
gs_13b = gs_orthogonalize(basis_13b_axes)
print(f"\nReference Cog-13b coverage on same sample: "
      f"{basis_coverage_on_sample(gs_13b, test_vectors) * 100:.2f}%")

np.savez("/Users/macn/Documents/embeddingexp/exp90_results.npz",
         selected_names=np.array(selected_names),
         final_coverage=prev_coverage)
print("\nSaved exp90_results.npz")
