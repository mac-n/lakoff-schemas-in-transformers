"""
exp92 — Expanded Roget pool + MASCULINE-FEMININE test.

Two things at once:
  1. Build 12 new Roget-style antonym categories (beyond the original 13)
  2. Build MASCULINE-FEMININE explicitly, test cos(MF, HARDNESS) — the gender hypothesis
  3. Rerun greedy on cognitive test categories with expanded pool

If HARDNESS captures cultural gender via embodied metaphor, then
cos(MF, HARDNESS) should be substantial (>0.5).
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
    used = [(a, c) for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    return unit(np.stack(offs).mean(axis=0)), used


def build_R(wv):
    eq, _ = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    v, _ = build_axis(wv, VALENCE_PAIRS)
    a, _ = build_axis(wv, AROUSAL_PAIRS)
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


# ============================================================================
# Build full candidate pool
# ============================================================================
print("Building full candidate pool...")

candidates = {}

# Cognitive (12)
candidates["C"]   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)[0]
candidates["W"]   = build_axis(wv, TARGET_WEIGHT_PAIRS)[0]
candidates["ATT"] = build_axis(wv, ATTENTION_CLEAN_PAIRS)[0]
candidates["INT"] = build_axis(wv, INTENTION_CLEAN_PAIRS)[0]
candidates["R"]   = build_R(wv)
candidates["D"]   = build_axis(wv, TARGET_SURPRISAL_PAIRS)[0]
candidates["IO"]  = build_axis(wv, IN_OUT_MML_CLEAN)[0]
candidates["DV"]  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)[0]
candidates["MB"]  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)[0]
candidates["EV"]  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)[0]
candidates["ABS"] = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)[0]
candidates["REAL_IMAG"] = build_axis(wv, REAL_IMAGINARY_PAIRS)[0]

# Lakoff (9)
candidates["UD"]    = build_axis(wv, UP_DOWN_MML)[0]
candidates["FB"]    = build_axis(wv, FORWARD_BACK_MML)[0]
candidates["LD"]    = build_axis(wv, LIGHT_DARK_MML)[0]
candidates["PATH"]  = build_axis(wv, PATH_MOTION_MML)[0]
candidates["EXIST"] = build_axis(wv, EXISTENCE_MML)[0]
candidates["FORCE"] = build_axis(wv, FORCE_MML)[0]
candidates["BAL"]   = build_axis(wv, BALANCE_MML)[0]
candidates["DIFF"]  = build_axis(wv, DIFFICULTY_BURDEN_MML)[0]
candidates["COH"]   = build_axis(wv, COHERENCE_PAIRS)[0]

# Cluster (4)
candidates["VALENCE"] = build_axis(wv, VALENCE_PAIRS)[0]
candidates["AROUSAL"] = build_axis(wv, AROUSAL_PAIRS)[0]
candidates["SUC"] = build_axis(wv, SUCCESS_FAILURE_PAIRS)[0]
candidates["LOSS"] = build_axis(wv, LOSS_PAIRS)[0]

# Original Roget (13)
original_roget_pairs = {
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
for name, pairs in original_roget_pairs.items():
    candidates[name] = build_axis(wv, pairs)[0]

# NEW Roget categories (12)
expanded_roget_pairs = {
    "OPEN_CLOSED":      [("open", "closed"), ("unfastened", "fastened"),
                          ("revealed", "concealed"), ("uncovered", "covered"),
                          ("exposed", "hidden")],
    "LIVING_DEAD":      [("alive", "dead"), ("living", "deceased"),
                          ("vital", "lifeless"), ("animate", "inanimate"),
                          ("breathing", "motionless")],
    "BEAUTIFUL_UGLY":   [("beautiful", "ugly"), ("pretty", "hideous"),
                          ("lovely", "repulsive"), ("gorgeous", "grotesque"),
                          ("attractive", "repulsive")],
    "PURE_IMPURE":      [("pure", "impure"), ("clean", "tainted"),
                          ("immaculate", "defiled"), ("sterile", "contaminated"),
                          ("untainted", "polluted")],
    "NATURAL_ARTIFICIAL": [("natural", "artificial"), ("organic", "synthetic"),
                            ("wild", "manufactured"), ("raw", "processed"),
                            ("genuine", "fake")],
    "ROUND_ANGULAR":    [("round", "angular"), ("curved", "straight"),
                          ("smooth", "jagged"), ("circular", "square"),
                          ("rounded", "sharp")],
    "WHOLE_BROKEN":     [("whole", "broken"), ("intact", "shattered"),
                          ("complete", "fragmented"), ("unified", "splintered"),
                          ("undamaged", "damaged")],
    "BOUND_FREE":       [("free", "bound"), ("liberated", "captive"),
                          ("released", "restrained"), ("unrestricted", "constrained"),
                          ("autonomous", "subjugated")],
    "VISIBLE_INVISIBLE": [("visible", "invisible"), ("apparent", "hidden"),
                           ("manifest", "latent"), ("seen", "unseen"),
                           ("evident", "obscured")],
    "ACTIVE_PASSIVE":   [("active", "passive"), ("energetic", "lethargic"),
                          ("dynamic", "static"), ("engaged", "withdrawn"),
                          ("vigorous", "inert")],
    "STABLE_UNSTABLE":  [("stable", "unstable"), ("steady", "wobbly"),
                          ("balanced", "precarious"), ("secure", "shaky"),
                          ("solid", "tottering")],
    "RICH_POOR":        [("rich", "poor"), ("wealthy", "impoverished"),
                          ("affluent", "destitute"), ("prosperous", "indigent"),
                          ("opulent", "penniless")],
}
for name, pairs in expanded_roget_pairs.items():
    candidates[name], _ = build_axis(wv, pairs)

# MASCULINE-FEMININE (gender hypothesis test)
MF_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "hers"),
    ("man", "woman"), ("boy", "girl"), ("male", "female"),
    ("father", "mother"), ("brother", "sister"), ("uncle", "aunt"),
    ("husband", "wife"), ("king", "queen"), ("prince", "princess"),
]
MF, mf_used = build_axis(wv, MF_PAIRS)
candidates["MF"] = MF
print(f"  MF built from {len(mf_used)}/{len(MF_PAIRS)} pairs")

print(f"\nTotal candidates: {len(candidates)}")


# ============================================================================
# Test 1 — cos(MF, HARDNESS): the gender hypothesis
# ============================================================================
print("\n" + "=" * 78)
print("TEST 1 — cos(MF, HARDNESS) and other relevant cosines")
print("=" * 78)

print(f"\ncos(MF, HARDNESS)    = {cos(MF, candidates['HARDNESS']):+.4f}")
print(f"cos(MF, STRENGTH)    = {cos(MF, candidates['STRENGTH']):+.4f}")
print(f"cos(MF, SIZE)        = {cos(MF, candidates['SIZE']):+.4f}")
print(f"cos(MF, SHARPNESS)   = {cos(MF, candidates['SHARPNESS']):+.4f}")
print(f"cos(MF, W)           = {cos(MF, candidates['W']):+.4f}")
print(f"cos(MF, INT)         = {cos(MF, candidates['INT']):+.4f}")
print(f"cos(MF, ATT)         = {cos(MF, candidates['ATT']):+.4f}")
print(f"cos(MF, ABS)         = {cos(MF, candidates['ABS']):+.4f}")
print(f"cos(MF, REAL_IMAG)   = {cos(MF, candidates['REAL_IMAG']):+.4f}")
print(f"cos(MF, BEAUTIFUL_UGLY) = {cos(MF, candidates['BEAUTIFUL_UGLY']):+.4f}")
print(f"cos(MF, BOUND_FREE)  = {cos(MF, candidates['BOUND_FREE']):+.4f}")
print(f"cos(MF, ACTIVE_PASSIVE) = {cos(MF, candidates['ACTIVE_PASSIVE']):+.4f}")

print("\nMF positive pole (male side, top 12):")
for w, s in wv.similar_by_vector(MF.astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")
print("\nMF negative pole (female side, top 12):")
for w, s in wv.similar_by_vector((-MF).astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")


# ============================================================================
# Test 2 — Greedy on cognitive test categories with expanded pool
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

test_vectors = []
for cat, words in test_categories.items():
    for w in words:
        v = get_deanisotropized(w)
        if v is not None:
            test_vectors.append(v)
test_vectors = np.stack(test_vectors)
print(f"\nCognitive test sample: {len(test_vectors)} words")


def basis_coverage_on_sample(gs_basis, sample):
    if not gs_basis:
        return 0.0
    B = np.stack(gs_basis)
    projections = sample @ B.T
    captured = (projections ** 2).sum(axis=1)
    explained = np.sqrt(np.maximum(0, captured))
    return float(explained.mean())


# Annotate
cog_set = {"C", "W", "ATT", "INT", "R", "D", "IO", "DV", "MB", "EV", "ABS", "REAL_IMAG"}
lakoff_set = {"UD", "FB", "LD", "PATH", "EXIST", "FORCE", "BAL", "DIFF", "COH"}
cluster_set = {"VALENCE", "AROUSAL", "SUC", "LOSS"}
orig_roget_set = set(original_roget_pairs.keys())
new_roget_set = set(expanded_roget_pairs.keys())
mf_set = {"MF"}

origin = {}
for n in candidates:
    if n in cog_set: origin[n] = "cog"
    elif n in lakoff_set: origin[n] = "lak"
    elif n in cluster_set: origin[n] = "clu"
    elif n in orig_roget_set: origin[n] = "r1"
    elif n in new_roget_set: origin[n] = "r2"
    elif n in mf_set: origin[n] = "MF"


print("\n" + "=" * 96)
print(f"Greedy selection — {len(candidates)} candidates, cognitive-test-words objective")
print("=" * 96)

selected_names = []
selected_orthonormal = []
remaining = dict(candidates)
prev_coverage = 0.0

print(f"\n{'step':<5} {'selected':<18} {'cov added':>10} {'cum cov':>9}  next-4 (name[orig], cov)")
print("-" * 115)

for step in range(1, 25):
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

    top4 = ", ".join(
        f"{n}[{origin[n][:2]}]({c * 100:.1f})"
        for n, _, _, c in candidate_results[1:5]
    )

    print(f"  {step:<3}  {best_name:<16} [{origin[best_name]:<2}]  "
          f"+{added * 100:>6.2f}pp  {best_cov * 100:>6.2f}%   {top4}")

    selected_names.append(best_name)
    selected_orthonormal.append(best_res)
    del remaining[best_name]
    prev_coverage = best_cov

    if added < 0.005 and step >= 10:
        print(f"  (added coverage < 0.5pp at step {step}; stopping)")
        break

from collections import Counter
origins_picked = Counter(origin[n] for n in selected_names)
print(f"\nFinal greedy basis ({len(selected_names)} axes):")
for i, name in enumerate(selected_names, 1):
    print(f"  {i:2}. {name:<18} ({origin[name]})")
print(f"\nComposition: {dict(origins_picked)}")
print(f"Final coverage: {prev_coverage * 100:.2f}%")

# Cog-13b reference
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
print(f"\nReference: Cog-13b coverage on same sample = "
      f"{basis_coverage_on_sample(gs_13b, test_vectors) * 100:.2f}%")
print(f"Reference: exp90 greedy (38 candidates) reached 48.43%")

np.savez("/Users/macn/Documents/embeddingexp/exp92_results.npz",
         MF=MF,
         cos_MF_HARDNESS=cos(MF, candidates['HARDNESS']),
         selected_names=np.array(selected_names))
print("\nSaved exp92_results.npz")
