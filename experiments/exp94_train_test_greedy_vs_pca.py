"""
exp94 — Proper train-test evaluation: greedy vs PCA on cognitive content.

Issue with exp93: greedy was selected on the same 86 cognitive words it was
evaluated on. Possibly overfit.

Clean test:
  - split 86 cognitive words into 40 train / 46 test (random seed)
  - greedy: select 24 axes by optimizing coverage on TRAIN words
  - PCA: top 24 PCs from 50K random GloVe vocab (standard PCA setup)
  - evaluate BOTH on the TEST words
  - see whether greedy still beats PCA on held-out cognitive content

We could also run cognitive-PCA (PCA on the train words specifically) for a
cleaner-but-trickier comparison.
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


# Build candidate pool (51 candidates from exp92)
print("Building candidate pool...")
candidates = {}

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
candidates["UD"]    = build_axis(wv, UP_DOWN_MML)
candidates["FB"]    = build_axis(wv, FORWARD_BACK_MML)
candidates["LD"]    = build_axis(wv, LIGHT_DARK_MML)
candidates["PATH"]  = build_axis(wv, PATH_MOTION_MML)
candidates["EXIST"] = build_axis(wv, EXISTENCE_MML)
candidates["FORCE"] = build_axis(wv, FORCE_MML)
candidates["BAL"]   = build_axis(wv, BALANCE_MML)
candidates["DIFF"]  = build_axis(wv, DIFFICULTY_BURDEN_MML)
candidates["COH"]   = build_axis(wv, COHERENCE_PAIRS)
candidates["VALENCE"] = build_axis(wv, VALENCE_PAIRS)
candidates["AROUSAL"] = build_axis(wv, AROUSAL_PAIRS)
candidates["SUC"] = build_axis(wv, SUCCESS_FAILURE_PAIRS)
candidates["LOSS"] = build_axis(wv, LOSS_PAIRS)

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
    candidates[name] = build_axis(wv, pairs)

expanded_roget_pairs = {
    "OPEN_CLOSED":      [("open", "closed"), ("unfastened", "fastened"),
                          ("revealed", "concealed"), ("uncovered", "covered"),
                          ("exposed", "hidden")],
    "LIVING_DEAD":      [("alive", "dead"), ("living", "deceased"),
                          ("vital", "lifeless"), ("animate", "inanimate")],
    "BEAUTIFUL_UGLY":   [("beautiful", "ugly"), ("pretty", "hideous"),
                          ("lovely", "repulsive"), ("gorgeous", "grotesque")],
    "PURE_IMPURE":      [("pure", "impure"), ("clean", "tainted"),
                          ("immaculate", "defiled"), ("sterile", "contaminated")],
    "NATURAL_ARTIFICIAL": [("natural", "artificial"), ("organic", "synthetic"),
                            ("wild", "manufactured"), ("raw", "processed"),
                            ("genuine", "fake")],
    "ROUND_ANGULAR":    [("round", "angular"), ("curved", "straight"),
                          ("circular", "square"), ("rounded", "sharp")],
    "WHOLE_BROKEN":     [("whole", "broken"), ("intact", "shattered"),
                          ("complete", "fragmented"), ("unified", "splintered")],
    "BOUND_FREE":       [("free", "bound"), ("liberated", "captive"),
                          ("released", "restrained"), ("unrestricted", "constrained")],
    "VISIBLE_INVISIBLE": [("visible", "invisible"), ("apparent", "hidden"),
                           ("manifest", "latent"), ("seen", "unseen")],
    "ACTIVE_PASSIVE":   [("active", "passive"), ("energetic", "lethargic"),
                          ("dynamic", "static"), ("engaged", "withdrawn")],
    "STABLE_UNSTABLE":  [("stable", "unstable"), ("steady", "wobbly"),
                          ("balanced", "precarious"), ("secure", "shaky")],
    "RICH_POOR":        [("rich", "poor"), ("wealthy", "impoverished"),
                          ("affluent", "destitute"), ("prosperous", "indigent")],
}
for name, pairs in expanded_roget_pairs.items():
    candidates[name] = build_axis(wv, pairs)

MF_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "hers"),
    ("man", "woman"), ("boy", "girl"), ("male", "female"),
    ("father", "mother"), ("brother", "sister"), ("uncle", "aunt"),
    ("husband", "wife"), ("king", "queen"), ("prince", "princess"),
]
candidates["MF"] = build_axis(wv, MF_PAIRS)

print(f"Total candidates: {len(candidates)}")


# All cognitive test words from exp92
all_cog_words = [
    "happiness", "sadness", "anger", "envy", "jealousy", "pride", "humility",
    "contentment", "longing", "delight", "melancholy", "rage", "elation",
    "despair", "serenity", "anguish", "disgust",
    "ambition", "determination", "resignation", "willpower", "discipline",
    "procrastination", "perseverance", "complacency", "vigilance",
    "diligence", "industriousness", "negligence", "carelessness",
    "trust", "betrayal", "friendship", "enmity", "loyalty", "rivalry",
    "respect", "contempt", "admiration", "scorn", "gratitude", "resentment",
    "knowledge", "ignorance", "belief", "doubt", "uncertainty",
    "conviction", "skepticism", "confidence", "hesitation",
    "speculation", "intuition", "memory", "forgetting",
    "chair", "table", "dog", "stone", "tree", "river", "mountain",
    "hammer", "rope", "lamp", "cup", "window",
    "theorem", "philosophy", "ontology", "epistemology", "axiom",
    "principle", "framework", "paradigm", "schema", "abstraction",
    "hypothetical", "imaginary", "fictional", "speculative", "conjectural",
    "perhaps", "supposedly", "allegedly", "putative",
]

# Filter to vocab
valid_words = [w for w in all_cog_words if get_deanisotropized(w) is not None]
print(f"\nValid cognitive words: {len(valid_words)}")


# ============================================================================
# Repeat the comparison over multiple random splits
# ============================================================================
def basis_coverage_on_vecs(gs_basis, vecs):
    if not gs_basis:
        return 0.0
    B = np.stack(gs_basis)
    projections = vecs @ B.T
    captured = (projections ** 2).sum(axis=1)
    explained = np.sqrt(np.maximum(0, captured))
    return float(explained.mean())


# Pre-compute PCA on 50K random vocab (used regardless of split)
print("\nComputing PCA on 50K random GloVe vocab (standard setup)...")
np.random.seed(42)
sample_idx = np.random.choice(len(wv.vectors), 50000, replace=False)
sample_vecs = wv.vectors[sample_idx] - mu
_, S_pca, Vt_pca = np.linalg.svd(sample_vecs, full_matrices=False)
pca_24 = [Vt_pca[i] for i in range(24)]
gs_pca_24 = gs_orthogonalize(pca_24)


def run_greedy(train_vecs, k=24):
    """Greedy selection optimizing coverage on train_vecs."""
    selected = []
    remaining = dict(candidates)
    while len(selected) < k and remaining:
        candidate_results = []
        for name, axis in remaining.items():
            res = axis.copy()
            for u in selected:
                res = res - (res @ u) * u
            n = np.linalg.norm(res)
            if n < 1e-6:
                continue
            res = res / n
            trial = selected + [res]
            cov = basis_coverage_on_vecs(trial, train_vecs)
            candidate_results.append((name, res, cov))
        if not candidate_results:
            break
        candidate_results.sort(key=lambda x: -x[2])
        best_name, best_res, _ = candidate_results[0]
        selected.append(best_res)
        del remaining[best_name]
    return selected


print("\n" + "=" * 78)
print("Running 5 random train/test splits (40 train / N test) for stability")
print("=" * 78)

results_greedy_test = []
results_pca_test = []
results_greedy_train = []

for seed in range(5):
    np.random.seed(seed)
    indices = np.random.permutation(len(valid_words))
    train_idx = indices[:40]
    test_idx = indices[40:]
    train_words = [valid_words[i] for i in train_idx]
    test_words = [valid_words[i] for i in test_idx]

    train_vecs = np.stack([get_deanisotropized(w) for w in train_words])
    test_vecs = np.stack([get_deanisotropized(w) for w in test_words])

    # Greedy on train
    gs_greedy = run_greedy(train_vecs, k=24)

    # Evaluate both on train and test
    g_train = basis_coverage_on_vecs(gs_greedy, train_vecs)
    g_test = basis_coverage_on_vecs(gs_greedy, test_vecs)
    p_test = basis_coverage_on_vecs(gs_pca_24, test_vecs)

    results_greedy_train.append(g_train)
    results_greedy_test.append(g_test)
    results_pca_test.append(p_test)

    print(f"  Seed {seed}:  train_words={len(train_words)} test_words={len(test_words)}  "
          f"greedy-train={g_train * 100:.2f}%  greedy-test={g_test * 100:.2f}%  "
          f"PCA-test={p_test * 100:.2f}%  Δ={(g_test - p_test) * 100:+.2f}pp")


print(f"\n{'metric':<28} {'mean':>10}  {'std':>8}")
print("-" * 50)
print(f"  greedy on TRAIN              {np.mean(results_greedy_train) * 100:>8.2f}%  "
      f"{np.std(results_greedy_train) * 100:>6.2f}")
print(f"  greedy on TEST               {np.mean(results_greedy_test) * 100:>8.2f}%  "
      f"{np.std(results_greedy_test) * 100:>6.2f}")
print(f"  PCA-24 on TEST               {np.mean(results_pca_test) * 100:>8.2f}%  "
      f"{np.std(results_pca_test) * 100:>6.2f}")
print(f"\n  Greedy advantage on test:    "
      f"{(np.mean(results_greedy_test) - np.mean(results_pca_test)) * 100:+.2f}pp")
print(f"  Greedy train-test gap:       "
      f"{(np.mean(results_greedy_train) - np.mean(results_greedy_test)) * 100:+.2f}pp  "
      f"(if large, greedy was overfitting to train)")
