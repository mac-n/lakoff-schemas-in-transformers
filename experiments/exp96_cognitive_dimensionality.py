"""
exp96 — Cognitive subspace dimensionality + expanded Roget pool.

Two questions:

PART A — empirical dimensionality of cognitive content.
  Take a curated cognitive vocabulary (200+ words across cognitive domains).
  PCA on those word vectors. How many cognitive-PCs to capture 80% / 90% /
  95% of cognitive variance? Compare to random-vocab equal-size sample.

PART B — expanded Roget pool (40+ antonym categories).
  Build a much bigger pool of antonym axes from Roget. Greedy on cognitive
  test categories with the full pool. See what gets selected.
"""
import numpy as np
import gensim.downloader as api


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(wv, pairs):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    if not offs:
        return None
    return unit(np.stack(offs).mean(axis=0))


def build_single(wv, pos, neg):
    if pos not in wv.key_to_index or neg not in wv.key_to_index:
        return None
    return unit(wv[pos] - wv[neg])


def gs_orthogonalize(axes):
    out = []
    for v in axes:
        if v is None:
            continue
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


# ============================================================================
# PART A — Dimensionality of cognitive vocabulary
# ============================================================================
print("\n" + "=" * 78)
print("PART A — Effective dimensionality of cognitive vocabulary")
print("=" * 78)

# Curated cognitive vocabulary — 200+ cognitively-loaded words across many domains
cognitive_vocab = [
    # Emotions
    "happiness", "sadness", "anger", "fear", "joy", "grief", "envy", "jealousy",
    "pride", "humility", "shame", "guilt", "contentment", "longing", "yearning",
    "delight", "melancholy", "rage", "elation", "despair", "serenity", "anguish",
    "love", "hate", "compassion", "cruelty", "tenderness", "harshness",
    "disgust", "awe", "wonder", "boredom", "ecstasy", "agony", "bliss",
    "loneliness", "togetherness", "intimacy", "alienation", "isolation",
    # Agentive states
    "ambition", "determination", "resignation", "willpower", "discipline",
    "procrastination", "perseverance", "complacency", "vigilance", "diligence",
    "industriousness", "negligence", "carelessness", "resolution", "indecision",
    "commitment", "hesitation", "courage", "cowardice", "boldness", "timidity",
    # Social/relational
    "trust", "betrayal", "friendship", "enmity", "loyalty", "rivalry",
    "respect", "contempt", "admiration", "scorn", "gratitude", "resentment",
    "cooperation", "competition", "kindness", "cruelty", "generosity", "greed",
    "empathy", "indifference", "honesty", "deception",
    # Epistemic
    "knowledge", "ignorance", "belief", "doubt", "uncertainty", "conviction",
    "skepticism", "confidence", "hesitation", "speculation", "intuition",
    "memory", "forgetting", "understanding", "confusion", "insight", "delusion",
    "wisdom", "foolishness", "discernment", "naivete",
    # Cognitive states
    "attention", "distraction", "focus", "absent-mindedness", "concentration",
    "curiosity", "interest", "boredom", "fascination", "disinterest",
    "creativity", "imagination", "logic", "reasoning", "judgment", "intuition",
    # Abstract/conceptual
    "theory", "principle", "concept", "framework", "paradigm", "schema",
    "abstraction", "generalization", "specification", "category", "definition",
    "axiom", "premise", "conclusion", "inference", "deduction", "induction",
    # Modal
    "possibility", "actuality", "necessity", "contingency", "probability",
    "certainty", "hypothesis", "fact", "truth", "falsehood", "myth", "reality",
    # Volitional
    "freedom", "constraint", "choice", "obligation", "permission", "prohibition",
    "intention", "accident", "purpose", "happenstance",
    # Phenomenological
    "experience", "perception", "sensation", "thought", "feeling", "awareness",
    "consciousness", "unconsciousness", "presence", "absence",
    # Self-related
    "self", "selfhood", "identity", "personhood", "individuality", "autonomy",
    "ego", "soul", "spirit", "mind", "body", "subjectivity", "objectivity",
    # Value
    "virtue", "vice", "morality", "ethics", "goodness", "evil", "righteousness",
    "depravity", "honor", "dishonor", "dignity", "indignity",
    # Time-related cognitive
    "past", "future", "memory", "anticipation", "regret", "hope", "nostalgia",
    "expectation", "remembrance", "foresight",
]

cog_vecs = []
for w in cognitive_vocab:
    v = get_deanisotropized(w)
    if v is not None:
        cog_vecs.append(v)
cog_vecs = np.stack(cog_vecs)
print(f"\nCognitive vocab in GloVe: {len(cog_vecs)} of {len(cognitive_vocab)} attempted")

# PCA on cognitive vocab
print("\nComputing PCA on cognitive vocabulary...")
_, S_cog, Vt_cog = np.linalg.svd(cog_vecs, full_matrices=False)
var_cog = (S_cog ** 2) / (S_cog ** 2).sum()
cum_var_cog = np.cumsum(var_cog)

# Compare to equal-size random vocab sample
print("Computing PCA on equal-size random vocab sample...")
np.random.seed(42)
random_idx = np.random.choice(len(wv.vectors), len(cog_vecs), replace=False)
rand_vecs = (wv.vectors[random_idx] - mu)
rand_norms = np.linalg.norm(rand_vecs, axis=1, keepdims=True)
rand_norms[rand_norms < 1e-10] = 1
rand_vecs = rand_vecs / rand_norms

_, S_rand, _ = np.linalg.svd(rand_vecs, full_matrices=False)
var_rand = (S_rand ** 2) / (S_rand ** 2).sum()
cum_var_rand = np.cumsum(var_rand)

print(f"\n{'K':<5}  {'cog cumvar':>12}  {'rand cumvar':>12}  ratio")
print("-" * 50)
for K in [1, 5, 10, 15, 20, 24, 30, 40, 50, 75, 100, 150, 200]:
    if K > len(S_cog):
        break
    cv_cog = cum_var_cog[K - 1]
    cv_rand = cum_var_rand[K - 1]
    print(f"  {K:<3}  {cv_cog * 100:>10.2f}%  {cv_rand * 100:>10.2f}%  {cv_cog/cv_rand:>5.2f}")

# Effective dimensionality at different variance thresholds
for thresh in [0.8, 0.9, 0.95, 0.99]:
    n_cog = int(np.searchsorted(cum_var_cog, thresh) + 1)
    n_rand = int(np.searchsorted(cum_var_rand, thresh) + 1)
    print(f"\n  To capture {thresh*100:.0f}% variance:")
    print(f"    cognitive vocab: {n_cog} PCs")
    print(f"    random vocab:    {n_rand} PCs")


# ============================================================================
# PART B — Larger Roget pool + greedy
# ============================================================================
print("\n" + "=" * 78)
print("PART B — Expanded Roget antonym pool")
print("=" * 78)

# Single-pair version of dominant axes (per exp95 finding that single-pair works
# for some axes, multi-pair for others). To make this less hand-curated, use
# SINGLE PAIRS for all 50 Roget categories.

roget_single_pairs = {
    "TEMPERATURE":    ("hot", "cold"),
    "MOISTURE":       ("wet", "dry"),
    "SPEED":          ("fast", "slow"),
    "SIZE":           ("large", "small"),
    "STRENGTH":       ("strong", "weak"),
    "TEXTURE":        ("smooth", "rough"),
    "TASTE":          ("sweet", "sour"),
    "AGE":            ("young", "old"),
    "SOUND_VOLUME":   ("loud", "quiet"),
    "CLEANLINESS":    ("clean", "dirty"),
    "HARDNESS":       ("hard", "soft"),
    "SHARPNESS":      ("sharp", "dull"),
    "DENSITY":        ("dense", "sparse"),
    "OPEN_CLOSED":    ("open", "closed"),
    "LIVING_DEAD":    ("alive", "dead"),
    "BEAUTIFUL_UGLY": ("beautiful", "ugly"),
    "PURE_IMPURE":    ("pure", "impure"),
    "NATURAL_ARTIFICIAL": ("natural", "artificial"),
    "ROUND_ANGULAR":  ("round", "angular"),
    "WHOLE_BROKEN":   ("whole", "broken"),
    "BOUND_FREE":     ("free", "bound"),
    "VISIBLE_INVISIBLE": ("visible", "invisible"),
    "ACTIVE_PASSIVE": ("active", "passive"),
    "STABLE_UNSTABLE": ("stable", "unstable"),
    "RICH_POOR":      ("rich", "poor"),
    # NEW ADDITIONS — expanding Roget coverage
    "DEEP_SHALLOW":   ("deep", "shallow"),
    "WIDE_NARROW":    ("wide", "narrow"),
    "TIGHT_LOOSE":    ("tight", "loose"),
    "EARLY_LATE":     ("early", "late"),
    "CHEAP_EXPENSIVE": ("cheap", "expensive"),
    "COMMON_RARE":    ("common", "rare"),
    "EASY_HARD":      ("easy", "hard"),
    "SAFE_DANGEROUS": ("safe", "dangerous"),
    "BRIGHT_DARK":    ("bright", "dark"),
    "FULL_EMPTY":     ("full", "empty"),
    "MANY_FEW":       ("many", "few"),
    "SIMPLE_COMPLEX": ("simple", "complex"),
    "QUIET_NOISY":    ("quiet", "noisy"),  # alternative SOUND_VOLUME
    "STRAIGHT_CURVED": ("straight", "curved"),
    "FRESH_STALE":    ("fresh", "stale"),
    "PUBLIC_PRIVATE": ("public", "private"),
    "FAMILIAR_STRANGE": ("familiar", "strange"),
    "PROUD_HUMBLE":   ("proud", "humble"),
    "BRAVE_AFRAID":   ("brave", "afraid"),
    "KIND_CRUEL":     ("kind", "cruel"),
    "WISE_FOOLISH":   ("wise", "foolish"),
    "GENEROUS_STINGY": ("generous", "stingy"),
    "PATIENT_IMPATIENT": ("patient", "impatient"),
    "HONEST_DISHONEST": ("honest", "dishonest"),
    "POLITE_RUDE":    ("polite", "rude"),
}

print(f"\nBuilding {len(roget_single_pairs)} single-pair Roget axes...")
roget_axes = {}
missing = []
for name, (pos, neg) in roget_single_pairs.items():
    axis = build_single(wv, pos, neg)
    if axis is None:
        missing.append((name, pos, neg))
    else:
        roget_axes[name] = axis
if missing:
    print(f"Missing (OOV): {missing}")
print(f"  Built {len(roget_axes)} axes")


# ============================================================================
# Greedy on cognitive test (using cognitive_vocab as the test sample)
# ============================================================================
print("\n" + "=" * 78)
print("Greedy selection on cognitive vocabulary (200+ words)")
print("=" * 78)

# Use all the cognitive vocab as the test sample
test_vecs = cog_vecs

selected_names = []
selected_orthonormal = []
remaining = dict(roget_axes)
prev_coverage = 0.0


def basis_coverage(gs_basis, vecs):
    if not gs_basis:
        return 0.0
    B = np.stack(gs_basis)
    projections = vecs @ B.T
    captured = (projections ** 2).sum(axis=1)
    explained = np.sqrt(np.maximum(0, captured))
    return float(explained.mean())


print(f"\n{'step':<5} {'selected':<22} {'cov added':>10} {'cum cov':>9}  next-3")
print("-" * 90)

for step in range(1, 30):
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
        cov = basis_coverage(trial, test_vecs)
        candidate_results.append((name, axis, res, cov))

    if not candidate_results:
        break
    candidate_results.sort(key=lambda x: -x[3])
    best_name, best_axis, best_res, best_cov = candidate_results[0]
    added = best_cov - prev_coverage

    top3 = ", ".join(f"{n}({c * 100:.1f})" for n, _, _, c in candidate_results[1:4])
    print(f"  {step:<3}  {best_name:<22}  +{added * 100:>6.2f}pp {best_cov * 100:>7.2f}%   {top3}")

    selected_names.append(best_name)
    selected_orthonormal.append(best_res)
    del remaining[best_name]
    prev_coverage = best_cov

    if added < 0.003 and step >= 15:
        print(f"  (added cov < 0.3pp at step {step}; stopping)")
        break

print(f"\nFinal Roget-only greedy basis ({len(selected_names)} axes):")
for i, name in enumerate(selected_names, 1):
    print(f"  {i:2}. {name}")
print(f"Final coverage on cognitive vocab: {prev_coverage * 100:.2f}%")

# Also report effective dimensionality of cognitive vocab as captured by this basis
print(f"\nFor reference, % of cognitive variance captured by:")
print(f"  Top 24 PCs of cognitive vocab itself: {cum_var_cog[23] * 100:.2f}%")
print(f"  Top 50 PCs of cognitive vocab itself: {cum_var_cog[min(49, len(S_cog)-1)] * 100:.2f}%")

np.savez("/Users/macn/Documents/embeddingexp/exp96_results.npz",
         var_cog=var_cog,
         var_rand=var_rand,
         selected_names=np.array(selected_names))
print("\nSaved exp96_results.npz")
