"""
exp95 — Single-pair simplification test + HARDNESS sense decomposition.

PART A — Single-pair sufficiency.
  Compare multi-pair axes (HARDNESS, NATURAL_ARTIFICIAL, ABS, UD, REAL_IMAG,
  C, INT, etc.) to their single-pair simplifications (hard/soft alone,
  natural/artificial alone, etc.). Are we over-engineering?

PART B — HARDNESS sense decomposition.
  HARDNESS as constructed is a centroid of multiple senses:
    physical (hard rock), difficulty (hard problem), severity (hard punishment),
    reality (hard facts), personality (hard person), effort (work hard).
  Build sense-specific axes and compute cos with HARDNESS to see which
  senses dominate.
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
    IN_OUT_MML_CLEAN, UP_DOWN_MML,
)
from exp52_target_axis_validation import VALENCE_PAIRS, AROUSAL_PAIRS


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


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


def build_single(wv, pos, neg):
    if pos not in wv.key_to_index or neg not in wv.key_to_index:
        return None
    return unit(wv[pos] - wv[neg])


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


# ============================================================================
# PART A — single-pair vs multi-pair
# ============================================================================
print("\n" + "=" * 78)
print("PART A — single-pair vs multi-pair axis comparison")
print("=" * 78)

# Multi-pair axes
multi_axes = {}
multi_axes["C"]   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
multi_axes["W"]   = build_axis(wv, TARGET_WEIGHT_PAIRS)
multi_axes["ATT"] = build_axis(wv, ATTENTION_CLEAN_PAIRS)
multi_axes["INT"] = build_axis(wv, INTENTION_CLEAN_PAIRS)
multi_axes["R"]   = build_R(wv)
multi_axes["D"]   = build_axis(wv, TARGET_SURPRISAL_PAIRS)
multi_axes["IO"]  = build_axis(wv, IN_OUT_MML_CLEAN)
multi_axes["UD"]  = build_axis(wv, UP_DOWN_MML)
multi_axes["DV"]  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
multi_axes["MB"]  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
multi_axes["EV"]  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)
multi_axes["ABS"] = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
multi_axes["REAL_IMAG"] = build_axis(wv, REAL_IMAGINARY_PAIRS)
multi_axes["HARDNESS"] = build_axis(wv, [("hard", "soft"), ("firm", "mushy"),
                                          ("rigid", "pliable"), ("solid", "flimsy")])
multi_axes["NATURAL_ARTIFICIAL"] = build_axis(wv, [
    ("natural", "artificial"), ("organic", "synthetic"),
    ("wild", "manufactured"), ("raw", "processed"), ("genuine", "fake")])
multi_axes["VALENCE"] = build_axis(wv, VALENCE_PAIRS)

# Single-pair candidates: dominant lexical pair for each axis
single_pairs = {
    "C":         ("flourishing", "suffering"),
    "W":         ("heavy", "weightless"),
    "ATT":       ("noticing", "missing"),
    "INT":       ("intending", "drifting"),
    "R":         ("stabilizing", "escalating"),
    "D":         ("familiar", "unfamiliar"),
    "IO":        ("inside", "outside"),
    "UD":        ("up", "down"),
    "DV":        ("chosen", "rejected"),
    "MB":        ("self", "other"),
    "EV":        ("curious", "indifferent"),
    "ABS":       ("abstract", "concrete"),
    "REAL_IMAG": ("imaginary", "real"),
    "HARDNESS":  ("hard", "soft"),
    "NATURAL_ARTIFICIAL": ("natural", "artificial"),
    "VALENCE":   ("good", "bad"),
}

# Test sample
all_cog_words = [
    "happiness", "sadness", "anger", "envy", "jealousy", "pride", "humility",
    "contentment", "longing", "delight", "melancholy", "rage", "elation",
    "ambition", "determination", "willpower", "discipline", "perseverance",
    "trust", "betrayal", "friendship", "respect", "contempt", "admiration",
    "knowledge", "ignorance", "belief", "doubt", "uncertainty",
    "chair", "table", "dog", "stone", "tree", "river",
    "theorem", "philosophy", "principle", "framework",
    "hypothetical", "imaginary", "fictional", "speculative",
]
test_vecs = np.stack([get_deanisotropized(w) for w in all_cog_words
                      if get_deanisotropized(w) is not None])
print(f"Test sample: {len(test_vecs)} cognitive words")


print(f"\n{'axis':<22} {'cos(single, multi)':>20} {'single cov':>11} {'multi cov':>11} {'ratio':>7}")
print("-" * 78)
results = []
for name, (pos, neg) in single_pairs.items():
    if name not in multi_axes:
        continue
    multi = multi_axes[name]
    single = build_single(wv, pos, neg)
    if single is None:
        print(f"  {name:<20} ({pos}, {neg}) — OOV")
        continue
    c = cos(single, multi)
    single_cov = float(np.mean([(single @ v) ** 2 for v in test_vecs]) ** 0.5)
    multi_cov = float(np.mean([(multi @ v) ** 2 for v in test_vecs]) ** 0.5)
    ratio = single_cov / multi_cov if multi_cov > 0 else 0
    results.append((name, c, single_cov, multi_cov, ratio))
    print(f"  {name:<20} {c:>+18.4f}    {single_cov * 100:>7.2f}%   "
          f"{multi_cov * 100:>7.2f}%   {ratio:>5.2f}")


# ============================================================================
# PART B — HARDNESS sense decomposition
# ============================================================================
print("\n" + "=" * 78)
print("PART B — HARDNESS sense decomposition")
print("=" * 78)

HARDNESS = multi_axes["HARDNESS"]

# Build sense-specific axes
sense_axes = {}

# Physical hardness (concrete material objects)
sense_axes["PHYSICAL_HARD"] = build_axis(wv, [
    ("rock", "cushion"), ("stone", "foam"), ("steel", "cotton"),
    ("concrete", "wool"), ("metal", "fabric"), ("brick", "feather"),
])

# Difficulty
sense_axes["DIFFICULTY"] = build_axis(wv, [
    ("difficult", "easy"), ("challenging", "simple"), ("arduous", "effortless"),
    ("tough", "trivial"), ("formidable", "manageable"),
])

# Severity
sense_axes["SEVERITY"] = build_axis(wv, [
    ("harsh", "gentle"), ("strict", "lenient"), ("severe", "mild"),
    ("stern", "kind"), ("austere", "indulgent"),
])

# Reality / factuality
sense_axes["REALITY_STATUS"] = build_axis(wv, [
    ("factual", "fictional"), ("real", "imaginary"), ("actual", "hypothetical"),
    ("concrete", "abstract"), ("verified", "speculative"),
])

# Personality hardness (interpersonal style)
sense_axes["PERSONALITY_HARD"] = build_axis(wv, [
    ("tough", "kind"), ("callous", "warm"), ("stern", "tender"),
    ("stoic", "sensitive"), ("brusque", "gentle"),
])

# Effort intensity
sense_axes["EFFORT"] = build_axis(wv, [
    ("intense", "casual"), ("strenuous", "leisurely"), ("vigorous", "slack"),
    ("rigorous", "easygoing"),
])

print(f"\n{'sense axis':<22} {'cos with HARDNESS':>20}")
print("-" * 45)
for name, axis in sense_axes.items():
    c = cos(HARDNESS, axis)
    interp = ""
    if c > 0.7: interp = "  ← dominant sense"
    elif c > 0.5: interp = "  ← substantial"
    elif c > 0.3: interp = "  ← moderate"
    elif c > 0.15: interp = "  ← small"
    else: interp = "  ← near-orthogonal"
    print(f"  {name:<20} {c:>+18.4f}{interp}")

print("\nIf one sense has cos > 0.7, HARDNESS leans toward it.")
print("If all senses are 0.3-0.6, HARDNESS is genuinely a composite.")

# Also: does single-pair (hard, soft) lean toward different senses than multi-pair HARDNESS?
print("\nFor contrast — cos(single (hard, soft), each sense):")
single_hardness = build_single(wv, "hard", "soft")
print(f"\n{'sense axis':<22} {'cos with single (hard,soft)':>26}  delta")
print("-" * 60)
for name, axis in sense_axes.items():
    c_single = cos(single_hardness, axis)
    c_multi = cos(HARDNESS, axis)
    print(f"  {name:<20} {c_single:>+24.4f}  {c_single - c_multi:+.4f}")


# ============================================================================
# Save
# ============================================================================
np.savez("/Users/macn/Documents/embeddingexp/exp95_results.npz",
         results_pairs=np.array([(n, c, sc, mc, r) for n, c, sc, mc, r in results],
                                 dtype=object))
print("\nSaved exp95_results.npz")
