"""
exp75 — Substrate-invariance + coverage validation for the 10-axis basis.

Two combined load-bearing validations for the project's first paper:

  PART A — substrate invariance (Thread 4 in BASIS_TESTS_TODO.md)
    Rebuild all 10 axes in fastText (1M vocab, Wiki+news+UMBC) and compare
    direction-by-direction to the GloVe-300 versions.

    Predicted: cos(GloVe_axis, fastText_axis) > 0.7 for each axis if the
    basis is substrate-invariant. Anything below 0.5 means the axis is
    substrate-specific. The pattern across axes is also informative —
    which primitives are most/least substrate-robust.

  PART B — coverage characterization (Thread 0.7)
    For diverse held-out word categories, compute fraction of each word
    vector explained by the 10-axis basis. Random baseline = 10/300 = 3.3%.

    Predicted: cognitive/affective words >25%; concrete nouns <15%; the
    basis discriminates by category — high coverage in the parts of
    word2vec space where it should be doing work, lower in concrete-noun
    space.

Run AFTER fastText download completes. Loads both substrates.
"""
import numpy as np
import gensim.downloader as api
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")

from project_axis_vocabulary import (
    TARGET_REWARD_COMPOSITE_PAIRS,        # C
    TARGET_WEIGHT_PAIRS,                  # W
    ATTENTION_CLEAN_PAIRS,                # ATT
    INTENTION_CLEAN_PAIRS,                # INT
    TARGET_EQUILIBRIUM_RUNAWAY_PAIRS,     # R (raw; V+A-residualized below)
    TARGET_SURPRISAL_PAIRS,               # D
    TARGET_DECISION_VERDICT_PAIRS,        # DV
    TARGET_MARKOV_BLANKET_PAIRS,          # MB
    TARGET_EPISTEMIC_VALUE_PAIRS,         # EV
)
from lakoff_canonical_vocabulary import IN_OUT_MML_CLEAN
from exp52_target_axis_validation import VALENCE_PAIRS, AROUSAL_PAIRS


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(wv, pairs):
    """Build an axis from anchor pairs in given word-vector model."""
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    if not offs:
        return None
    raw = np.stack(offs).mean(axis=0)
    return unit(raw)


def build_R_residualized(wv):
    """R = EQ_RUN with VALENCE and AROUSAL projected out (per exp60)."""
    eq_raw = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    valence = build_axis(wv, VALENCE_PAIRS)
    arousal = build_axis(wv, AROUSAL_PAIRS)
    if eq_raw is None or valence is None or arousal is None:
        return None
    r = eq_raw - (eq_raw @ valence) * valence
    r = r - (r @ arousal) * arousal
    return unit(r)


def build_all_axes(wv, label):
    """Build all 10 basis axes in a given substrate. Reports OOV counts."""
    print(f"\n  Building 10-axis basis in {label}...")
    axes = {}
    axes["C"]   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
    axes["W"]   = build_axis(wv, TARGET_WEIGHT_PAIRS)
    axes["ATT"] = build_axis(wv, ATTENTION_CLEAN_PAIRS)
    axes["INT"] = build_axis(wv, INTENTION_CLEAN_PAIRS)
    axes["R"]   = build_R_residualized(wv)
    axes["D"]   = build_axis(wv, TARGET_SURPRISAL_PAIRS)
    axes["IO"]  = build_axis(wv, IN_OUT_MML_CLEAN)
    axes["DV"]  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
    axes["MB"]  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
    axes["EV"]  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)

    # OOV reporting per axis
    for name, pairs in [
        ("C",   TARGET_REWARD_COMPOSITE_PAIRS),
        ("W",   TARGET_WEIGHT_PAIRS),
        ("ATT", ATTENTION_CLEAN_PAIRS),
        ("INT", INTENTION_CLEAN_PAIRS),
        ("D",   TARGET_SURPRISAL_PAIRS),
        ("IO",  IN_OUT_MML_CLEAN),
        ("DV",  TARGET_DECISION_VERDICT_PAIRS),
        ("MB",  TARGET_MARKOV_BLANKET_PAIRS),
        ("EV",  TARGET_EPISTEMIC_VALUE_PAIRS),
    ]:
        in_v = sum(1 for a, c in pairs
                   if a in wv.key_to_index and c in wv.key_to_index)
        if in_v < len(pairs):
            missing = [(a, c) for a, c in pairs
                       if a not in wv.key_to_index or c not in wv.key_to_index]
            print(f"    {name}: {in_v}/{len(pairs)} pairs in {label} "
                  f"(missing: {missing})")
    return axes


# ============================================================================
# Load both substrates
# ============================================================================
print("Loading glove-wiki-gigaword-300...")
glove = api.load("glove-wiki-gigaword-300")
print(f"  GloVe vocab: {len(glove.key_to_index)}")

print("Loading fasttext-wiki-news-subwords-300...")
ftxt = api.load("fasttext-wiki-news-subwords-300")
print(f"  fastText vocab: {len(ftxt.key_to_index)}")

glove_axes = build_all_axes(glove, "GloVe")
ftxt_axes  = build_all_axes(ftxt, "fastText")

axis_names = ["C", "W", "ATT", "INT", "R", "D", "IO", "DV", "MB", "EV"]


# ============================================================================
# PART A — substrate invariance
# ============================================================================
print("\n" + "=" * 84)
print("PART A — substrate invariance: cos(GloVe_axis, fastText_axis)")
print("=" * 84)
print()
print(f"  {'axis':<6}  {'cos':>8}  interpretation")
print(f"  {'-' * 6}  {'-' * 8}  {'-' * 50}")
substrate_cosines = []
for name in axis_names:
    g = glove_axes[name]
    f = ftxt_axes[name]
    if g is None or f is None:
        print(f"  {name:<6}  {'N/A':>8}  (axis missing in one substrate)")
        substrate_cosines.append(None)
        continue
    c = cos(g, f)
    substrate_cosines.append(c)
    if c > 0.7:
        interp = "substrate-invariant"
    elif c > 0.5:
        interp = "moderate"
    elif c > 0.3:
        interp = "weak (caution)"
    else:
        interp = "substrate-specific (problem)"
    print(f"  {name:<6}  {c:>+8.4f}  {interp}")

valid_cos = [c for c in substrate_cosines if c is not None]
print(f"\n  Across 10 axes:  mean = {np.mean(valid_cos):.3f}"
      f"   min = {min(valid_cos):.3f}   max = {max(valid_cos):.3f}")
print(f"  Axes >0.7:  {sum(1 for c in valid_cos if c > 0.7)}/{len(valid_cos)}")
print(f"  Axes <0.5:  {sum(1 for c in valid_cos if c < 0.5)}/{len(valid_cos)}")


# ============================================================================
# PART A.2 — inter-axis cosine matrix in fastText (compare to exp71 GloVe)
# ============================================================================
print("\n" + "=" * 84)
print("PART A.2 — inter-axis cosine matrix in fastText (compare to GloVe via exp71)")
print("=" * 84)
print()
print(f"  {'':<6}" + "".join(f"{n:>8}" for n in axis_names))

M_ftxt = np.zeros((10, 10))
for i, ni in enumerate(axis_names):
    row = f"  {ni:<6}"
    for j, nj in enumerate(axis_names):
        if ftxt_axes[ni] is None or ftxt_axes[nj] is None:
            row += f"  {'N/A':>6}"
            continue
        if i == j:
            M_ftxt[i, j] = 1.0
            row += f"  {' 1.000':>6}"
        else:
            c = cos(ftxt_axes[ni], ftxt_axes[nj])
            M_ftxt[i, j] = c
            row += f"  {c:>+6.3f}"
    print(row)

abs_off_ftxt = np.abs(M_ftxt - np.eye(10))
flat_ftxt = abs_off_ftxt[np.triu_indices(10, k=1)]
print(f"\n  fastText off-diagonal magnitude: "
      f"max = {flat_ftxt.max():.3f}  mean = {flat_ftxt.mean():.3f}  "
      f"median = {np.median(flat_ftxt):.3f}")
print(f"  Pairs with |cos| > 0.35: {(flat_ftxt > 0.35).sum()}/{len(flat_ftxt)}")
print()
print("  GloVe equivalent (from exp71): max ≈ 0.34 (C-W), 0 pairs above 0.35.")
print("  If fastText pattern is similar, basis structure is substrate-invariant.")


# ============================================================================
# PART B — coverage characterization (Thread 0.7)
# ============================================================================
print("\n" + "=" * 84)
print("PART B — coverage: fraction of word vector explained by 10-axis basis")
print("=" * 84)

# Gram-Schmidt orthogonalize the basis (use GloVe for primary coverage analysis)
gs_order = ["C", "W", "MB", "D", "IO", "R", "DV", "ATT", "INT", "EV"]
gs_basis = []
for n in gs_order:
    u = glove_axes[n].copy()
    for prev in gs_basis:
        u = u - (u @ prev) * prev
    u = unit(u)
    gs_basis.append(u)


def coverage(wv, w):
    """Fraction of word vector w explained by the 10-axis basis."""
    if w not in wv.key_to_index:
        return None
    v = unit(wv[w])
    v_residual = v.copy()
    for u_gs in gs_basis:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


# Diverse held-out word categories. NONE of these are in the anchor lists.
held_out_categories = {
    "EMOTIONS_AFFECTIVE": [
        "happiness", "sadness", "anger", "disgust", "envy", "jealousy",
        "pride", "humility", "contentment", "longing", "yearning", "delight",
        "melancholy", "rage", "elation", "despair", "serenity", "anguish",
    ],
    "AGENTIVE_STATES": [
        "ambition", "determination", "resignation", "indecision",
        "willpower", "discipline", "procrastination", "perseverance",
        "complacency", "vigilance", "diligence", "industriousness",
        "negligence", "carelessness",
    ],
    "SOCIAL_RELATIONAL": [
        "trust", "betrayal", "friendship", "enmity", "loyalty", "rivalry",
        "respect", "contempt", "admiration", "scorn", "gratitude",
        "resentment", "cooperation", "competition",
    ],
    "EPISTEMIC_STATES": [
        "knowledge", "ignorance", "belief", "doubt", "uncertainty",
        "conviction", "skepticism", "confidence", "hesitation",
        "speculation", "intuition", "memory", "forgetting",
    ],
    "CONCRETE_NOUNS_ANIMATE": [
        "dog", "cat", "horse", "elephant", "sparrow", "frog",
        "tiger", "rabbit", "bee", "spider",
    ],
    "CONCRETE_NOUNS_INANIMATE": [
        "chair", "table", "lamp", "cup", "hammer", "rope",
        "stone", "tree", "river", "mountain", "window", "shovel",
    ],
    "ABSTRACT_FORMAL": [
        "logic", "theorem", "mathematics", "geometry", "algebra",
        "philosophy", "ontology", "epistemology", "definition", "axiom",
    ],
    "RANDOM_NOUNS_PROJECT_HISTORY": [
        # The ones from exp58/60 — for direct comparison to prior results
        "sausage", "pyjamas", "marigold", "stapler", "pebble",
        "accordion", "casserole", "lamppost",
    ],
}

print()
print(f"  {'category':<35} {'n':>4} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}")
print("  " + "-" * 75)

category_results = {}
for cat, words in held_out_categories.items():
    cs = [coverage(glove, w) for w in words]
    cs = [c for c in cs if c is not None]
    if not cs:
        print(f"  {cat:<35} {'0':>4}  (no words in vocab)")
        continue
    category_results[cat] = cs
    print(f"  {cat:<35} {len(cs):>4} "
          f"{np.mean(cs) * 100:>7.1f}% {np.median(cs) * 100:>7.1f}% "
          f"{min(cs) * 100:>7.1f}% {max(cs) * 100:>7.1f}%")

print()
print("  Reference: 10-axis random baseline = 10/300 = 3.3% if axes were random.")
print("  Predicted: cognitive/affective categories >25%; concrete nouns <15%.")


# ============================================================================
# Save
# ============================================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp75_results.npz",
    axis_names=np.array(axis_names),
    substrate_cosines=np.array([c if c is not None else np.nan
                                for c in substrate_cosines]),
    ftxt_inter_axis=M_ftxt,
    category_means={cat: float(np.mean(cs))
                    for cat, cs in category_results.items()},
)
print("\nSaved exp75_results.npz")
