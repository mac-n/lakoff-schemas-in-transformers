"""
exp135_compositionality_test.py — does compositionality predict
differentiated vs collapsed cluster geometry?

Hypothesis (Niamh, 2026-06-07): operator clusters where members compose with
each other show differentiated relational structure (like Lakoff schemas).
Operator clusters where members are mutually exclusive at the same slot
collapse to a single direction (like morphology).

Test clusters:
- LOGICAL_OPERATORS (predicted differentiated)
- QUANTIFIERS (predicted differentiated)
- DETERMINERS (predicted collapsed, control)

Reference clusters (already known):
- MORPHOLOGY (confirmed collapsed)
- LAKOFF (confirmed differentiated)

Metrics per cluster per layer:
- Within-cluster cosine matrix (signature)
- Variance of off-diagonal cosines (high = differentiated, low = collapsed)
- Cross-layer signature preservation (high = stable, low = noisy)
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers
LAYERS = list(range(N_LAYERS))
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]


# ============================================================================
# CLUSTER DEFINITIONS
# ============================================================================

# Each axis is a pair (POS_list, NEG_list) — direction = mean(POS) − mean(NEG)

LOGICAL_OPERATORS = {
    "conjunction_disjunction": (
        ["and", "plus", "with", "both", "together", "also"],
        ["or", "either", "alternatively", "otherwise", "whether"]
    ),
    "affirmation_negation": (
        ["yes", "is", "all", "always", "everything", "everywhere"],
        ["not", "no", "never", "none", "nothing", "nowhere"]
    ),
    "conditional_assertion": (
        ["if", "when", "suppose", "assuming", "whenever"],
        ["then", "therefore", "thus", "because", "since"]
    ),
    "necessity_possibility": (
        ["must", "will", "certainly", "definitely", "always"],
        ["may", "might", "can", "perhaps", "possibly", "maybe"]
    ),
    "cause_contrast": (
        ["because", "since", "due", "given", "as"],
        ["despite", "although", "however", "yet", "nevertheless"]
    ),
}

QUANTIFIERS = {
    "universal_existential": (
        ["all", "every", "each", "everyone"],
        ["some", "any", "several", "few"]
    ),
    "universal_affirmative_negative": (
        ["all", "everyone", "everything", "everywhere"],
        ["none", "no", "nothing", "nowhere", "nobody"]
    ),
    "cardinal_many_few": (
        ["many", "most", "plenty", "much", "lots"],
        ["few", "little", "scarce", "rare", "less"]
    ),
    "distributive_collective": (
        ["each", "individually", "separately", "alone"],
        ["together", "jointly", "collectively", "with"]
    ),
    "precise_vague": (
        ["exactly", "precisely", "specifically"],
        ["about", "around", "approximately", "roughly"]
    ),
}

DETERMINERS = {
    "definite_indefinite": (
        ["the", "this", "that", "these", "those"],
        ["a", "an", "some", "any"]
    ),
    "proximal_distal": (
        ["this", "these", "here", "now"],
        ["that", "those", "there", "then"]
    ),
    "sing_plur_demonstrative": (
        ["this", "that"],
        ["these", "those"]
    ),
    "possessive_neutral": (
        ["my", "your", "his", "her"],
        ["the", "a", "an"]
    ),
    "interrogative_assertive": (
        ["which", "what", "whose"],
        ["this", "that", "the"]
    ),
}

# Reference: morphology (suffix directions)
SUFFIX_PAIRS = {
    "ER_comparative": [("big","bigger"),("small","smaller"),("tall","taller"),
        ("high","higher"),("low","lower"),("deep","deeper"),("wide","wider"),
        ("fast","faster"),("slow","slower"),("old","older"),("new","newer"),
        ("hot","hotter"),("cold","colder"),("hard","harder"),("soft","softer")],
    "EST_superlative": [("big","biggest"),("small","smallest"),("tall","tallest"),
        ("high","highest"),("low","lowest"),("deep","deepest"),("old","oldest"),
        ("new","newest"),("hot","hottest"),("cold","coldest"),("hard","hardest")],
    "ING_progressive": [("walk","walking"),("run","running"),("jump","jumping"),
        ("sit","sitting"),("stand","standing"),("swim","swimming"),("think","thinking"),
        ("talk","talking"),("sing","singing"),("dance","dancing"),("play","playing"),
        ("work","working"),("read","reading"),("write","writing"),("eat","eating")],
    "ED_past": [("walk","walked"),("jump","jumped"),("look","looked"),
        ("talk","talked"),("play","played"),("work","worked"),("ask","asked"),
        ("call","called"),("learn","learned"),("move","moved"),("stop","stopped"),
        ("start","started")],
    "S_plural": [("cat","cats"),("dog","dogs"),("book","books"),("house","houses"),
        ("car","cars"),("tree","trees"),("bird","birds"),("hand","hands"),
        ("eye","eyes"),("girl","girls"),("boy","boys"),("year","years")],
    "UN_negation": [("happy","unhappy"),("kind","unkind"),("healthy","unhealthy"),
        ("safe","unsafe"),("clear","unclear"),("clean","unclean"),("fair","unfair"),
        ("certain","uncertain"),("known","unknown"),("seen","unseen")],
    "RE_repetition": [("do","redo"),("make","remake"),("build","rebuild"),
        ("write","rewrite"),("read","reread"),("start","restart"),
        ("create","recreate"),("paint","repaint")],
}

# Reference: Lakoff schemas (from LAKOFF_SCHEMAS_MML)
LAKOFF_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# ============================================================================
# Collect all words
# ============================================================================

all_words = set(COMMON + RARE)

# Logical, quantifiers, determiners
for cluster in [LOGICAL_OPERATORS, QUANTIFIERS, DETERMINERS]:
    for axis_name, (pos, neg) in cluster.items():
        all_words.update(pos); all_words.update(neg)

# Morphology
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)

# Lakoff
for sn in LAKOFF_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)

all_words = sorted(all_words)
print(f"\nCollecting residuals for {len(all_words)} words at all {N_LAYERS} layers...")

residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = np.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy() for L in LAYERS],
        axis=0
    )
    if (k+1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)


def build_pair_direction(pos_words, neg_words, layer, freq_strip=True):
    raw = mean_acts(pos_words, layer) - mean_acts(neg_words, layer)
    raw = raw / np.linalg.norm(raw)
    if freq_strip:
        freq_raw = mean_acts(COMMON, layer) - mean_acts(RARE, layer)
        freq = freq_raw / np.linalg.norm(freq_raw)
        raw = raw - (raw @ freq) * freq
        raw = raw / np.linalg.norm(raw)
    return raw


def build_suffix_direction(suffix_name, layer):
    pairs = SUFFIX_PAIRS[suffix_name]
    diffs = []
    for base, infl in pairs:
        b = residuals[base][layer]; i = residuals[infl][layer]
        b_unit = b / np.linalg.norm(b); i_unit = i / np.linalg.norm(i)
        diffs.append(i_unit - b_unit)
    raw = np.mean(diffs, axis=0)
    return raw / np.linalg.norm(raw)


def build_lakoff_direction(schema_name, layer):
    pairs = LAKOFF_SCHEMAS_MML[schema_name]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    return build_pair_direction(pos, neg, layer, freq_strip=True)


# ============================================================================
# Build all cluster directions per layer
# ============================================================================

print("\nBuilding cluster directions per layer...")

CLUSTERS = {
    "LOGICAL_OPERATORS": LOGICAL_OPERATORS,
    "QUANTIFIERS": QUANTIFIERS,
    "DETERMINERS": DETERMINERS,
}

# directions[cluster_name][axis_name] = [N_LAYERS, D_MODEL]
directions = {}

for cname, cluster in CLUSTERS.items():
    directions[cname] = {}
    for axis_name, (pos, neg) in cluster.items():
        directions[cname][axis_name] = np.array(
            [build_pair_direction(pos, neg, L) for L in LAYERS]
        )

# Reference clusters
directions["MORPHOLOGY"] = {sn: np.array([build_suffix_direction(sn, L) for L in LAYERS])
                            for sn in SUFFIX_PAIRS}
directions["LAKOFF"] = {sn: np.array([build_lakoff_direction(sn, L) for L in LAYERS])
                        for sn in LAKOFF_NAMES}


# ============================================================================
# Compute within-cluster cosine matrix per layer + diagnostics
# ============================================================================

print("\n" + "=" * 78)
print("PER-CLUSTER STATISTICS")
print("=" * 78)

results = {}
for cname, axes in directions.items():
    axis_names = list(axes.keys())
    N_axes = len(axis_names)
    # Stack: [N_LAYERS, N_axes, D_MODEL]
    stack = np.stack([axes[a] for a in axis_names], axis=1)
    # Per-layer cos matrix
    cos_matrices = np.zeros((N_LAYERS, N_axes, N_axes))
    for L in LAYERS:
        cos_matrices[L] = stack[L] @ stack[L].T

    # Off-diagonal stats per layer
    off_mask = ~np.eye(N_axes, dtype=bool)
    off_means = []
    off_stds = []
    off_abs_means = []
    for L in LAYERS:
        off_vals = cos_matrices[L][off_mask]
        off_means.append(off_vals.mean())
        off_stds.append(off_vals.std())
        off_abs_means.append(np.abs(off_vals).mean())

    # Cross-layer signature preservation
    # Vectorise upper triangle at each layer
    sigs = np.array([cos_matrices[L][np.triu_indices(N_axes, k=1)] for L in LAYERS])
    layer_sim = np.zeros((N_LAYERS, N_LAYERS))
    for a in range(N_LAYERS):
        for b in range(N_LAYERS):
            na = np.linalg.norm(sigs[a]); nb = np.linalg.norm(sigs[b])
            if na > 1e-9 and nb > 1e-9:
                layer_sim[a, b] = sigs[a] @ sigs[b] / (na * nb)

    off_layer_mask = ~np.eye(N_LAYERS, dtype=bool)
    cross_layer_preservation = layer_sim[off_layer_mask].mean()

    results[cname] = {
        "N_axes": N_axes,
        "off_mean_per_layer": off_means,
        "off_std_per_layer": off_stds,
        "off_abs_mean_per_layer": off_abs_means,
        "cross_layer_preservation": cross_layer_preservation,
        "cos_matrices": cos_matrices,
    }

    # Print summary
    print(f"\n{cname} ({N_axes} axes):")
    print(f"  axes: {axis_names}")
    print(f"  {'layer':<6}  {'off mean':>8}  {'off std':>8}  {'off |mean|':>10}")
    for L_show in [0, 4, 8, 12, 16, 20, 23]:
        print(f"  L{L_show:<5}  {off_means[L_show]:>+8.3f}  "
              f"{off_stds[L_show]:>8.3f}  {off_abs_means[L_show]:>10.3f}")
    print(f"  Mean off-diag std across layers: {np.mean(off_stds):.4f}  "
          f"(HIGH = differentiated, LOW = collapsed)")
    print(f"  Cross-layer signature preservation: {cross_layer_preservation:+.4f}  "
          f"(HIGH = stable structure across layers)")


# ============================================================================
# Diagnostic: classify each cluster
# ============================================================================

print("\n" + "=" * 78)
print("CLASSIFICATION — predicted vs observed")
print("=" * 78)

print(f"\n  {'cluster':<22}  {'mean off std':>15}  {'cross-layer pres':>18}  "
      f"{'classification':>25}")
for cname in ["LAKOFF", "MORPHOLOGY", "LOGICAL_OPERATORS", "QUANTIFIERS", "DETERMINERS"]:
    r = results[cname]
    mean_std = np.mean(r["off_std_per_layer"])
    preservation = r["cross_layer_preservation"]

    if mean_std > 0.1 and preservation > 0.5:
        classification = "DIFFERENTIATED + STABLE"
    elif mean_std < 0.05 and preservation > 0.5:
        classification = "COLLAPSED + STABLE"
    elif mean_std > 0.1 and preservation < 0.3:
        classification = "DIFFERENTIATED + NOISY"
    else:
        classification = "INTERMEDIATE"

    print(f"  {cname:<22}  {mean_std:>15.4f}  {preservation:>+18.3f}  "
          f"{classification:>25}")


# Predicted-vs-observed table
print("\n  Predictions for compositionality hypothesis:")
print("    LAKOFF              : DIFFERENTIATED + STABLE (composes with itself)")
print("    LOGICAL_OPERATORS   : DIFFERENTIATED + STABLE (composes)")
print("    QUANTIFIERS         : DIFFERENTIATED + STABLE (composes)")
print("    MORPHOLOGY          : COLLAPSED + STABLE (mutually exclusive)")
print("    DETERMINERS         : COLLAPSED + STABLE (mutually exclusive)")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp135_results.npz",
         **{f"{cname}_cos_matrices": r["cos_matrices"]
            for cname, r in results.items()},
         **{f"{cname}_off_std_per_layer": np.array(r["off_std_per_layer"])
            for cname, r in results.items()},
         **{f"{cname}_cross_layer_preservation": r["cross_layer_preservation"]
            for cname, r in results.items()})
print("\nSaved exp135_results.npz")
