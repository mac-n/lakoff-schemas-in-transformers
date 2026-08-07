"""
exp136_per_axis_stability.py — for each axis in each cluster, measure
cross-layer stability of that single direction.

Niamh's question (2026-06-07): logical operators had low CLUSTER cross-layer
preservation (+0.78, at null level). But maybe some individual axes within
the cluster are stable across layers, even if the relational structure
between them isn't preserved.

Per-axis metrics:
- Adjacent-layer cos (smooth drift across depth)
- Mean off-diagonal cos (overall similarity across all layer pairs)
- Range from L4 to L20 (a working-layer range)
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


# Same cluster definitions as exp135
LOGICAL_OPERATORS = {
    "conjunction_disjunction": (["and", "plus", "with", "both", "together", "also"],
                                ["or", "either", "alternatively", "otherwise", "whether"]),
    "affirmation_negation": (["yes", "is", "all", "always", "everything", "everywhere"],
                             ["not", "no", "never", "none", "nothing", "nowhere"]),
    "conditional_assertion": (["if", "when", "suppose", "assuming", "whenever"],
                              ["then", "therefore", "thus", "because", "since"]),
    "necessity_possibility": (["must", "will", "certainly", "definitely", "always"],
                              ["may", "might", "can", "perhaps", "possibly", "maybe"]),
    "cause_contrast": (["because", "since", "due", "given", "as"],
                       ["despite", "although", "however", "yet", "nevertheless"]),
}

QUANTIFIERS = {
    "universal_existential": (["all", "every", "each", "everyone"],
                              ["some", "any", "several", "few"]),
    "universal_affirmative_negative": (["all", "everyone", "everything", "everywhere"],
                                       ["none", "no", "nothing", "nowhere", "nobody"]),
    "cardinal_many_few": (["many", "most", "plenty", "much", "lots"],
                          ["few", "little", "scarce", "rare", "less"]),
    "distributive_collective": (["each", "individually", "separately", "alone"],
                                ["together", "jointly", "collectively", "with"]),
    "precise_vague": (["exactly", "precisely", "specifically"],
                      ["about", "around", "approximately", "roughly"]),
}

DETERMINERS = {
    "definite_indefinite": (["the", "this", "that", "these", "those"],
                            ["a", "an", "some", "any"]),
    "proximal_distal": (["this", "these", "here", "now"],
                        ["that", "those", "there", "then"]),
    "sing_plur_demonstrative": (["this", "that"], ["these", "those"]),
    "possessive_neutral": (["my", "your", "his", "her"], ["the", "a", "an"]),
    "interrogative_assertive": (["which", "what", "whose"], ["this", "that", "the"]),
}

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

LAKOFF_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# Gather all words
all_words = set(COMMON + RARE)
for cluster in [LOGICAL_OPERATORS, QUANTIFIERS, DETERMINERS]:
    for axis_name, (pos, neg) in cluster.items():
        all_words.update(pos); all_words.update(neg)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
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


# ============================================================================
# Per-axis cross-layer stability
# ============================================================================

# Collect all axes from all clusters
ALL_AXES = {}  # (cluster, axis_name) -> [N_LAYERS, D_MODEL]

for cname, cluster in [("LOGICAL_OPERATORS", LOGICAL_OPERATORS),
                        ("QUANTIFIERS", QUANTIFIERS),
                        ("DETERMINERS", DETERMINERS)]:
    for axis_name, (pos, neg) in cluster.items():
        ALL_AXES[(cname, axis_name)] = np.array([build_pair_direction(pos, neg, L) for L in LAYERS])

for sn in SUFFIX_PAIRS:
    ALL_AXES[("MORPHOLOGY", sn)] = np.array([build_suffix_direction(sn, L) for L in LAYERS])

for sn in LAKOFF_NAMES:
    pairs = LAKOFF_SCHEMAS_MML[sn]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    ALL_AXES[("LAKOFF", sn)] = np.array([build_pair_direction(pos, neg, L) for L in LAYERS])


# ============================================================================
# For each axis, compute:
#   - adjacent-layer cos (mean over adjacent pairs L4-L22)
#   - mean off-diag cross-layer cos (L4-L22 range, excluding output)
# ============================================================================

WORK_LAYERS = list(range(4, 23))  # exclude L0-3 and L23

print(f"\n{'='*78}")
print(f"PER-AXIS CROSS-LAYER STABILITY (working range L4-L22)")
print(f"{'='*78}")

# Sort axes by cluster for readability
clusters_order = ["LAKOFF", "MORPHOLOGY", "LOGICAL_OPERATORS", "QUANTIFIERS", "DETERMINERS"]

results = {}
for (cname, axis_name), dirs in ALL_AXES.items():
    work_dirs = dirs[WORK_LAYERS]
    # Adjacent cos
    adj = [float(work_dirs[i] @ work_dirs[i+1]) for i in range(len(WORK_LAYERS)-1)]
    # Cross-layer cos matrix
    cos_mat = work_dirs @ work_dirs.T
    off = ~np.eye(len(WORK_LAYERS), dtype=bool)
    mean_off = float(cos_mat[off].mean())
    min_off = float(cos_mat[off].min())
    results[(cname, axis_name)] = {
        "adj_mean": np.mean(adj),
        "adj_min": np.min(adj),
        "cross_layer_mean": mean_off,
        "cross_layer_min": min_off,
    }

for cname in clusters_order:
    print(f"\n  --- {cname} ---")
    print(f"  {'axis':<32}  {'adj mean':>8}  {'adj min':>8}  {'xL mean':>8}  {'xL min':>8}")
    for (c, a), r in sorted(results.items()):
        if c != cname:
            continue
        print(f"  {a:<32}  {r['adj_mean']:>+8.3f}  {r['adj_min']:>+8.3f}  "
              f"{r['cross_layer_mean']:>+8.3f}  {r['cross_layer_min']:>+8.3f}")


# ============================================================================
# Summary ranking — most-to-least stable
# ============================================================================

print(f"\n{'='*78}")
print("RANKING — all axes by cross-layer mean cos (most stable first)")
print(f"{'='*78}\n")

sorted_axes = sorted(results.items(), key=lambda kv: -kv[1]["cross_layer_mean"])
print(f"  {'rank':<5}  {'cluster':<20}  {'axis':<32}  {'xL mean':>8}  {'adj mean':>8}")
for rank, ((cname, aname), r) in enumerate(sorted_axes, 1):
    print(f"  {rank:<5}  {cname:<20}  {aname:<32}  "
          f"{r['cross_layer_mean']:>+8.3f}  {r['adj_mean']:>+8.3f}")
