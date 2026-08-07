"""
exp137_anisotropy_check.py — for each axis in each cluster, compute
cos(axis_direction, anisotropy_direction) at each layer.

If determiner axes have low cos with anisotropy → their stability is real.
If high → may be inflated by uncancelled anisotropy.
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


# Cluster definitions (same as exp135/136)
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
# Compute anisotropy direction at each layer (mean of all residuals)
# ============================================================================

anisotropy_per_layer = []
for L in LAYERS:
    all_r = np.stack([residuals[w][L] for w in all_words], axis=0)
    mean_r = all_r.mean(axis=0)
    anisotropy_per_layer.append(mean_r / np.linalg.norm(mean_r))
anisotropy_per_layer = np.array(anisotropy_per_layer)


# ============================================================================
# For each axis, compute cos(axis, anisotropy) at L4, L12, L20, and mean across L4-L22
# ============================================================================

WORK_LAYERS = list(range(4, 23))

print(f"\n{'='*78}")
print("cos(axis_direction, anisotropy_direction) — should be near 0 if axis is clean")
print(f"{'='*78}")

results = {}

def report(cluster, axes_dict, builder):
    print(f"\n  --- {cluster} ---")
    print(f"  {'axis':<32}  {'L4':>7}  {'L12':>7}  {'L20':>7}  {'|mean|':>8}  {'max |.|':>8}")
    for axis_name, payload in axes_dict.items():
        cosines = []
        for L in LAYERS:
            if cluster == "MORPHOLOGY":
                d = builder(axis_name, L)
            else:
                pos, neg = payload
                d = builder(pos, neg, L)
            cosines.append(float(d @ anisotropy_per_layer[L]))
        cosines = np.array(cosines)
        work = cosines[WORK_LAYERS]
        results[(cluster, axis_name)] = cosines
        print(f"  {axis_name:<32}  {cosines[4]:>+7.3f}  {cosines[12]:>+7.3f}  "
              f"{cosines[20]:>+7.3f}  {np.abs(work).mean():>8.3f}  {np.abs(work).max():>8.3f}")

report("LOGICAL_OPERATORS", LOGICAL_OPERATORS, build_pair_direction)
report("QUANTIFIERS", QUANTIFIERS, build_pair_direction)
report("DETERMINERS", DETERMINERS, build_pair_direction)
report("MORPHOLOGY", {sn: None for sn in SUFFIX_PAIRS}, build_suffix_direction)
# Lakoff
print(f"\n  --- LAKOFF ---")
print(f"  {'axis':<32}  {'L4':>7}  {'L12':>7}  {'L20':>7}  {'|mean|':>8}  {'max |.|':>8}")
for sn in LAKOFF_NAMES:
    pairs = LAKOFF_SCHEMAS_MML[sn]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    cosines = []
    for L in LAYERS:
        d = build_pair_direction(pos, neg, L)
        cosines.append(float(d @ anisotropy_per_layer[L]))
    cosines = np.array(cosines)
    work = cosines[WORK_LAYERS]
    results[("LAKOFF", sn)] = cosines
    print(f"  {sn:<32}  {cosines[4]:>+7.3f}  {cosines[12]:>+7.3f}  "
          f"{cosines[20]:>+7.3f}  {np.abs(work).mean():>8.3f}  {np.abs(work).max():>8.3f}")


# ============================================================================
# Summary: which axes are most anisotropy-contaminated?
# ============================================================================

print(f"\n{'='*78}")
print("RANKING — axes by mean |cos with anisotropy| (most contaminated first)")
print(f"{'='*78}\n")
print(f"  {'rank':<5}  {'cluster':<20}  {'axis':<32}  {'mean |cos|':>10}")
sorted_axes = sorted(results.items(),
                     key=lambda kv: -np.abs(kv[1][WORK_LAYERS]).mean())
for rank, ((cname, aname), cosines) in enumerate(sorted_axes, 1):
    print(f"  {rank:<5}  {cname:<20}  {aname:<32}  "
          f"{np.abs(cosines[WORK_LAYERS]).mean():>10.3f}")
