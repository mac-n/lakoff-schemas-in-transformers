"""
exp133_suffix_relational.py — do suffix-suffix relationships preserve
across layers better than random vectors do? (exp123 for morphology)

Same methodology as exp123 (Lakoff schema relational structure):
1. For each layer, compute the suffix-suffix cosine matrix (7x7)
2. Vectorise the upper triangle (21 unique pairs) → "signature" per layer
3. Cross-layer similarity = cos(signature_L1, signature_L2)
4. Compare to null: random pseudo-suffix groups of 7

If real cross-layer signature similarity > null, morphology preserves
its relational structure across layers, suggesting "morphology space"
has consistent geometry.
"""

import numpy as np
import torch
from itertools import combinations
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers
print(f"  {N_LAYERS} layers")

# Sweep all layers this time
LAYERS = list(range(N_LAYERS))
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

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
SUFFIX_NAMES = list(SUFFIX_PAIRS.keys())
N_SUFFIX = len(SUFFIX_NAMES)
N_PAIRS = N_SUFFIX * (N_SUFFIX - 1) // 2

all_words = set()
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
all_words = sorted(all_words)

print(f"\nExtracting residuals for {len(all_words)} words at all {N_LAYERS} layers...")
residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = np.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy() for L in LAYERS],
        axis=0
    )
    if (k+1) % 50 == 0:
        print(f"  {k+1}/{len(all_words)}")


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
# Real suffix relational signatures per layer
# ============================================================================

print("\nComputing real suffix relational signatures per layer...")
real_signatures = np.zeros((N_LAYERS, N_PAIRS))
suffix_dirs_per_layer = {}
for L in LAYERS:
    dirs = np.array([build_suffix_direction(sn, L) for sn in SUFFIX_NAMES])
    suffix_dirs_per_layer[L] = dirs
    cos_mat = dirs @ dirs.T  # [N_SUFFIX, N_SUFFIX]
    real_signatures[L] = cos_mat[np.triu_indices(N_SUFFIX, k=1)]


def sig_cos(s1, s2):
    n1 = np.linalg.norm(s1); n2 = np.linalg.norm(s2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float('nan')
    return float(s1 @ s2 / (n1 * n2))


# Cross-layer signature similarity matrix
real_layer_sim = np.zeros((N_LAYERS, N_LAYERS))
for a in range(N_LAYERS):
    for b in range(N_LAYERS):
        real_layer_sim[a, b] = sig_cos(real_signatures[a], real_signatures[b])


# ============================================================================
# Null: K trials of 7 random pseudo-suffixes each, same machinery
# ============================================================================

K_NULL = 100
PSEUDO_PAIRS_PER_FAKE = 12
all_word_list = list(all_words)

print(f"\nRunning K={K_NULL} null trials (groups of {N_SUFFIX} fake suffixes)...")
null_signatures = np.zeros((K_NULL, N_LAYERS, N_PAIRS))
rng = np.random.default_rng(seed=42)
for k in range(K_NULL):
    # Build N_SUFFIX fake suffix directions per layer using random word pairings
    pool = all_word_list.copy()
    for L in LAYERS:
        fake_dirs = []
        for s in range(N_SUFFIX):
            rng.shuffle(pool)
            diffs = []
            for p in range(PSEUDO_PAIRS_PER_FAKE):
                w1, w2 = pool[2*p], pool[2*p + 1]
                v1 = residuals[w1][L] / np.linalg.norm(residuals[w1][L])
                v2 = residuals[w2][L] / np.linalg.norm(residuals[w2][L])
                diffs.append(v2 - v1)
            raw = np.mean(diffs, axis=0)
            fake_dirs.append(raw / np.linalg.norm(raw))
        fake_dirs = np.array(fake_dirs)
        cos_mat = fake_dirs @ fake_dirs.T
        null_signatures[k, L] = cos_mat[np.triu_indices(N_SUFFIX, k=1)]
    if (k+1) % 20 == 0:
        print(f"  null trial {k+1}/{K_NULL}")


# Null per-trial cross-layer similarity
null_layer_sims = np.zeros((K_NULL, N_LAYERS, N_LAYERS))
for k in range(K_NULL):
    for a in range(N_LAYERS):
        for b in range(N_LAYERS):
            null_layer_sims[k, a, b] = sig_cos(null_signatures[k, a], null_signatures[k, b])


# ============================================================================
# Summary
# ============================================================================

off = ~np.eye(N_LAYERS, dtype=bool)
real_off_mean = real_layer_sim[off].mean()
null_off_means = np.array([null_layer_sims[k][off].mean() for k in range(K_NULL)])

print("\n" + "=" * 78)
print("MORPHOLOGY RELATIONAL STRUCTURE — cross-layer signature similarity")
print("=" * 78)
print(f"\n  REAL morphology mean off-diagonal cross-layer cos:  {real_off_mean:+.4f}")
print(f"  NULL pseudo-suffixes mean off-diagonal cos:")
print(f"    mean:   {null_off_means.mean():+.4f}")
print(f"    std:    {null_off_means.std():.4f}")
print(f"    95pct:  {np.percentile(null_off_means, 95):+.4f}")
print(f"    99pct:  {np.percentile(null_off_means, 99):+.4f}")
print(f"  Effect size (real - null mean): {real_off_mean - null_off_means.mean():+.4f}")
z_score = (real_off_mean - null_off_means.mean()) / null_off_means.std()
print(f"  Z-score:   {z_score:+.2f}")


# Print real layer-similarity matrix abbreviated
print(f"\n  Real layer-similarity matrix at every 4 layers:")
display_layers = list(range(0, N_LAYERS, 4))
print(f"  {'':<6}" + "".join(f"  L{L:>2}  " for L in display_layers))
for a in display_layers:
    row = f"  L{a:>2}  "
    for b in display_layers:
        row += f" {real_layer_sim[a, b]:+.2f} "
    print(row)


np.savez("/Users/macn/Documents/embeddingexp/exp133_results.npz",
         real_layer_sim=real_layer_sim,
         null_layer_sims=null_layer_sims,
         real_signatures=real_signatures,
         null_signatures=null_signatures)
print("\nSaved exp133_results.npz")
