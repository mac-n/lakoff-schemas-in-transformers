"""
exp139_morphology_lakoff_system.py — do morphology and Lakoff schemas form
a COHERENT INTEGRATED SYSTEM whose relational geometry is preserved across
layers?

After exp138 (proper anisotropy strip), the suffix × schema cosine matrix
shows a stable shape across layers visually. Test this rigorously:

1. Build 7×8 morphology×Lakoff cosine matrices at every layer (cleaned)
2. Cross-layer signature similarity: cos(vec(M_L1), vec(M_L2))
3. K=100 nulls: shuffle words within suffix/schema pools, rebuild
   pseudo-matrices, measure cross-layer preservation
4. If real ≫ null, morphology and Lakoff schemas form a coherent system
   where the cross-cluster relationships are layer-stable

If this holds: the paper's framing becomes "vector grounding establishes
a unified geometric system spanning grammar (morphology) and semantics
(Lakoff schemas) that's preserved across transformer processing."
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

SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]

# Collect words
all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
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

anisotropy_dirs = []
for L in LAYERS:
    all_r = np.stack([residuals[w][L] for w in all_words], axis=0)
    m = all_r.mean(axis=0)
    anisotropy_dirs.append(m / np.linalg.norm(m))


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)


def strip_aniso_freq(direction, layer):
    freq_raw = mean_acts(COMMON, layer) - mean_acts(RARE, layer)
    freq = freq_raw / np.linalg.norm(freq_raw)
    aniso = anisotropy_dirs[layer]
    direction = direction - (direction @ aniso) * aniso
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)
    direction = direction - (direction @ freq_orth) * freq_orth
    return direction / np.linalg.norm(direction)


def build_schema(pos, neg, layer):
    raw = mean_acts(pos, layer) - mean_acts(neg, layer)
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


def build_suffix(pairs, layer):
    diffs = []
    for base, infl in pairs:
        b = residuals[base][layer]; i = residuals[infl][layer]
        b_unit = b / np.linalg.norm(b); i_unit = i / np.linalg.norm(i)
        diffs.append(i_unit - b_unit)
    raw = np.mean(diffs, axis=0)
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


# ============================================================================
# REAL: morphology × Lakoff matrices at each layer
# ============================================================================

N_SUFFIX = len(SUFFIX_PAIRS)
N_SCHEMA = len(SCHEMA_NAMES)
real_matrices = np.zeros((N_LAYERS, N_SUFFIX, N_SCHEMA))

print("\nBuilding real morphology × Lakoff matrices at all layers...")
for L in LAYERS:
    suffix_dirs = [build_suffix(SUFFIX_PAIRS[sn], L) for sn in SUFFIX_PAIRS]
    schema_dirs = [build_schema(sorted(set(p[0] for p in LAKOFF_SCHEMAS_MML[sn])),
                                sorted(set(p[1] for p in LAKOFF_SCHEMAS_MML[sn])), L)
                   for sn in SCHEMA_NAMES]
    for i in range(N_SUFFIX):
        for j in range(N_SCHEMA):
            real_matrices[L, i, j] = float(suffix_dirs[i] @ schema_dirs[j])


# Cross-layer signature similarity
real_sigs = real_matrices.reshape(N_LAYERS, -1)  # [L, 56]
real_layer_sim = np.zeros((N_LAYERS, N_LAYERS))
for a in range(N_LAYERS):
    for b in range(N_LAYERS):
        na = np.linalg.norm(real_sigs[a]); nb = np.linalg.norm(real_sigs[b])
        if na > 1e-9 and nb > 1e-9:
            real_layer_sim[a, b] = real_sigs[a] @ real_sigs[b] / (na * nb)


# ============================================================================
# NULL: K trials of shuffled-word morphology × Lakoff matrices
# ============================================================================

# Word pools
suffix_pool_base = list(set(b for pairs in SUFFIX_PAIRS.values() for b, _ in pairs))
suffix_pool_infl = list(set(i for pairs in SUFFIX_PAIRS.values() for _, i in pairs))
schema_pool = list(set(w for sn in SCHEMA_NAMES
                       for p, n in LAKOFF_SCHEMAS_MML[sn]
                       for w in (p, n)))

K_NULL = 50
rng = np.random.default_rng(42)
print(f"\nRunning K={K_NULL} null trials...")

null_layer_sims = np.zeros((K_NULL, N_LAYERS, N_LAYERS))

# Suffix sizes (number of pairs per suffix)
suffix_sizes = [len(SUFFIX_PAIRS[sn]) for sn in SUFFIX_PAIRS]
schema_sizes = [(len(set(p[0] for p in LAKOFF_SCHEMAS_MML[sn])),
                 len(set(p[1] for p in LAKOFF_SCHEMAS_MML[sn])))
                for sn in SCHEMA_NAMES]

for k in range(K_NULL):
    null_matrices = np.zeros((N_LAYERS, N_SUFFIX, N_SCHEMA))
    # Create K fake suffix pair lists by random sampling
    base_perm = list(suffix_pool_base); rng.shuffle(base_perm)
    infl_perm = list(suffix_pool_infl); rng.shuffle(infl_perm)
    fake_suffix_pairs = []
    cursor_b = 0; cursor_i = 0
    for sz in suffix_sizes:
        pairs = []
        for _ in range(sz):
            if cursor_b >= len(base_perm):
                rng.shuffle(base_perm); cursor_b = 0
            if cursor_i >= len(infl_perm):
                rng.shuffle(infl_perm); cursor_i = 0
            pairs.append((base_perm[cursor_b], infl_perm[cursor_i]))
            cursor_b += 1; cursor_i += 1
        fake_suffix_pairs.append(pairs)
    # Fake schema word lists
    schema_perm = list(schema_pool); rng.shuffle(schema_perm)
    fake_schemas = []
    cursor = 0
    for n_pos, n_neg in schema_sizes:
        if cursor + n_pos + n_neg > len(schema_perm):
            rng.shuffle(schema_perm); cursor = 0
        pos = schema_perm[cursor:cursor + n_pos]
        neg = schema_perm[cursor + n_pos:cursor + n_pos + n_neg]
        fake_schemas.append((pos, neg))
        cursor += n_pos + n_neg

    for L in LAYERS:
        fake_suf_dirs = [build_suffix(fp, L) for fp in fake_suffix_pairs]
        fake_sch_dirs = [build_schema(pos, neg, L) for pos, neg in fake_schemas]
        for i in range(N_SUFFIX):
            for j in range(N_SCHEMA):
                null_matrices[L, i, j] = float(fake_suf_dirs[i] @ fake_sch_dirs[j])

    null_sigs = null_matrices.reshape(N_LAYERS, -1)
    for a in range(N_LAYERS):
        for b in range(N_LAYERS):
            na = np.linalg.norm(null_sigs[a]); nb = np.linalg.norm(null_sigs[b])
            if na > 1e-9 and nb > 1e-9:
                null_layer_sims[k, a, b] = null_sigs[a] @ null_sigs[b] / (na * nb)

    if (k+1) % 10 == 0:
        print(f"  null trial {k+1}/{K_NULL}")


# ============================================================================
# Results
# ============================================================================

off = ~np.eye(N_LAYERS, dtype=bool)
real_off = real_layer_sim[off]
null_off = np.array([null_layer_sims[k][off].mean() for k in range(K_NULL)])

print("\n" + "=" * 78)
print("CROSS-LAYER PRESERVATION of morphology × Lakoff matrix")
print("=" * 78)
print(f"\n  REAL mean off-diag layer similarity: {real_off.mean():+.4f}")
print(f"  NULL mean off-diag (K={K_NULL}):       {null_off.mean():+.4f}")
print(f"  NULL std:                              {null_off.std():.4f}")
print(f"  NULL 95th percentile:                  {np.percentile(null_off, 95):+.4f}")
print(f"  Effect size (real - null mean):        {real_off.mean() - null_off.mean():+.4f}")
print(f"  Z-score:                               "
      f"{(real_off.mean() - null_off.mean()) / null_off.std():+.2f}")

# Detail at working layers
print("\n  Real layer-similarity at every 4 layers:")
display = list(range(0, N_LAYERS, 4))
print(f"  {'':<6}" + "".join(f"  L{L:>2}  " for L in display))
for a in display:
    row = f"  L{a:>2}  "
    for b in display:
        row += f" {real_layer_sim[a, b]:+.2f} "
    print(row)


np.savez("/Users/macn/Documents/embeddingexp/exp139_results.npz",
         real_layer_sim=real_layer_sim,
         null_layer_sims=null_layer_sims,
         real_matrices=real_matrices)
print("\nSaved exp139_results.npz")
