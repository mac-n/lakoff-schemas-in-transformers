"""
exp134_morphology_axis_stability.py — is the morphology direction itself
stable across layers, or does it rotate?

Compute morphology direction m_L = unit(mean of 7 suffix directions at L).
Then compute cos(m_L1, m_L2) for all layer pairs.

If cos ≈ 1 across layers: morphology is a fixed axis in residual stream.
                          Schema-projection variation reflects schema rotation.
If cos varies: morphology itself rotates. Less clear what's varying.

Also: compute cos(m_L, lakoff_schema_dir_L) across layers to see whether
specific schemas track morphology more than others.
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

all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
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


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)

def build_schema_direction(schema_name, layer):
    pairs = LAKOFF_SCHEMAS_MML[schema_name]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    raw = mean_acts(pos, layer) - mean_acts(neg, layer)
    raw = raw / np.linalg.norm(raw)
    freq_raw = mean_acts(COMMON, layer) - mean_acts(RARE, layer)
    freq = freq_raw / np.linalg.norm(freq_raw)
    clean = raw - (raw @ freq) * freq
    return clean / np.linalg.norm(clean)

def build_suffix_direction(suffix_name, layer):
    pairs = SUFFIX_PAIRS[suffix_name]
    diffs = []
    for base, infl in pairs:
        b = residuals[base][layer]; i = residuals[infl][layer]
        b_unit = b / np.linalg.norm(b); i_unit = i / np.linalg.norm(i)
        diffs.append(i_unit - b_unit)
    raw = np.mean(diffs, axis=0)
    return raw / np.linalg.norm(raw)

def build_morphology_direction(layer):
    """Mean of all suffix directions, normalised."""
    suffix_dirs = np.array([build_suffix_direction(sn, layer) for sn in SUFFIX_PAIRS])
    m = suffix_dirs.mean(axis=0)
    return m / np.linalg.norm(m)


# ============================================================================
# Morphology direction stability across layers
# ============================================================================

print("\nComputing morphology direction at each layer...")
m_per_layer = np.array([build_morphology_direction(L) for L in LAYERS])
schema_dirs_per_layer = {sn: np.array([build_schema_direction(sn, L) for L in LAYERS])
                         for sn in SCHEMA_NAMES}

# Cross-layer cosines of morphology direction
m_cross_layer = m_per_layer @ m_per_layer.T  # [N_LAYERS, N_LAYERS]

print("\n" + "=" * 78)
print("MORPHOLOGY DIRECTION cross-layer cosine matrix")
print("=" * 78)
print(f"\n  {'':<6}" + "".join(f"  L{L:>2}  " for L in range(0, N_LAYERS, 4)))
for a in range(0, N_LAYERS, 2):
    row = f"  L{a:>2}  "
    for b in range(0, N_LAYERS, 4):
        row += f" {m_cross_layer[a, b]:+.2f} "
    print(row)

off = ~np.eye(N_LAYERS, dtype=bool)
print(f"\n  Mean off-diagonal (cross-layer cos): {m_cross_layer[off].mean():+.4f}")
print(f"  Adjacent-layer cos (L_n vs L_{{n+1}}):")
for L in range(N_LAYERS - 1):
    print(f"    L{L:>2} ↔ L{L+1:<2}: {m_cross_layer[L, L+1]:+.3f}")


# ============================================================================
# Schema direction stability across layers — for comparison
# ============================================================================

print("\n" + "=" * 78)
print("SCHEMA DIRECTION cross-layer mean cosines (for comparison)")
print("=" * 78)
for sn in SCHEMA_NAMES:
    sd = schema_dirs_per_layer[sn]
    cross = sd @ sd.T
    print(f"  {sn:<22}: mean off-diag = {cross[off].mean():+.3f}  "
          f"adjacent-layer mean = {np.mean([cross[L, L+1] for L in range(N_LAYERS-1)]):+.3f}")


# ============================================================================
# Does morphology direction track ANY schema systematically?
# ============================================================================

print("\n" + "=" * 78)
print("cos(morphology_L, schema_L) across layers — does morphology track a schema?")
print("=" * 78)

print(f"\n  {'layer':<6}  " + "  ".join(f"{sn[:9]:>9}" for sn in SCHEMA_NAMES))
for L in range(0, N_LAYERS, 2):
    m = m_per_layer[L]
    row = f"  L{L:>3}  "
    for sn in SCHEMA_NAMES:
        sd = schema_dirs_per_layer[sn][L]
        row += f"  {float(m @ sd):>+8.3f}"
    print(row)

# Aggregate: mean and std of (m, schema) across layers
print(f"\n  schema:                  mean(m,schema)   std    range")
for sn in SCHEMA_NAMES:
    cosines = [float(m_per_layer[L] @ schema_dirs_per_layer[sn][L]) for L in LAYERS]
    print(f"  {sn:<22}: {np.mean(cosines):>+10.3f}   {np.std(cosines):.3f}    "
          f"[{min(cosines):+.3f}, {max(cosines):+.3f}]")


np.savez("/Users/macn/Documents/embeddingexp/exp134_results.npz",
         m_cross_layer=m_cross_layer,
         m_per_layer=m_per_layer)
print("\nSaved exp134_results.npz")
