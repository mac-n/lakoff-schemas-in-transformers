"""
exp131_anisotropy_vs_structure.py — is the convergent suffix pattern from
exp130 real structure or anisotropy?

Three discriminating tests:
1. Suffix-suffix cosine matrix per layer.
2. Project each suffix direction onto the per-layer mean residual direction
   (the anisotropy/common direction).
3. Pure random-pair null: differences between random words.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()

LAYERS_OF_INTEREST = [4, 8, 12, 16, 20]
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS_OF_INTEREST]

# Same suffix pairs as exp130
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

SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# Collect all needed words
all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b,i in pairs:
        all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
    for p,n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
all_words = sorted(all_words)
print(f"\nCollecting residuals for {len(all_words)} words at layers {LAYERS_OF_INTEREST}...")

residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                    for L in LAYERS_OF_INTEREST}
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


# ============================================================================
# TEST 1: suffix-suffix cosine matrix per layer
# ============================================================================

print("\n" + "=" * 78)
print("TEST 1 — suffix × suffix cosines (do all suffixes cluster together?)")
print("=" * 78)

for L in LAYERS_OF_INTEREST:
    print(f"\nLayer {L}:")
    suffix_dirs = {sn: build_suffix_direction(sn, L) for sn in SUFFIX_NAMES}
    print(f"  {'':<18}" + "  ".join(f"{sn[:6]:>7}" for sn in SUFFIX_NAMES))
    cos_mat = np.zeros((len(SUFFIX_NAMES), len(SUFFIX_NAMES)))
    for i, s1 in enumerate(SUFFIX_NAMES):
        row = f"  {s1[:16]:<18}"
        for j, s2 in enumerate(SUFFIX_NAMES):
            cos_mat[i,j] = float(suffix_dirs[s1] @ suffix_dirs[s2])
            row += f"  {cos_mat[i,j]:+.3f}"
        print(row)
    off = cos_mat[~np.eye(len(SUFFIX_NAMES), dtype=bool)]
    print(f"  Mean off-diagonal: {off.mean():+.3f}  std: {off.std():.3f}")


# ============================================================================
# TEST 2: project suffix directions onto per-layer mean direction (anisotropy)
# ============================================================================

print("\n" + "=" * 78)
print("TEST 2 — cos(suffix, mean residual direction) — anisotropy alignment")
print("  HIGH (>0.6) = suffix is mostly anisotropy; LOW (~0) = suffix is orthogonal")
print("=" * 78)

for L in LAYERS_OF_INTEREST:
    print(f"\nLayer {L}:")
    # Anisotropy = mean of all word residuals at this layer
    all_resids = np.stack([residuals[w][L] for w in all_words], axis=0)
    aniso_raw = all_resids.mean(axis=0)
    aniso = aniso_raw / np.linalg.norm(aniso_raw)
    print(f"  ‖aniso‖={np.linalg.norm(aniso_raw):.2f}  (mean residual magnitude)")
    for sn in SUFFIX_NAMES:
        sd = build_suffix_direction(sn, L)
        cos_with_aniso = float(sd @ aniso)
        print(f"  {sn:<18}  cos(suffix, aniso) = {cos_with_aniso:+.3f}")


# ============================================================================
# TEST 3: pure random-word-pair null
# ============================================================================

print("\n" + "=" * 78)
print("TEST 3 — pure random-pair null (cos between TWO random word-pair directions)")
print("=" * 78)

K = 500
np.random.seed(42)
all_word_list = list(all_words)

for L in LAYERS_OF_INTEREST:
    # Generate K random pair-difference directions
    fake_dirs = []
    for _ in range(K):
        np.random.shuffle(all_word_list)
        n_pairs = 12
        diffs = []
        for p in range(n_pairs):
            w1, w2 = all_word_list[2*p], all_word_list[2*p + 1]
            v1 = residuals[w1][L] / np.linalg.norm(residuals[w1][L])
            v2 = residuals[w2][L] / np.linalg.norm(residuals[w2][L])
            diffs.append(v2 - v1)
        raw = np.mean(diffs, axis=0)
        fake_dirs.append(raw / np.linalg.norm(raw))
    fake_dirs = np.array(fake_dirs)

    # cos between pairs of fake directions
    pair_cosines = []
    for i in range(K-1):
        for j in range(i+1, min(i+10, K)):
            pair_cosines.append(float(fake_dirs[i] @ fake_dirs[j]))
    pair_cosines = np.array(pair_cosines)

    # real suffix-suffix cosines for comparison
    suffix_dirs = np.array([build_suffix_direction(sn, L) for sn in SUFFIX_NAMES])
    real_off = []
    for i in range(len(SUFFIX_NAMES)):
        for j in range(i+1, len(SUFFIX_NAMES)):
            real_off.append(float(suffix_dirs[i] @ suffix_dirs[j]))
    real_off = np.array(real_off)

    print(f"\nLayer {L}:")
    print(f"  null pair-cos:    mean={pair_cosines.mean():+.3f}  "
          f"95pct={np.percentile(pair_cosines, 95):+.3f}  "
          f"99pct={np.percentile(pair_cosines, 99):+.3f}")
    print(f"  real suffix-suffix off-diag: mean={real_off.mean():+.3f}  "
          f"max={real_off.max():+.3f}  min={real_off.min():+.3f}")
    print(f"  → suffixes cluster {'MORE' if real_off.mean() > pair_cosines.mean() else 'similar to'} random pairs")
    print(f"  → effect size (real_mean - null_mean): {real_off.mean() - pair_cosines.mean():+.3f}")
