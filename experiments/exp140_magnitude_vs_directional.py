"""
exp140_magnitude_vs_directional.py — does MORE-IS-UP dominate UP-IS-UP
in Pythia 410M because vectors have magnitude as substrate-intrinsic
primitive but not direction?

Build two cleanly-distinct axes:
- MAGNITUDE: pure size words (huge/tiny etc.) no directional content
- DIRECTIONAL: pure positional words (above/below etc.) no magnitude content

Then test:
1. cos(MAGNITUDE, DIRECTIONAL) — orthogonal or conflated?
2. cos with anisotropy before strip — which is more substrate-aligned?
3. Cross-layer stability — which is more layer-stable (substrate-rooted)?
4. Lakoff UP-DOWN aligns with which?
5. Test words: project deep/shallow/tall etc on both, see which dominates
6. Compare to GloVe

Substrate-primitive prediction: MAGNITUDE more anisotropy-aligned,
more layer-stable, more dominant for UP-DOWN representation.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer
import gensim.downloader as api

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers
LAYERS = list(range(N_LAYERS))
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]


# ============================================================================
# Axis word lists — chosen for purity
# ============================================================================

# MAGNITUDE: pure size/extent, no directional content
MAGNITUDE_BIG = ["huge", "big", "large", "enormous", "vast", "massive",
                 "gigantic", "great"]
MAGNITUDE_SMALL = ["tiny", "small", "little", "minute", "microscopic",
                   "miniature"]

# DIRECTIONAL: pure positional, no magnitude content
# Use prepositions and adverbs that mark spatial position only
DIRECTIONAL_UP = ["above", "over", "atop", "upward", "overhead", "upper"]
DIRECTIONAL_DOWN = ["below", "under", "underneath", "downward", "beneath",
                    "lower"]

# Standard Lakoff UP-DOWN for reference
LAKOFF_UP = ["up", "rise", "rose", "rising", "ascend", "raise", "climb",
             "lift", "above", "over", "top", "high", "higher", "upward"]
LAKOFF_DOWN = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
               "below", "under", "bottom", "low", "lower", "downward"]

# Test words — discriminating cases
TEST_WORDS = {
    "depth_extent": ["deep", "deeper", "deepest", "abyss"],
    "shallow": ["shallow"],
    "height_extent": ["tall", "taller", "tallest"],
    "magnitude_pure": ["huge", "massive", "enormous", "tiny", "small"],
    "horizontal_extent": ["wide", "narrow", "broad", "thin"],
    "metaphorical_up": ["status", "power", "important", "happy"],
    "metaphorical_down": ["sad", "low", "depressed", "humble"],
}

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]

# Collect all words
all_words = set(COMMON + RARE)
for words in [MAGNITUDE_BIG, MAGNITUDE_SMALL, DIRECTIONAL_UP, DIRECTIONAL_DOWN,
              LAKOFF_UP, LAKOFF_DOWN]:
    all_words.update(words)
for ws in TEST_WORDS.values():
    all_words.update(ws)
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
    if (k+1) % 30 == 0:
        print(f"  {k+1}/{len(all_words)}")

# Anisotropy per layer
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


def build_axis_raw(pos, neg, layer):
    raw = mean_acts(pos, layer) - mean_acts(neg, layer)
    return raw / np.linalg.norm(raw)


def build_axis_clean(pos, neg, layer):
    raw = build_axis_raw(pos, neg, layer)
    return strip_aniso_freq(raw, layer)


# ============================================================================
# TEST 1: cos(MAGNITUDE, DIRECTIONAL) — orthogonal or conflated?
# ============================================================================

print("\n" + "=" * 78)
print("TEST 1 — cos(MAGNITUDE, DIRECTIONAL) per layer")
print("=" * 78)
print(f"\n  layer    raw cos     clean cos")
mag_dirs_raw = []
dir_dirs_raw = []
mag_dirs_clean = []
dir_dirs_clean = []
for L in LAYERS:
    m_raw = build_axis_raw(MAGNITUDE_BIG, MAGNITUDE_SMALL, L)
    d_raw = build_axis_raw(DIRECTIONAL_UP, DIRECTIONAL_DOWN, L)
    m_cln = build_axis_clean(MAGNITUDE_BIG, MAGNITUDE_SMALL, L)
    d_cln = build_axis_clean(DIRECTIONAL_UP, DIRECTIONAL_DOWN, L)
    mag_dirs_raw.append(m_raw); dir_dirs_raw.append(d_raw)
    mag_dirs_clean.append(m_cln); dir_dirs_clean.append(d_cln)
    if L in [0, 4, 8, 12, 16, 20, 23]:
        print(f"  L{L:>3}    {float(m_raw @ d_raw):>+8.3f}    {float(m_cln @ d_cln):>+8.3f}")


# ============================================================================
# TEST 2: cos with anisotropy
# ============================================================================

print("\n" + "=" * 78)
print("TEST 2 — cos with anisotropy (before strip)")
print("=" * 78)
print(f"\n  layer    MAG vs aniso    DIR vs aniso    LAKOFF vs aniso")
lakoff_dirs_clean = []
for L in LAYERS:
    aniso = anisotropy_dirs[L]
    m_raw = mag_dirs_raw[L]
    d_raw = dir_dirs_raw[L]
    lkf_raw = build_axis_raw(LAKOFF_UP, LAKOFF_DOWN, L)
    lakoff_dirs_clean.append(build_axis_clean(LAKOFF_UP, LAKOFF_DOWN, L))
    if L in [0, 4, 8, 12, 16, 20, 23]:
        print(f"  L{L:>3}    {float(m_raw @ aniso):>+12.3f}    "
              f"{float(d_raw @ aniso):>+12.3f}    {float(lkf_raw @ aniso):>+15.3f}")


# ============================================================================
# TEST 3: cross-layer stability of each axis (after stripping)
# ============================================================================

WORK = list(range(4, 23))
def cross_layer_mean(dirs, work):
    arr = np.array([dirs[L] for L in work])
    cm = arr @ arr.T
    return cm[~np.eye(len(work), dtype=bool)].mean()


m_clean_stab = cross_layer_mean(mag_dirs_clean, WORK)
d_clean_stab = cross_layer_mean(dir_dirs_clean, WORK)
l_clean_stab = cross_layer_mean(lakoff_dirs_clean, WORK)

print("\n" + "=" * 78)
print("TEST 3 — cross-layer stability of each axis (after anisotropy strip, L4-L22)")
print("=" * 78)
print(f"\n  MAGNITUDE axis:    mean cross-layer cos = {m_clean_stab:+.4f}")
print(f"  DIRECTIONAL axis:  mean cross-layer cos = {d_clean_stab:+.4f}")
print(f"  LAKOFF UP-DOWN:    mean cross-layer cos = {l_clean_stab:+.4f}")


# ============================================================================
# TEST 4: Lakoff UP-DOWN aligns with which? (after strip)
# ============================================================================

print("\n" + "=" * 78)
print("TEST 4 — cos(LAKOFF, MAG) vs cos(LAKOFF, DIR) at each layer (clean)")
print("=" * 78)
print(f"\n  layer    LAKOFF · MAG    LAKOFF · DIR    diff (mag−dir)")
for L in LAYERS:
    if L not in [0, 4, 8, 12, 16, 20, 23]:
        continue
    c_lm = float(lakoff_dirs_clean[L] @ mag_dirs_clean[L])
    c_ld = float(lakoff_dirs_clean[L] @ dir_dirs_clean[L])
    print(f"  L{L:>3}    {c_lm:>+12.3f}    {c_ld:>+12.3f}    {c_lm - c_ld:>+13.3f}")


# ============================================================================
# TEST 5: test words on MAGNITUDE vs DIRECTIONAL (L12 only for clarity)
# ============================================================================

print("\n" + "=" * 78)
print("TEST 5 — test word projections on MAG vs DIR (L12, clean)")
print("=" * 78)
L = 12
m = mag_dirs_clean[L]
d = dir_dirs_clean[L]
print(f"\n  {'word':<14}  {'on MAG':>10}  {'on DIR':>10}  "
      f"{'|mag|>|dir|?':>14}")
for category, words in TEST_WORDS.items():
    print(f"\n  --- {category} ---")
    for w in words:
        rw = residuals[w][L]; rw_unit = rw / np.linalg.norm(rw)
        pm = float(rw_unit @ m); pd = float(rw_unit @ d)
        winner = "MAG" if abs(pm) > abs(pd) else "DIR"
        print(f"  {w:<14}  {pm:>+10.3f}  {pd:>+10.3f}  {winner:>14}")


# ============================================================================
# TEST 6: GloVe comparison
# ============================================================================

print("\n" + "=" * 78)
print("TEST 6 — GloVe comparison")
print("=" * 78)
print("\nLoading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
print(f"  loaded: {len(wv.key_to_index)} words")


def glove_axis(pos, neg):
    p = [wv[w] for w in pos if w in wv.key_to_index]
    n = [wv[w] for w in neg if w in wv.key_to_index]
    raw = np.mean(p, axis=0) - np.mean(n, axis=0)
    return raw / np.linalg.norm(raw)


g_mag = glove_axis(MAGNITUDE_BIG, MAGNITUDE_SMALL)
g_dir = glove_axis(DIRECTIONAL_UP, DIRECTIONAL_DOWN)
g_lakoff = glove_axis(LAKOFF_UP, LAKOFF_DOWN)

print(f"\n  GloVe cos(MAGNITUDE, DIRECTIONAL): {float(g_mag @ g_dir):+.3f}")
print(f"  GloVe cos(LAKOFF, MAGNITUDE):      {float(g_lakoff @ g_mag):+.3f}")
print(f"  GloVe cos(LAKOFF, DIRECTIONAL):    {float(g_lakoff @ g_dir):+.3f}")

print(f"\n  GloVe test word projections:")
print(f"  {'word':<14}  {'on MAG':>10}  {'on DIR':>10}")
for category, words in TEST_WORDS.items():
    print(f"\n  --- {category} ---")
    for w in words:
        if w not in wv.key_to_index:
            continue
        v = wv[w] / np.linalg.norm(wv[w])
        pm = float(v @ g_mag); pd = float(v @ g_dir)
        print(f"  {w:<14}  {pm:>+10.3f}  {pd:>+10.3f}")
