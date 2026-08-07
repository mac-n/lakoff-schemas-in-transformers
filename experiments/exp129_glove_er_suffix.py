"""
exp129_glove_er_suffix.py — does the -ER suffix project on UP in GloVe?

If yes: morphology-in-semantic-geometry is general to distributional
learning (concepts and grammatical operators both organised on the same
semantic axes).

If no: transformer-specific. Pythia integrates grammatical operators
with semantic geometry in a way static distributional embeddings don't.

Method:
1. Collect base/comparative pairs across categories (directional,
   magnitude, neutral, valence)
2. Build -ER direction = mean(comparatives) - mean(bases), normalised
3. Project onto UP direction (from Lakoff MML literal-motion anchors)
4. Report: cos(-ER direction, UP direction)

If |cos| is substantial (say > 0.2), -ER is geometrically located in the
UP-DOWN subspace in GloVe too. If near zero, -ER lives elsewhere and
Pythia did something specific.

We also check the per-pair shifts to compare to exp128 Pythia.
"""

import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")
print(f"  loaded: {len(wv.key_to_index)} words")


# Anchors for UP direction
UP_WORDS = ["up", "rise", "rose", "rising", "ascend", "raise", "climb",
            "lift", "above", "over", "top", "high", "higher", "upward"]
DOWN_WORDS = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
              "below", "under", "bottom", "low", "lower", "downward"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]


# Base / comparative pairs (regular -er only; skip irregulars like good/better)
ER_PAIRS = {
    "DIRECTIONAL": [
        ("high",    "higher"),
        ("low",     "lower"),
        ("tall",    "taller"),
        ("deep",    "deeper"),
        ("shallow", "shallower"),
        ("short",   "shorter"),
        ("wide",    "wider"),
        ("narrow",  "narrower"),
    ],
    "MAGNITUDE": [
        ("big",     "bigger"),
        ("small",   "smaller"),
        ("large",   "larger"),
        ("thick",   "thicker"),
        ("thin",    "thinner"),
    ],
    "VALENCE": [
        ("happy",   "happier"),
        ("sad",     "sadder"),
        ("kind",    "kinder"),
        ("mean",    "meaner"),
    ],
    "NEUTRAL_COLOR": [
        ("red",     "redder"),
        ("blue",    "bluer"),
        ("green",   "greener"),
        ("yellow",  "yellower"),
    ],
    "NEUTRAL_SHAPE": [
        ("round",   "rounder"),
    ],
    "NEUTRAL_TEXTURE": [
        ("smooth",  "smoother"),
        ("rough",   "rougher"),
        ("soft",    "softer"),
        ("hard",    "harder"),
    ],
    "NEUTRAL_TIME": [
        ("old",     "older"),
        ("new",     "newer"),
        ("young",   "younger"),
    ],
    "NEUTRAL_SOUND": [
        ("loud",    "louder"),
        ("quiet",   "quieter"),
    ],
    "TEMPERATURE": [
        ("hot",     "hotter"),
        ("cold",    "colder"),
        ("warm",    "warmer"),
        ("cool",    "cooler"),
    ],
    "SPEED": [
        ("fast",    "faster"),
        ("slow",    "slower"),
        ("quick",   "quicker"),
    ],
    "LIGHTNESS": [
        ("light",   "lighter"),
        ("dark",    "darker"),
        ("bright",  "brighter"),
        ("dim",     "dimmer"),
    ],
}

def get(w):
    try:
        return wv[w]
    except KeyError:
        return None

def mean_unit(words):
    vecs = [get(w) for w in words if get(w) is not None]
    if not vecs:
        return None
    m = np.mean(vecs, axis=0)
    return m / np.linalg.norm(m)


# Build UP and frequency axes
up_raw = mean_unit(UP_WORDS) - mean_unit(DOWN_WORDS)
up_dir = up_raw / np.linalg.norm(up_raw)

freq_raw = mean_unit(COMMON) - mean_unit(RARE)
freq_dir = freq_raw / np.linalg.norm(freq_raw)

up_clean = up_dir - (up_dir @ freq_dir) * freq_dir
up_clean = up_clean / np.linalg.norm(up_clean)

print(f"\ncos(up_dir, freq_dir) = {float(up_dir @ freq_dir):+.4f}  (low → freq strip is small)")


# ============================================================================
# Per-pair: compute (comp - base) as a difference vector and project on UP
# ============================================================================

print("\n" + "=" * 78)
print("Per-pair (comp − base) projection on UP_clean")
print("=" * 78)

all_diff_vectors = []
all_pair_projections_on_up = []

for cat, pairs in ER_PAIRS.items():
    print(f"\n  --- {cat} ---")
    cat_diffs = []
    cat_projs = []
    for base, comp in pairs:
        bv = get(base)
        cv = get(comp)
        if bv is None or cv is None:
            print(f"    {base}/{comp}: MISSING")
            continue
        # Pair difference
        bu = bv / np.linalg.norm(bv)
        cu = cv / np.linalg.norm(cv)
        diff = cu - bu
        diff_unit = diff / np.linalg.norm(diff) if np.linalg.norm(diff) > 1e-9 else diff
        proj_diff = float(diff_unit @ up_clean)
        # Also: projections of base and comp individually on UP
        proj_base = float(bu @ up_clean)
        proj_comp = float(cu @ up_clean)
        delta = proj_comp - proj_base
        cat_diffs.append(diff)
        cat_projs.append(delta)
        all_diff_vectors.append(diff)
        all_pair_projections_on_up.append(delta)
        print(f"    {base:>10}={proj_base:+.3f}  {comp:>11}={proj_comp:+.3f}  "
              f"Δ(comp-base)={delta:+.3f}  cos(diff,UP)={proj_diff:+.3f}")
    if cat_projs:
        print(f"    mean Δ(comp-base) on UP: {np.mean(cat_projs):+.4f}  "
              f"(std {np.std(cat_projs):.4f})")


# ============================================================================
# The big question: build an aggregate -ER direction, project on UP
# ============================================================================

print("\n" + "=" * 78)
print("AGGREGATE -ER DIRECTION  (mean of all comp - base difference vectors)")
print("=" * 78)

er_raw = np.mean(all_diff_vectors, axis=0)
er_dir = er_raw / np.linalg.norm(er_raw)

cos_er_up = float(er_dir @ up_clean)
cos_er_up_raw = float(er_dir @ up_dir)
cos_er_freq = float(er_dir @ freq_dir)

print(f"\n  cos(-ER direction, UP_clean)   = {cos_er_up:+.4f}")
print(f"  cos(-ER direction, UP_raw)     = {cos_er_up_raw:+.4f}")
print(f"  cos(-ER direction, freq axis)  = {cos_er_freq:+.4f}")
print(f"\n  Mean of pair Δ(comp-base) on UP_clean: "
      f"{np.mean(all_pair_projections_on_up):+.4f}")
print(f"  Std: {np.std(all_pair_projections_on_up):.4f}")
print(f"  Sign-consistent across pairs: "
      f"{(np.array(all_pair_projections_on_up) > 0).sum()}/{len(all_pair_projections_on_up)} positive, "
      f"{(np.array(all_pair_projections_on_up) < 0).sum()}/{len(all_pair_projections_on_up)} negative")


# ============================================================================
# Comparison to exp128 Pythia at L4
# ============================================================================

print("\n" + "=" * 78)
print("Comparison to exp128 Pythia 410M L4 (comp − base on UP)")
print("=" * 78)

PYTHIA_L4_COMP_BASE = {
    "DIRECTIONAL":      -0.04,
    "MAGNITUDE":        -0.11,
    "VALENCE":          -0.04,
    "NEUTRAL_COLOR":    -0.08,
    "NEUTRAL_SHAPE":    -0.11,
    "NEUTRAL_TEXTURE":  -0.09,
    "NEUTRAL_TIME":     -0.05,
    "NEUTRAL_SOUND":    -0.07,
    "TEMPERATURE":      -0.17,  # was LAKOFF_TEMP
    "LIGHTNESS":        -0.13,  # was LAKOFF_LIGHT
}

print(f"\n  {'category':<18}  {'GloVe mean Δ':>14}  {'Pythia L4 mean Δ':>20}")
for cat, pairs in ER_PAIRS.items():
    glove_diffs = []
    for base, comp in pairs:
        bv = get(base); cv = get(comp)
        if bv is None or cv is None:
            continue
        bu = bv / np.linalg.norm(bv)
        cu = cv / np.linalg.norm(cv)
        glove_diffs.append(float(cu @ up_clean) - float(bu @ up_clean))
    if not glove_diffs:
        continue
    glove_mean = np.mean(glove_diffs)
    pythia_mean = PYTHIA_L4_COMP_BASE.get(cat, None)
    pythia_str = f"{pythia_mean:+.4f}" if pythia_mean is not None else "n/a"
    print(f"  {cat:<18}  {glove_mean:>+14.4f}  {pythia_str:>20}")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp129_results.npz",
         er_direction=er_dir,
         up_clean=up_clean,
         cos_er_up=cos_er_up,
         cos_er_freq=cos_er_freq,
         pair_projections_on_up=np.array(all_pair_projections_on_up))
print("\nSaved exp129_results.npz")
