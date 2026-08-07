"""
exp126_glove_more_vs_up.py — does GloVe show magnitude-dominance or
directional-dominance on UP-DOWN?

In Pythia 410M L12 we found (exp116) that pool depth co-shifts UP with
height under cleaned UP-steering — magnitude-dominance, not bipolar
verticality. Niamh's question (2026-06-07): is this a property of
*distributional models generally* or specific to deep-layer transformer
abstraction?

GloVe test: static distributional embeddings, no layers, no steering —
just project test words onto the UP-DOWN direction and see whether
"deep" and "shallow" land on the magnitude-prediction side or the
directional-prediction side.

Discriminating predictions:
                       magnitude    directional
  tall                 +            +              (not discriminating)
  short                -            -              (not discriminating)
  deep                 +            -              ← DISCRIMINATOR
  shallow              -            +              ← DISCRIMINATOR
  huge                 +            +              (MORE-IS-UP entanglement)
  tiny                 -            -              (not discriminating)
  wide                 +            ~0             (partial discriminator)
  narrow               -            ~0             (partial discriminator)

We also report a freq-stripped variant, since exp99-103 showed GloVe
contrast vectors are frequency-dominated and stripping is necessary.
"""

import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300 (cached after first run)...")
wv = api.load("glove-wiki-gigaword-300")
print(f"  loaded: {len(wv.key_to_index)} words, dim={wv.vector_size}")


# ============================================================================
# Anchors — Lakoff MML literal-motion sublist (same as v3 exp112)
# ============================================================================

UP_WORDS = ["up", "rise", "rose", "rising", "ascend", "raise", "climb",
            "lift", "above", "over", "top", "high", "higher", "upward"]
DOWN_WORDS = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
              "below", "under", "bottom", "low", "lower", "downward"]


def get(word):
    try:
        return wv[word]
    except KeyError:
        return None


def build_direction(pos, neg):
    pos_vecs = [get(w) for w in pos]
    pos_vecs = [v for v in pos_vecs if v is not None]
    neg_vecs = [get(w) for w in neg]
    neg_vecs = [v for v in neg_vecs if v is not None]
    pmean = np.mean(pos_vecs, axis=0)
    nmean = np.mean(neg_vecs, axis=0)
    raw = pmean - nmean
    return raw / np.linalg.norm(raw), pmean, nmean


up_dir, _, _ = build_direction(UP_WORDS, DOWN_WORDS)
print(f"\nUP direction built from {len(UP_WORDS)} UP + {len(DOWN_WORDS)} DOWN anchors")


# ============================================================================
# Frequency axis (same recipe as v3)
# ============================================================================

COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]

freq_dir, _, _ = build_direction(COMMON, RARE)
cos_up_freq = float(up_dir @ freq_dir)
print(f"cos(up_dir, freq_dir) = {cos_up_freq:+.3f}")


# Stripped variant: orthogonalise against freq, renormalise
up_dir_clean = up_dir - (up_dir @ freq_dir) * freq_dir
up_dir_clean = up_dir_clean / np.linalg.norm(up_dir_clean)
cos_clean_raw = float(up_dir_clean @ up_dir)
print(f"cos(up_dir_clean, up_dir) = {cos_clean_raw:+.3f}")


# ============================================================================
# Test words and their predictions
# ============================================================================

TEST_WORDS = [
    # Clear UP (both theories predict +)
    ("tall",            "+", "+"),
    ("high",            "+", "+"),
    ("top",             "+", "+"),
    ("towering",        "+", "+"),
    ("ceiling",         "+", "+"),
    # Clear DOWN (both theories predict -)
    ("short",           "-", "-"),
    ("low",             "-", "-"),
    ("bottom",          "-", "-"),
    ("ground",          "-", "-"),
    ("floor",           "-", "-"),
    # CRITICAL DISCRIMINATORS — depth (downward extent)
    ("deep",            "+", "-"),  # magnitude vs directional
    ("deeper",          "+", "-"),
    ("deepest",         "+", "-"),
    ("abyss",           "+", "-"),
    ("shallow",         "-", "+"),
    # Horizontal-extent (partial discriminators)
    ("wide",            "+", "0"),
    ("narrow",          "-", "0"),
    ("broad",           "+", "0"),
    ("thin",            "-", "0"),
    # Pure magnitude (no spatial direction)
    ("huge",            "+", "+"),  # entangled with MORE-IS-UP regardless
    ("massive",         "+", "+"),
    ("enormous",        "+", "+"),
    ("giant",           "+", "+"),
    ("tiny",            "-", "-"),
    ("small",           "-", "-"),
    # Concept words to see how STATUS / VALENCE / MORE land
    ("status",          "+", "?"),  # high-status is up
    ("power",           "+", "?"),
    ("important",       "+", "?"),
    ("happy",           "+", "?"),  # happy-is-up
    ("sad",             "-", "?"),
    ("more",            "+", "?"),  # more-is-up canonically
    ("less",            "-", "?"),
    # Downward objects
    ("underground",     "-", "-"),
    ("basement",        "-", "-"),
    ("pit",             "-", "-"),
    ("well",            "-", "-"),  # ambiguous (well-being vs water-well)
    # Upward objects
    ("sky",             "+", "+"),
    ("mountain",        "+", "+"),
    ("skyscraper",      "+", "+"),
    # Negative valence (Pollyanna check)
    ("evil",            "?", "?"),
    ("good",            "?", "?"),
]


def proj(word, direction):
    v = get(word)
    if v is None:
        return None
    vn = v / np.linalg.norm(v)
    return float(vn @ direction)


print("\n" + "=" * 76)
print("PROJECTIONS onto UP direction (raw, no freq strip)")
print("=" * 76)
print(f"\n  {'word':<15}  {'mag pred':>8}  {'dir pred':>8}  {'proj raw':>9}  "
      f"{'proj clean':>11}  matches")
for w, mag_pred, dir_pred in TEST_WORDS:
    pr = proj(w, up_dir)
    pc = proj(w, up_dir_clean)
    if pr is None:
        print(f"  {w:<15}  (not in vocab)")
        continue
    # Determine which prediction the projection matches
    sign_raw = "+" if pr > 0.02 else ("-" if pr < -0.02 else "0")
    sign_clean = "+" if pc > 0.02 else ("-" if pc < -0.02 else "0")
    if mag_pred == dir_pred:
        verdict = "agree" if sign_clean == mag_pred else "noise"
    else:
        if sign_clean == mag_pred:
            verdict = "MAGNITUDE"
        elif sign_clean == dir_pred:
            verdict = "DIRECTIONAL"
        else:
            verdict = "?"
    print(f"  {w:<15}  {mag_pred:>8}  {dir_pred:>8}  {pr:>+9.3f}  {pc:>+11.3f}  {verdict}")


# ============================================================================
# Summary on the CRITICAL discriminators
# ============================================================================

print("\n" + "=" * 76)
print("CRITICAL DISCRIMINATORS — depth and width")
print("=" * 76)

CRITICAL_DEPTH = ["deep", "deeper", "deepest", "abyss", "shallow"]
CRITICAL_WIDTH = ["wide", "narrow", "broad", "thin"]

print(f"\n  depth/shallow predictions:")
print(f"    magnitude theory: deep words project POSITIVE")
print(f"    directional theory: deep words project NEGATIVE")
print(f"\n  observed (freq-stripped):")
for w in CRITICAL_DEPTH:
    p = proj(w, up_dir_clean)
    if p is not None:
        verdict = "→ MAGNITUDE" if (p > 0 and w != "shallow") or (p < 0 and w == "shallow") else "→ DIRECTIONAL"
        print(f"    {w:<15}: {p:+.3f}  {verdict}")

print(f"\n  width predictions:")
print(f"    magnitude theory: wide > 0, narrow < 0")
print(f"    directional theory: wide ≈ 0, narrow ≈ 0 (horizontal)")
print(f"\n  observed (freq-stripped):")
for w in CRITICAL_WIDTH:
    p = proj(w, up_dir_clean)
    if p is not None:
        print(f"    {w:<15}: {p:+.3f}")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp126_results.npz",
         test_words=[w for w, _, _ in TEST_WORDS],
         projections_raw=[proj(w, up_dir) for w, _, _ in TEST_WORDS],
         projections_clean=[proj(w, up_dir_clean) for w, _, _ in TEST_WORDS],
         cos_up_freq=cos_up_freq)
print("\nSaved exp126_results.npz")
