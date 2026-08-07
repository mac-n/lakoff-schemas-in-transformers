"""
exp128_comparative_suffix_test.py — does the comparative/superlative
suffix itself project on the UP axis, or only when applied to words
with directional/magnitude/valence content?

Niamh's worry (2026-06-07): if -er and -est encode "more-ness" they might
intrinsically project on the UP axis (since UP encodes MORE-IS-UP in
Pythia). The clean test: take X / X-er / X-est triples across categories
and project on UP. Categories:

  DIRECTIONAL (predicted to project strongly with sign of root):
    high/higher/highest, low/lower/lowest, tall/taller/tallest,
    deep/deeper/deepest, shallow/shallower/shallowest,
    short/shorter/shortest

  MAGNITUDE (predicted to project + via MORE-IS-UP):
    big/bigger/biggest, small/smaller/smallest, large/larger/largest

  VALENCE (predicted to project via HAPPY-IS-UP):
    happy/happier/happiest, sad/sadder/saddest, good/better/best,
    bad/worse/worst

  CONTROL — as-neutral-as-possible (predicted to project ≈ 0 if
  suffix isn't intrinsically directional):
    red/redder/reddest, blue/bluer/bluest, green/greener/greenest,
    yellow/yellower/yellowest,
    round/rounder/roundest, square/squarer/squarest,
    smooth/smoother/smoothest, rough/rougher/roughest,
    old/older/oldest, new/newer/newest,
    loud/louder/loudest, quiet/quieter/quietest

  LAKOFF SCHEMA (already known to project — confirming our framework):
    hard/harder/hardest, soft/softer/softest (FORCE),
    light/lighter/lightest, dark/darker/darkest (LIGHT-DARK),
    warm/warmer/warmest, cold/colder/coldest (TEMPERATURE)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers

UP_WORDS = ["up", "rise", "rose", "rising", "ascend", "raise", "climb",
            "lift", "above", "over", "top", "high", "higher", "upward"]
DOWN_WORDS = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
              "below", "under", "bottom", "low", "lower", "downward"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]


TRIPLES = {
    "DIRECTIONAL": [
        ("high",    "higher",   "highest"),
        ("low",     "lower",    "lowest"),
        ("tall",    "taller",   "tallest"),
        ("deep",    "deeper",   "deepest"),
        ("shallow", "shallower","shallowest"),
        ("short",   "shorter",  "shortest"),
    ],
    "MAGNITUDE": [
        ("big",     "bigger",   "biggest"),
        ("small",   "smaller",  "smallest"),
        ("large",   "larger",   "largest"),
    ],
    "VALENCE": [
        ("happy",   "happier",  "happiest"),
        ("sad",     "sadder",   "saddest"),
        ("good",    "better",   "best"),
        ("bad",     "worse",    "worst"),
    ],
    "NEUTRAL_COLOR": [
        ("red",     "redder",   "reddest"),
        ("blue",    "bluer",    "bluest"),
        ("green",   "greener",  "greenest"),
        ("yellow",  "yellower", "yellowest"),
    ],
    "NEUTRAL_SHAPE": [
        ("round",   "rounder",  "roundest"),
        ("square",  "squarer",  "squarest"),
    ],
    "NEUTRAL_TEXTURE": [
        ("smooth",  "smoother", "smoothest"),
        ("rough",   "rougher",  "roughest"),
    ],
    "NEUTRAL_TIME": [
        ("old",     "older",    "oldest"),
        ("new",     "newer",    "newest"),
    ],
    "NEUTRAL_SOUND": [
        ("loud",    "louder",   "loudest"),
        ("quiet",   "quieter",  "quietest"),
    ],
    "LAKOFF_FORCE": [
        ("hard",    "harder",   "hardest"),
        ("soft",    "softer",   "softest"),
    ],
    "LAKOFF_LIGHT": [
        ("light",   "lighter",  "lightest"),
        ("dark",    "darker",   "darkest"),
    ],
    "LAKOFF_TEMP": [
        ("warm",    "warmer",   "warmest"),
        ("cold",    "colder",   "coldest"),
    ],
}

# Collect all test words
all_test_words = set()
for cat, triples in TRIPLES.items():
    for base, comp, sup in triples:
        all_test_words.add(base)
        all_test_words.add(comp)
        all_test_words.add(sup)
all_words = sorted(set(UP_WORDS + DOWN_WORDS + COMMON + RARE) | all_test_words)


# We only need L4 (peak directional in exp127) and L12 (mid-stable)
LAYERS_OF_INTEREST = [0, 4, 8, 12, 16, 20, 23]
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS_OF_INTEREST]

print(f"\nExtracting residuals at layers {LAYERS_OF_INTEREST} for {len(all_words)} words...")
residuals = {}
for i, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                    for L in LAYERS_OF_INTEREST}
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(all_words)}")


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)


def build_direction(pos, neg, layer):
    raw = mean_acts(pos, layer) - mean_acts(neg, layer)
    return raw / np.linalg.norm(raw)


def build_up_clean(layer):
    freq = build_direction(COMMON, RARE, layer)
    up_raw = build_direction(UP_WORDS, DOWN_WORDS, layer)
    clean = up_raw - (up_raw @ freq) * freq
    return clean / np.linalg.norm(clean)


def proj(word, layer, up_unit):
    v = residuals[word][layer]
    return float(v @ up_unit / np.linalg.norm(v))


# ============================================================================
# Compute and report per layer
# ============================================================================

results = {L: {} for L in LAYERS_OF_INTEREST}
for L in LAYERS_OF_INTEREST:
    up_unit = build_up_clean(L)
    for cat, triples in TRIPLES.items():
        results[L][cat] = []
        for base, comp, sup in triples:
            results[L][cat].append({
                "base": (base, proj(base, L, up_unit)),
                "comp": (comp, proj(comp, L, up_unit)),
                "sup":  (sup,  proj(sup, L, up_unit)),
            })


# Print table at L4 (peak directional)
for L_of_interest in [4, 12]:
    print("\n" + "=" * 78)
    print(f"PROJECTIONS at L{L_of_interest} (freq-stripped UP)")
    print("=" * 78)
    for cat, triples in TRIPLES.items():
        print(f"\n  --- {cat} ---")
        print(f"  {'base':>14}        {'comp':>14}        {'sup':>14}")
        for r in results[L_of_interest][cat]:
            base_w, base_p = r["base"]
            comp_w, comp_p = r["comp"]
            sup_w, sup_p = r["sup"]
            print(f"  {base_w:>10}={base_p:+.3f}  "
                  f"{comp_w:>10}={comp_p:+.3f}  "
                  f"{sup_w:>10}={sup_p:+.3f}")


# ============================================================================
# Summary: mean absolute projection per category per layer
# ============================================================================

print("\n" + "=" * 78)
print("MEAN |PROJECTION| per category per layer")
print("=" * 78)
print(f"\n  {'category':<18}  " + "  ".join(f"L{L:>2}" for L in LAYERS_OF_INTEREST))
for cat in TRIPLES:
    means = []
    for L in LAYERS_OF_INTEREST:
        vals = []
        for r in results[L][cat]:
            vals.extend([r["base"][1], r["comp"][1], r["sup"][1]])
        means.append(np.mean(np.abs(vals)))
    row = f"  {cat:<18}  "
    for m in means:
        row += f"{m:.3f}  "
    print(row)


# ============================================================================
# Comp - Base difference (does -er amplify the root's directional content?)
# ============================================================================

print("\n" + "=" * 78)
print("(comp − base) at L=4 by category — does -er amplify root's direction?")
print("=" * 78)
for cat, triples in TRIPLES.items():
    diffs = []
    for r in results[4][cat]:
        diffs.append(r["comp"][1] - r["base"][1])
    print(f"  {cat:<18}  mean(comp-base) = {np.mean(diffs):+.4f}  "
          f"std = {np.std(diffs):.4f}  "
          f"items: {[f'{r[1]:+.2f}' for r in [(r['comp'][0], r['comp'][1]-r['base'][1]) for r in results[4][cat]]]}")


# ============================================================================
# Plot
# ============================================================================

fig, axes = plt.subplots(2, 6, figsize=(24, 9), sharey=True)
axes = axes.flatten()

for ax, (cat, triples) in zip(axes, TRIPLES.items()):
    for r in results[4][cat]:
        ys = [r["base"][1], r["comp"][1], r["sup"][1]]
        xs = ["base", "comp", "sup"]
        ax.plot(xs, ys, "-o", markersize=6, label=r["base"][0])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(cat, fontsize=11)
    ax.set_ylabel("projection on UP @ L4")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

fig.suptitle("exp128 — comparative/superlative projection on UP at L4 of Pythia 410M\n"
             "Are 'neutral' comparatives near zero? Or does the suffix project on its own?",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp128_comparatives.png", dpi=120)
print("\nSaved exp128_comparatives.png")

np.savez("/Users/macn/Documents/embeddingexp/exp128_results.npz",
         layers=np.array(LAYERS_OF_INTEREST),
         # serialise into flat arrays
         words=np.array(list(all_test_words)))
print("Saved exp128_results.npz")
