"""
exp127_pythia_depth_trajectory.py — does the literal-direction encoding
of "deep" decay across Pythia 410M layers?

Per Niamh's substrate-primitive framing (2026-06-07): vectors have
magnitude as substrate-intrinsic primitive (every vector has a norm by
mathematical necessity); direction is only ever a learned secondary
feature. As Pythia's layers process, the substrate-intrinsic primitive
should progressively reassert itself, while learned directional encoding
fades.

Experiment: for each of layers 0-23 in Pythia 410M, build a per-layer
UP direction (Lakoff MML literal-motion anchors), freq-strip it, then
project test words onto it. Track:

  - depth words (deep/deeper/deepest/abyss): predicted to project
    negative in early layers (directional) and to drift toward 0 or
    positive at later layers (substrate-intrinsic magnitude wins)
  - height words (tall/towering): predicted to stay positive throughout
  - magnitude words (huge/massive): predicted to stay positive
    throughout (MORE-IS-UP entanglement) — but possibly to grow more
    positive as the magnitude-primitive asserts itself
  - status/valence words: positive throughout, possibly growing
  - width words (wide/narrow): orthogonal in early layers (~0), maybe
    becoming positive at later layers if "wide = more extent"

Killer figure: layer on x-axis, projection on y-axis, multiple
test-word curves.

GloVe baseline from exp126:
  deep -0.144, deeper -0.218, deepest -0.141, abyss -0.264
  status +0.154, power +0.107, more +0.180, huge +0.096

If Pythia early layers look like GloVe and late layers look like exp116
(magnitude-dominated), we've found the transition.
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
D_MODEL = model.cfg.d_model
print(f"  {N_LAYERS} layers, d_model={D_MODEL}")


# ============================================================================
# Anchors — Lakoff MML literal-motion (same as exp123, exp126)
# ============================================================================

UP_WORDS = ["up", "rise", "rose", "rising", "ascend", "raise", "climb",
            "lift", "above", "over", "top", "high", "higher", "upward"]
DOWN_WORDS = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
              "below", "under", "bottom", "low", "lower", "downward"]

COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]


# ============================================================================
# Test words to track across layers
# ============================================================================

TEST_WORDS = {
    # Critical discriminators — depth (downward extent)
    "deep":     "depth",
    "deeper":   "depth",
    "deepest":  "depth",
    "abyss":    "depth",
    "shallow":  "depth",
    # Height (upward extent) — should stay + throughout
    "tall":      "height",
    "towering":  "height",
    "high":      "height",
    "ceiling":   "height",
    # Pure magnitude — should grow + with substrate-primitive
    "huge":      "magnitude",
    "massive":   "magnitude",
    "enormous":  "magnitude",
    "tiny":      "magnitude",
    "small":     "magnitude",
    # Status / valence — should grow + via metaphorical extensions
    "status":    "status",
    "power":     "status",
    "important": "status",
    "happy":     "valence",
    "sad":       "valence",
    "more":      "more",
    "less":      "more",
    # Horizontal extent — directional theory says ~0
    "wide":      "horizontal",
    "narrow":    "horizontal",
    "broad":     "horizontal",
    # Downward objects
    "underground": "downward",
    "basement":    "downward",
    "pit":         "downward",
}


# ============================================================================
# Extract residuals for ALL words at ALL layers (cached once)
# ============================================================================

all_words = sorted(set(
    UP_WORDS + DOWN_WORDS + COMMON + RARE + list(TEST_WORDS.keys())
))
hook_names = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]

print(f"\nExtracting residuals at all {N_LAYERS} layers for {len(all_words)} words...")
residuals = {}  # word -> [N_LAYERS, D_MODEL]
for i, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    per_layer = torch.stack(
        [cache[hn][0, -1, :].cpu() for hn in hook_names], dim=0
    )
    residuals[w] = per_layer.numpy()
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(all_words)}")


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)


def build_direction(up_words, down_words, layer):
    u = mean_acts(up_words, layer)
    d = mean_acts(down_words, layer)
    raw = u - d
    n = np.linalg.norm(raw)
    if n < 1e-9:
        return None
    return raw / n


# ============================================================================
# Per-layer: build UP direction, freq-strip, project test words
# ============================================================================

print("\nBuilding per-layer UP directions and projecting test words...")
trajectories = {w: [] for w in TEST_WORDS}
up_unit_per_layer = []
cos_up_freq_per_layer = []

for L in range(N_LAYERS):
    freq_dir = build_direction(COMMON, RARE, L)
    up_raw = build_direction(UP_WORDS, DOWN_WORDS, L)
    cos_up_freq_per_layer.append(float(up_raw @ freq_dir))
    # Freq-strip
    up_clean = up_raw - (up_raw @ freq_dir) * freq_dir
    up_clean = up_clean / np.linalg.norm(up_clean)
    up_unit_per_layer.append(up_clean)
    # Project test words
    for w in TEST_WORDS:
        r = residuals[w][L]
        rn = r / np.linalg.norm(r)
        proj = float(rn @ up_clean)
        trajectories[w].append(proj)

for w in trajectories:
    trajectories[w] = np.array(trajectories[w])


# ============================================================================
# Print key table
# ============================================================================

print("\n" + "=" * 80)
print("DEPTH WORDS (critical discriminators) — projection on UP per layer")
print("=" * 80)
print(f"  {'word':<10}  " + "  ".join(f"L{L:>2}" for L in range(0, N_LAYERS, 2)))
for w in ["deep", "deeper", "deepest", "abyss", "shallow"]:
    row = f"  {w:<10}  "
    for L in range(0, N_LAYERS, 2):
        row += f"{trajectories[w][L]:+.2f}"
        row += "  "
    print(row)

print("\nGloVe baseline for these (from exp126):")
print("  deep -0.144, deeper -0.218, deepest -0.141, abyss -0.264, shallow -0.268")


# ============================================================================
# Plot trajectories
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()

# GloVe baselines (from exp126 results)
GLOVE_BASELINES = {
    "deep": -0.144, "deeper": -0.218, "deepest": -0.141, "abyss": -0.264,
    "shallow": -0.268,
    "tall": +0.205, "towering": +0.188, "high": +0.249, "ceiling": -0.063,
    "huge": +0.096, "massive": +0.044, "enormous": +0.150, "tiny": -0.072,
    "small": -0.027,
    "status": +0.154, "power": +0.107, "important": +0.097,
    "happy": +0.026, "sad": -0.084,
    "more": +0.180, "less": +0.089,
    "wide": -0.008, "narrow": -0.097, "broad": -0.006,
    "underground": -0.042, "basement": -0.136, "pit": -0.170,
}

groups = {
    "DEPTH (downward extent)":   ["deep", "deeper", "deepest", "abyss", "shallow"],
    "HEIGHT (upward extent)":    ["tall", "towering", "high", "ceiling"],
    "MAGNITUDE (no direction)":  ["huge", "massive", "enormous", "tiny", "small"],
    "STATUS / VALENCE":          ["status", "power", "important", "happy", "sad"],
    "MORE / LESS":               ["more", "less"],
    "HORIZONTAL (wide/narrow)":  ["wide", "narrow", "broad"],
}

xs = np.arange(N_LAYERS)
for ax, (title, words) in zip(axes, groups.items()):
    for w in words:
        ax.plot(xs, trajectories[w], "-o", markersize=3, linewidth=1.2,
                label=f"{w} (GloVe={GLOVE_BASELINES.get(w, '?'):+.2f})")
        # Mark GloVe baseline on left margin
        if w in GLOVE_BASELINES:
            ax.scatter([-0.7], [GLOVE_BASELINES[w]], marker=">", s=30,
                       color=ax.lines[-1].get_color(), zorder=5)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("projection on cleaned UP")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-1, N_LAYERS)

fig.suptitle("exp127 — Pythia 410M: per-layer projection of test words on freq-stripped UP\n"
             "GloVe baseline marked at left margin (▶). "
             "Substrate-primitive prediction: depth crosses from − to 0/+ across layers.",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp127_layer_trajectories.png", dpi=120)
print(f"\nSaved exp127_layer_trajectories.png")


# Compact single-pane plot focused on the critical discriminator
fig2, ax = plt.subplots(figsize=(10, 6))
critical = ["deep", "deeper", "deepest", "abyss"]
height_ref = ["tall", "towering"]
status_ref = ["status", "power"]

for w in critical:
    ax.plot(xs, trajectories[w], "-o", markersize=4, linewidth=2,
            label=f"DEPTH: {w}")
for w in height_ref:
    ax.plot(xs, trajectories[w], "--s", markersize=3, linewidth=1.5,
            alpha=0.7, label=f"HEIGHT: {w}")
for w in status_ref:
    ax.plot(xs, trajectories[w], ":^", markersize=3, linewidth=1.5,
            alpha=0.7, label=f"STATUS: {w}")

ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("layer")
ax.set_ylabel("projection on cleaned UP direction")
ax.set_title("exp127 — The substrate-primitive transition in Pythia 410M\n"
             "DEPTH words should cross from − (directional) to ≥0 (magnitude) "
             "across layers", fontsize=11)
ax.legend(fontsize=9, loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp127_depth_transition.png", dpi=120)
print(f"Saved exp127_depth_transition.png")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp127_results.npz",
         n_layers=N_LAYERS,
         **{f"traj_{w}": trajectories[w] for w in TEST_WORDS},
         cos_up_freq_per_layer=np.array(cos_up_freq_per_layer),
)
print("Saved exp127_results.npz")
