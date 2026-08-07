"""
exp130_suffix_schema_matrix.py — do morphological operators map onto
their theoretically-predicted Lakoff schemas?

For each suffix/affix, build a "suffix direction" =
mean(inflected_residual − base_residual). Then project this direction
onto each of the 8 Lakoff schema directions. Build heatmap.

Predictions:
  -ER comparative    → UP-DOWN (already shown in exp128/129)
  -EST superlative   → UP-DOWN
  -ING progressive   → PATH-MOTION
  -ED past tense     → FORWARD-BACK (back direction)
  -S plural          → IN-OUT or UP-DOWN (MORE-IS-UP)
  un- negation       → LIGHT-DARK (negation as darkening)
  re- repetition     → FORWARD-BACK (back direction)

Run at L4 (peak directional in Pythia per exp127) and L12 (mid-stable).

Also: random base-base pair baseline as null.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers

LAYERS_OF_INTEREST = [4, 8, 12, 16, 20]
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS_OF_INTEREST]


# ============================================================================
# Suffix pairs
# ============================================================================

SUFFIX_PAIRS = {
    "ER_comparative": [
        ("big", "bigger"), ("small", "smaller"), ("tall", "taller"),
        ("high", "higher"), ("low", "lower"), ("deep", "deeper"),
        ("wide", "wider"), ("fast", "faster"), ("slow", "slower"),
        ("old", "older"), ("new", "newer"), ("hot", "hotter"),
        ("cold", "colder"), ("hard", "harder"), ("soft", "softer"),
    ],
    "EST_superlative": [
        ("big", "biggest"), ("small", "smallest"), ("tall", "tallest"),
        ("high", "highest"), ("low", "lowest"), ("deep", "deepest"),
        ("old", "oldest"), ("new", "newest"), ("hot", "hottest"),
        ("cold", "coldest"), ("hard", "hardest"),
    ],
    "ING_progressive": [
        ("walk", "walking"), ("run", "running"), ("jump", "jumping"),
        ("sit", "sitting"), ("stand", "standing"), ("swim", "swimming"),
        ("think", "thinking"), ("talk", "talking"), ("sing", "singing"),
        ("dance", "dancing"), ("play", "playing"), ("work", "working"),
        ("read", "reading"), ("write", "writing"), ("eat", "eating"),
    ],
    "ED_past": [
        ("walk", "walked"), ("jump", "jumped"), ("look", "looked"),
        ("talk", "talked"), ("play", "played"), ("work", "worked"),
        ("ask", "asked"), ("call", "called"), ("learn", "learned"),
        ("move", "moved"), ("stop", "stopped"), ("start", "started"),
    ],
    "S_plural": [
        ("cat", "cats"), ("dog", "dogs"), ("book", "books"),
        ("house", "houses"), ("car", "cars"), ("tree", "trees"),
        ("bird", "birds"), ("hand", "hands"), ("eye", "eyes"),
        ("girl", "girls"), ("boy", "boys"), ("year", "years"),
    ],
    "UN_negation": [
        ("happy", "unhappy"), ("kind", "unkind"), ("healthy", "unhealthy"),
        ("safe", "unsafe"), ("clear", "unclear"), ("clean", "unclean"),
        ("fair", "unfair"), ("certain", "uncertain"),
        ("known", "unknown"), ("seen", "unseen"),
    ],
    "RE_repetition": [
        ("do", "redo"), ("make", "remake"), ("build", "rebuild"),
        ("write", "rewrite"), ("read", "reread"), ("start", "restart"),
        ("create", "recreate"), ("paint", "repaint"),
    ],
}

# Schemas to project onto
SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]

# Frequency axis
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]


# Collect all unique words we need
all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for schema_name in SCHEMA_NAMES:
    for pos, neg in LAKOFF_SCHEMAS_MML[schema_name]:
        all_words.add(pos); all_words.add(neg)
all_words = sorted(all_words)
print(f"\nCollecting residuals for {len(all_words)} words at layers {LAYERS_OF_INTEREST}...")

residuals = {}
for i, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                    for L in LAYERS_OF_INTEREST}
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(all_words)}")


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)


def build_schema_direction(schema_name, layer):
    """Schema direction with freq strip."""
    pairs = LAKOFF_SCHEMAS_MML[schema_name]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    raw = mean_acts(pos, layer) - mean_acts(neg, layer)
    raw = raw / np.linalg.norm(raw)
    # Freq strip
    freq_raw = mean_acts(COMMON, layer) - mean_acts(RARE, layer)
    freq = freq_raw / np.linalg.norm(freq_raw)
    clean = raw - (raw @ freq) * freq
    return clean / np.linalg.norm(clean)


def build_suffix_direction(suffix_name, layer):
    """Suffix direction = mean(inflected − base), normalised."""
    pairs = SUFFIX_PAIRS[suffix_name]
    diffs = []
    for base, infl in pairs:
        b = residuals[base][layer]
        i = residuals[infl][layer]
        b_unit = b / np.linalg.norm(b)
        i_unit = i / np.linalg.norm(i)
        diffs.append(i_unit - b_unit)
    raw = np.mean(diffs, axis=0)
    return raw / np.linalg.norm(raw)


# ============================================================================
# Build matrix at each layer
# ============================================================================

matrices = {L: np.zeros((len(SUFFIX_PAIRS), len(SCHEMA_NAMES)))
            for L in LAYERS_OF_INTEREST}

for L in LAYERS_OF_INTEREST:
    schema_dirs = {s: build_schema_direction(s, L) for s in SCHEMA_NAMES}
    for i, suffix_name in enumerate(SUFFIX_PAIRS):
        suffix_dir = build_suffix_direction(suffix_name, L)
        for j, schema_name in enumerate(SCHEMA_NAMES):
            matrices[L][i, j] = float(suffix_dir @ schema_dirs[schema_name])


# ============================================================================
# Random control: 100 trials of randomly-paired word-pair "pseudo-suffixes"
# ============================================================================

# Pool of base words from all suffix pairs
all_base_words = []
for pairs in SUFFIX_PAIRS.values():
    for b, _ in pairs:
        all_base_words.append(b)
all_base_words = list(set(all_base_words))

K_NULL = 100
np.random.seed(42)
null_matrices = {L: np.zeros((K_NULL, len(SCHEMA_NAMES))) for L in LAYERS_OF_INTEREST}

for L in LAYERS_OF_INTEREST:
    schema_dirs = {s: build_schema_direction(s, L) for s in SCHEMA_NAMES}
    for k in range(K_NULL):
        # Make a fake "suffix" by random word pairings (10 pairs, like our suffixes)
        n_pairs = 12
        words_shuffled = all_base_words.copy()
        np.random.shuffle(words_shuffled)
        diffs = []
        for p in range(n_pairs):
            w1, w2 = words_shuffled[2*p], words_shuffled[2*p + 1]
            v1 = residuals[w1][L] / np.linalg.norm(residuals[w1][L])
            v2 = residuals[w2][L] / np.linalg.norm(residuals[w2][L])
            diffs.append(v2 - v1)
        raw = np.mean(diffs, axis=0)
        fake_dir = raw / np.linalg.norm(raw)
        for j, schema_name in enumerate(SCHEMA_NAMES):
            null_matrices[L][k, j] = float(fake_dir @ schema_dirs[schema_name])


# ============================================================================
# Print
# ============================================================================

for L in LAYERS_OF_INTEREST:
    print(f"\n{'=' * 78}")
    print(f"Layer {L} — suffix × schema cosines")
    print(f"{'=' * 78}")
    print(f"\n  {'suffix':<18}  " + "  ".join(f"{s[:9]:>9}" for s in SCHEMA_NAMES))
    for i, suffix_name in enumerate(SUFFIX_PAIRS):
        row = f"  {suffix_name:<18}  "
        for j in range(len(SCHEMA_NAMES)):
            row += f"{matrices[L][i, j]:>+9.3f}  "
        print(row)
    print(f"\n  null abs(mean)/95pct per schema:")
    row = f"  {'null':<18}  "
    for j in range(len(SCHEMA_NAMES)):
        null_abs = np.abs(null_matrices[L][:, j])
        row += f" {np.percentile(null_abs, 95):>+8.3f}  "
    print(row)


# ============================================================================
# Plot heatmaps at each layer
# ============================================================================

fig, axes = plt.subplots(1, len(LAYERS_OF_INTEREST), figsize=(20, 5), sharey=True)
for ax, L in zip(axes, LAYERS_OF_INTEREST):
    im = ax.imshow(matrices[L], cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(SCHEMA_NAMES)))
    ax.set_xticklabels([s[:8] for s in SCHEMA_NAMES], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(SUFFIX_PAIRS)))
    ax.set_yticklabels(list(SUFFIX_PAIRS.keys()), fontsize=8)
    for i in range(len(SUFFIX_PAIRS)):
        for j in range(len(SCHEMA_NAMES)):
            v = matrices[L][i, j]
            null_95 = np.percentile(np.abs(null_matrices[L][:, j]), 95)
            sig = abs(v) > null_95
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.2 else "black",
                    fontsize=7, fontweight="bold" if sig else "normal")
    ax.set_title(f"Layer {L}", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.04)

fig.suptitle("exp130 — suffix × Lakoff-schema cosines across layers of Pythia 410M\n"
             "Bold = exceeds 95th percentile of null (random word-pair pseudo-suffix)",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp130_suffix_schema_heatmap.png", dpi=120)
print("\nSaved exp130_suffix_schema_heatmap.png")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp130_results.npz",
         layers=np.array(LAYERS_OF_INTEREST),
         schema_names=np.array(SCHEMA_NAMES),
         suffix_names=np.array(list(SUFFIX_PAIRS.keys())),
         **{f"matrix_L{L}": matrices[L] for L in LAYERS_OF_INTEREST},
         **{f"null_L{L}": null_matrices[L] for L in LAYERS_OF_INTEREST})
print("Saved exp130_results.npz")
