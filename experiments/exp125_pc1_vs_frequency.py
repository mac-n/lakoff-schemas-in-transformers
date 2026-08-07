"""
exp125_pc1_vs_frequency.py — does PC1 of residual streams look like a
frequency axis or a "median band" axis?

Niamh's hypothesis: the dominant direction in residual streams encodes
"informativeness/middle-band-ness", not raw frequency. So PC1 projection
should be inverted-U-shaped against log-frequency (peak in the middle,
drops at both extremes) — not monotonic.

Test:
1. Sample ~1500 words spanning the full frequency range using SUBTLEX.
2. At each of 7 layers (0, 4, 8, 12, 16, 20, 23), extract residual
   (last-position) for each word.
3. Compute PC1 of these residuals at each layer.
4. Plot PC1 projection vs log-frequency.
5. Bonus: also compute and overlay (a) the freq axis projection, and
   (b) a "salience axis" projection (middle band vs spike+tail).
6. Direct visual answer to "is it monotonic or bell-curve?"
"""

import numpy as np
import pandas as pd
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

LAYERS = [0, 4, 8, 12, 16, 20, 23]


# ============================================================================
# Get SUBTLEX-frequency words, stratified sample across frequency range
# ============================================================================

print("\nLoading Brysbaert/SUBTLEX...")
brys = pd.read_csv("/Users/macn/Documents/embeddingexp/norms/Brysbaert_concreteness.txt",
                   sep="\t")
print(f"  {len(brys)} rows in Brysbaert")

# SUBTLEX column has frequency counts (in some unit). Keep words with freq>0
brys = brys[brys["SUBTLEX"] > 0].copy()
brys["log_freq"] = np.log10(brys["SUBTLEX"])
brys["word"] = brys["Word"].astype(str).str.lower()
print(f"  {len(brys)} words with SUBTLEX freq > 0")


def n_toks(w):
    return len(model.tokenizer.encode(w, add_special_tokens=False))


# Stratified sample across log-freq bins
N_BINS = 30
TARGET_PER_BIN = 75
print(f"\nStratified-sampling ~{N_BINS * TARGET_PER_BIN} words across "
      f"{N_BINS} log-freq bins...")

bins = pd.qcut(brys["log_freq"], q=N_BINS, duplicates="drop")
brys["bin"] = bins

sampled_words = []
for bin_label, group in brys.groupby("bin", observed=True):
    cands = group["word"].tolist()
    np.random.seed(hash(str(bin_label)) % 2**32)
    np.random.shuffle(cands)
    kept = 0
    for w in cands:
        if not isinstance(w, str) or not w.isalpha():
            continue
        if n_toks(w) != 1:
            continue
        sampled_words.append(w)
        kept += 1
        if kept >= TARGET_PER_BIN:
            break

# Deduplicate while preserving order
seen = set()
sampled_words = [w for w in sampled_words
                 if not (w in seen or seen.add(w))]

# Get their frequencies in our sample
freq_lookup = dict(zip(brys["word"], brys["SUBTLEX"]))
log_freq_lookup = dict(zip(brys["word"], brys["log_freq"]))

sample_log_freqs = np.array([log_freq_lookup[w] for w in sampled_words])
print(f"  collected {len(sampled_words)} single-token words")
print(f"  log_freq range: [{sample_log_freqs.min():.2f}, {sample_log_freqs.max():.2f}]")


# ============================================================================
# Extract residuals at all selected layers
# ============================================================================

hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
residuals_per_layer = {L: [] for L in LAYERS}

print(f"\nExtracting residuals at {len(LAYERS)} layers...")
for i, w in enumerate(sampled_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    for L in LAYERS:
        r = cache[f"blocks.{L}.hook_resid_post"][0, -1, :].clone().cpu().numpy()
        residuals_per_layer[L].append(r)
    if (i + 1) % 250 == 0:
        print(f"  {i+1}/{len(sampled_words)} words")

for L in LAYERS:
    residuals_per_layer[L] = np.array(residuals_per_layer[L])
    print(f"  Layer {L}: residual matrix {residuals_per_layer[L].shape}")


# ============================================================================
# Compute PC1 and project at each layer
# ============================================================================

print("\nComputing PC1 and projecting at each layer...")
pc1_projections = {}
pc1_dirs = {}
mean_dirs = {}
explained_var_ratio = {}

for L in LAYERS:
    R = residuals_per_layer[L]
    mean_dir = R.mean(axis=0)
    mean_dirs[L] = mean_dir / np.linalg.norm(mean_dir)
    R_centered = R - mean_dir
    U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
    pc1 = Vt[0]
    pc1_dirs[L] = pc1
    pc1_projections[L] = R_centered @ pc1
    explained_var_ratio[L] = (S[0]**2) / (S**2).sum()
    print(f"  L{L}: ‖mean‖={np.linalg.norm(mean_dir):.2f}, "
          f"PC1 explains {explained_var_ratio[L]:.1%} of variance")


# ============================================================================
# Identify median-band words and build "salience axis"
# ============================================================================

# Token-occurrence-weighted CDF over the sample
order = np.argsort(sample_log_freqs)  # ascending in freq
sorted_freqs = np.array([freq_lookup[sampled_words[i]] for i in order])
cum = np.cumsum(sorted_freqs) / sorted_freqs.sum()

# Identify top/bottom 3% by cumulative MASS
# Bottom 3%: words whose cumulative mass from the bottom is below 0.03
# Top 3%: words whose cumulative mass from the top exceeds 0.97
bottom_mask = cum < 0.03
top_mask = cum > 0.97
middle_mask = ~(bottom_mask | top_mask)

bottom_words = [sampled_words[i] for i in order[bottom_mask]]
top_words = [sampled_words[i] for i in order[top_mask]]
middle_words = [sampled_words[i] for i in order[middle_mask]]

print(f"\nMass-weighted partition:")
print(f"  Bottom 3% of mass: {len(bottom_words)} words (e.g. {bottom_words[:5]}...)")
print(f"  Top 3% of mass: {len(top_words)} words ({top_words})")
print(f"  Middle 94%: {len(middle_words)} words")

# Build salience axis per layer
salience_dirs = {}
salience_projections = {}
for L in LAYERS:
    R = residuals_per_layer[L]
    bot_idx = [sampled_words.index(w) for w in bottom_words]
    top_idx = [sampled_words.index(w) for w in top_words]
    mid_idx = [sampled_words.index(w) for w in middle_words]
    extreme_mean = R[bot_idx + top_idx].mean(axis=0)
    middle_mean = R[mid_idx].mean(axis=0)
    raw = middle_mean - extreme_mean
    salience_dirs[L] = raw / np.linalg.norm(raw)
    salience_projections[L] = R @ salience_dirs[L]


# ============================================================================
# Plot — direct visual of "is it bell curve or monotonic?"
# ============================================================================

# Smooth via moving average over freq-sorted points
def smoothed(x, y, window=30):
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    pad = window // 2
    y_smooth = np.convolve(y_sorted, np.ones(window)/window, mode="same")
    return x_sorted, y_smooth


fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
axes = axes.flatten()

for i, L in enumerate(LAYERS):
    ax = axes[i]
    proj = pc1_projections[L]
    # Sign-align PC1: ensure most-common words tend to be on the positive side
    # for consistent display (sign of PC1 is arbitrary)
    if np.corrcoef(sample_log_freqs, proj)[0, 1] < 0:
        proj = -proj
    ax.scatter(sample_log_freqs, proj, s=4, alpha=0.3, color="tab:blue")
    xs_s, ys_s = smoothed(sample_log_freqs, proj, window=80)
    ax.plot(xs_s, ys_s, color="black", linewidth=2.0,
            label=f"PC1 projection (smoothed)")
    # Also show the salience-axis projection on a 2nd y axis
    ax2 = ax.twinx()
    sal_proj = salience_projections[L]
    xs_s2, ys_s2 = smoothed(sample_log_freqs, sal_proj, window=80)
    ax2.plot(xs_s2, ys_s2, color="tab:red", linewidth=1.6, linestyle="--",
             label="salience axis projection (smoothed)", alpha=0.85)
    ax.set_title(f"Layer {L}  (PC1 explains {explained_var_ratio[L]:.1%} var)",
                 fontsize=11)
    if i % 4 == 0:
        ax.set_ylabel("PC1 projection")
    if i >= 4:
        ax.set_xlabel("log10(SUBTLEX freq)")
    ax2.set_ylabel("salience projection", color="tab:red", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="grey", lw=0.5)
    cos_pc1_sal = float(pc1_dirs[L] @ salience_dirs[L])
    cos_pc1_mean = float(pc1_dirs[L] @ mean_dirs[L])
    ax.text(0.02, 0.95, f"cos(PC1, salience)={cos_pc1_sal:+.2f}\n"
            f"cos(PC1, mean)={cos_pc1_mean:+.2f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# Last subplot empty
if len(LAYERS) < 8:
    axes[-1].axis("off")

fig.suptitle("exp125 — PC1 projection vs log-frequency at each layer\n"
             "Monotonic = PC1 is freq axis; bell-curve = PC1 is median-band axis\n"
             "(blue = PC1 projection, red = salience axis projection [middle vs extreme])",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp125_pc1_vs_frequency.png", dpi=120)
print(f"\n  saved exp125_pc1_vs_frequency.png")


# ============================================================================
# Quantitative comparison: cos(PC1, X) for X in {salience, freq, mean}
# ============================================================================

# Build a "freq axis" from spike vs tail (no middle)
freq_axis_per_layer = {}
for L in LAYERS:
    R = residuals_per_layer[L]
    bot_idx = [sampled_words.index(w) for w in bottom_words]
    top_idx = [sampled_words.index(w) for w in top_words]
    bot_mean = R[bot_idx].mean(axis=0)
    top_mean = R[top_idx].mean(axis=0)
    raw = top_mean - bot_mean
    freq_axis_per_layer[L] = raw / np.linalg.norm(raw)

print("\n" + "=" * 70)
print("Quantitative comparison: which axis does PC1 align with?")
print("=" * 70)
print(f"\n  {'layer':>6}  {'cos(PC1, salience)':>20}  {'cos(PC1, freq_axis)':>20}  "
      f"{'cos(PC1, mean_dir)':>20}")
for L in LAYERS:
    c_sal = float(pc1_dirs[L] @ salience_dirs[L])
    c_freq = float(pc1_dirs[L] @ freq_axis_per_layer[L])
    c_mean = float(pc1_dirs[L] @ mean_dirs[L])
    print(f"  {L:>6}  {c_sal:>+20.3f}  {c_freq:>+20.3f}  {c_mean:>+20.3f}")

# Also: correlation of PC1 projection with log-freq directly
print(f"\n  {'layer':>6}  {'Pearson r(PC1_proj, log_freq)':>34}  "
      f"{'Spearman r':>14}")
from scipy.stats import pearsonr, spearmanr
for L in LAYERS:
    p = pc1_projections[L]
    if np.corrcoef(sample_log_freqs, p)[0, 1] < 0:
        p = -p
    pr, _ = pearsonr(sample_log_freqs, p)
    sr, _ = spearmanr(sample_log_freqs, p)
    print(f"  {L:>6}  {pr:>+34.3f}  {sr:>+14.3f}")

# Save
np.savez("/Users/macn/Documents/embeddingexp/exp125_results.npz",
         layers=np.array(LAYERS),
         sample_log_freqs=sample_log_freqs,
         **{f"pc1_proj_L{L}": pc1_projections[L] for L in LAYERS},
         **{f"salience_proj_L{L}": salience_projections[L] for L in LAYERS},
         **{f"explained_var_L{L}": explained_var_ratio[L] for L in LAYERS},
         sample_words=np.array(sampled_words),
)
print("\nSaved exp125_results.npz")
