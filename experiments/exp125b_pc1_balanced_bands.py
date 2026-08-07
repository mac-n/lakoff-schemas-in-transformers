"""
exp125b_pc1_balanced_bands.py — same hypothesis, better sampling.

The original exp125 had a sampling problem: stratified by log-freq bins
of the BRYSBAERT table, but Brysbaert is sample-biased toward content
words — the very-most-common function words ("the/of/and") got under-
represented at the top, while the long tail saturated the bottom 3%
of cumulative-mass even though only 965 words occupied that mass.

Fix: take an explicit balanced sample across 9 frequency bands by RANK,
not by cumulative mass:

  Band 1 (spike):           rank   1-10
  Band 2 (top function):    rank  11-50
  Band 3 (top content):     rank  51-300
  Band 4 (high content):    rank 301-1000
  Band 5 (MID content):     rank 1001-3000
  Band 6 (mid-low content): rank 3001-7000
  Band 7 (low content):     rank 7001-15000
  Band 8 (rare content):    rank 15001-30000
  Band 9 (tail):            rank 30001+

Sample ~80 single-token Pythia words from each band.

Then plot PC1 projection per word vs band (or vs log-rank), grouped
by band. If salience-as-middle-band is right, the middle bands (5, 6)
should have systematically different PC1 projections than the extreme
bands (1, 9). Bell curve = peak (or trough) in the middle bands.
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
print(f"  {N_LAYERS} layers")

LAYERS = [0, 4, 8, 12, 16, 20, 23]

# Load Brysbaert/SUBTLEX, rank by frequency
brys = pd.read_csv("/Users/macn/Documents/embeddingexp/norms/Brysbaert_concreteness.txt",
                   sep="\t")
brys = brys[brys["SUBTLEX"] > 0].copy()
brys["word"] = brys["Word"].astype(str).str.lower()
brys = brys[brys["word"].apply(lambda w: isinstance(w, str) and w.isalpha())]
brys = brys.sort_values("SUBTLEX", ascending=False).reset_index(drop=True)
brys["rank"] = np.arange(1, len(brys) + 1)
brys["log_rank"] = np.log10(brys["rank"])
brys["log_freq"] = np.log10(brys["SUBTLEX"])
print(f"  {len(brys)} Brysbaert words sorted by SUBTLEX rank")
print(f"  rank 1 = '{brys.iloc[0]['word']}'  freq={brys.iloc[0]['SUBTLEX']:.0f}")
print(f"  rank 100 = '{brys.iloc[99]['word']}'  freq={brys.iloc[99]['SUBTLEX']:.0f}")
print(f"  rank 1000 = '{brys.iloc[999]['word']}'  freq={brys.iloc[999]['SUBTLEX']:.0f}")
print(f"  rank 10000 = '{brys.iloc[9999]['word']}'  freq={brys.iloc[9999]['SUBTLEX']:.0f}")


def n_toks(w):
    return len(model.tokenizer.encode(w, add_special_tokens=False))


BANDS = [
    ("B1_spike",         1, 10,       "top-10 function words"),
    ("B2_top_function", 11, 50,       "rank 11-50"),
    ("B3_top_content", 51, 300,      "rank 51-300"),
    ("B4_high",       301, 1000,     "rank 301-1000"),
    ("B5_mid",       1001, 3000,     "rank 1001-3000"),
    ("B6_mid_low",   3001, 7000,     "rank 3001-7000"),
    ("B7_low",       7001, 15000,    "rank 7001-15000"),
    ("B8_rare",     15001, 30000,    "rank 15001-30000"),
    ("B9_tail",     30001, len(brys), "rank 30001+"),
]

TARGET_PER_BAND = 80

print("\nSampling from balanced rank bands...")
band_words = {}
for name, lo, hi, descr in BANDS:
    candidates = brys[(brys["rank"] >= lo) & (brys["rank"] <= hi)]["word"].tolist()
    np.random.seed(hash(name) % 2**32)
    np.random.shuffle(candidates)
    kept = []
    for w in candidates:
        if n_toks(w) != 1:
            continue
        kept.append(w)
        if len(kept) >= TARGET_PER_BAND:
            break
    band_words[name] = kept
    print(f"  {name}: {len(kept)}/{TARGET_PER_BAND} single-token words "
          f"({descr})  e.g. {kept[:5]}")

# Combine
all_words = []
word_band = {}
for name, lo, hi, _ in BANDS:
    for w in band_words[name]:
        if w not in word_band:
            all_words.append(w)
            word_band[w] = name
word_rank = dict(zip(brys["word"], brys["rank"]))
word_logfreq = dict(zip(brys["word"], brys["log_freq"]))

print(f"\nTotal unique words sampled: {len(all_words)}")


# Extract residuals
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
residuals_per_layer = {L: [] for L in LAYERS}

print(f"\nExtracting residuals at {len(LAYERS)} layers for {len(all_words)} words...")
for i, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    for L in LAYERS:
        r = cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
        residuals_per_layer[L].append(r)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(all_words)} words")

for L in LAYERS:
    residuals_per_layer[L] = np.array(residuals_per_layer[L])


# PC1 per layer
pc1_proj_per_layer = {}
pc1_dirs = {}
mean_dirs = {}
for L in LAYERS:
    R = residuals_per_layer[L]
    mean_dir = R.mean(axis=0)
    mean_dirs[L] = mean_dir / np.linalg.norm(mean_dir)
    Rc = R - mean_dir
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    pc1 = Vt[0]
    pc1_dirs[L] = pc1
    pc1_proj_per_layer[L] = Rc @ pc1


# Group by band for plotting
band_names = [b[0] for b in BANDS]
band_indices = {bn: [i for i, w in enumerate(all_words) if word_band[w] == bn]
                for bn in band_names}

word_log_ranks = np.array([np.log10(word_rank[w]) for w in all_words])
word_log_freqs = np.array([word_logfreq[w] for w in all_words])


# Plot: PC1 projection grouped by band
print("\nPlotting...")
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

for i, L in enumerate(LAYERS):
    ax = axes[i]
    proj = pc1_proj_per_layer[L].copy()
    # sign-align so common words tend positive
    if np.mean(proj[band_indices["B1_spike"]]) < np.mean(proj[band_indices["B5_mid"]]):
        # already as we want
        pass

    means_per_band = []
    ses_per_band = []
    for bn in band_names:
        idx = band_indices[bn]
        if len(idx) == 0:
            means_per_band.append(np.nan)
            ses_per_band.append(np.nan)
            continue
        vals = proj[idx]
        means_per_band.append(np.mean(vals))
        ses_per_band.append(np.std(vals) / np.sqrt(len(vals)))

    xs = np.arange(len(band_names))
    # error bars
    ax.errorbar(xs, means_per_band, yerr=ses_per_band, fmt='o-', capsize=4,
                color='tab:blue', linewidth=2, label="PC1 projection mean")
    # individual points
    for j, bn in enumerate(band_names):
        idx = band_indices[bn]
        if len(idx) == 0:
            continue
        ax.scatter([j]*len(idx), proj[idx], s=6, alpha=0.25, color='tab:blue')

    ax.set_xticks(xs)
    ax.set_xticklabels([bn.replace('B', '').replace('_', '\n', 1) for bn in band_names],
                       rotation=0, fontsize=7)
    ax.set_title(f"Layer {L}", fontsize=11)
    if i % 4 == 0:
        ax.set_ylabel("PC1 projection")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='grey', lw=0.5)

if len(LAYERS) < 8:
    axes[-1].axis('off')

fig.suptitle("exp125b — PC1 projection across rank bands at each layer\n"
             "(Bell-curve hypothesis: middle bands B4-B6 differ from extremes B1, B9)",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp125b_pc1_by_band.png", dpi=120)
print("  saved exp125b_pc1_by_band.png")


# Quantitative summary
print("\n" + "=" * 70)
print("Mean PC1 projection per band per layer")
print("=" * 70)
header = f"\n  {'band':<22}  " + "  ".join(f"L{L:>2}" for L in LAYERS)
print(header)
for bn in band_names:
    row = f"  {bn:<22}  "
    for L in LAYERS:
        proj = pc1_proj_per_layer[L]
        idx = band_indices[bn]
        if len(idx) == 0:
            row += f"   --  "
        else:
            row += f" {np.mean(proj[idx]):+5.2f} "
    print(row)


# How "bell-curve-like" is the relationship?
# Quadratic fit vs linear fit — compare R²
print("\n" + "=" * 70)
print("Quadratic vs linear fit of PC1 projection ~ log_rank")
print("=" * 70)
print(f"  Linear R² → frequency-axis-like; Quadratic ≫ Linear → bell-curve-like")
print(f"  {'layer':>6}  {'linear R²':>12}  {'quad R²':>12}  {'quad/linear':>12}")
for L in LAYERS:
    proj = pc1_proj_per_layer[L]
    x = word_log_ranks
    # linear
    A_lin = np.column_stack([x, np.ones_like(x)])
    coef_lin, _, _, _ = np.linalg.lstsq(A_lin, proj, rcond=None)
    pred_lin = A_lin @ coef_lin
    ss_res_lin = ((proj - pred_lin) ** 2).sum()
    ss_tot = ((proj - proj.mean()) ** 2).sum()
    r2_lin = 1 - ss_res_lin / ss_tot
    # quadratic
    A_quad = np.column_stack([x**2, x, np.ones_like(x)])
    coef_quad, _, _, _ = np.linalg.lstsq(A_quad, proj, rcond=None)
    pred_quad = A_quad @ coef_quad
    ss_res_quad = ((proj - pred_quad) ** 2).sum()
    r2_quad = 1 - ss_res_quad / ss_tot
    print(f"  {L:>6}  {r2_lin:>12.4f}  {r2_quad:>12.4f}  {r2_quad/max(r2_lin, 1e-9):>12.2f}")
