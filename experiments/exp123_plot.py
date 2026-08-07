"""
exp123_plot.py — visualise the relational structure result.
Produces:
  - exp123_predicted_vs_unpredicted.png    (M3 result)
  - exp123_couplings_across_layers.png     (per-pair trajectories)
  - exp123_layer_similarity.png            (M2 matrix)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import json

# Load
d = np.load("/Users/macn/Documents/embeddingexp/exp123_results.npz")
with open("/Users/macn/Documents/embeddingexp/exp123_config.json") as f:
    cfg = json.load(f)

real = d["real_cos_matrices"]                  # [L, N, N]
nullm = d["null_cos_matrices"]                  # [K, L, N, N]
real_sim = d["real_layer_sim"]                  # [L, L]
null_sim = d["mean_null_layer_sim"]             # [L, L]
schemas = list(d["schema_names"])
N_LAYERS, N, _ = real.shape

PREDICTED = [tuple(p) for p in cfg["predicted_positive_couplings"]]


# ---------------------------------------------------------------------------
# Plot 1 — per-pair cosine trajectories across layers
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))
xs = range(N_LAYERS)

# Plot all unpredicted couplings in grey
for i, j in combinations(range(N), 2):
    pair = (schemas[i], schemas[j])
    if pair in PREDICTED or (pair[1], pair[0]) in PREDICTED:
        continue
    ys = real[:, i, j]
    ax.plot(xs, ys, color="grey", alpha=0.35, linewidth=0.8)

# Plot predicted couplings in colour
cmap = plt.get_cmap("tab10")
for k, (a, b) in enumerate(PREDICTED):
    i = schemas.index(a)
    j = schemas.index(b)
    ys = real[:, i, j]
    ax.plot(xs, ys, color=cmap(k), linewidth=2.2,
            label=f"{a} ↔ {b}  (mean={ys.mean():+.3f})")

ax.axhline(0, color="black", lw=0.5, alpha=0.5)
ax.set_xlabel("layer")
ax.set_ylabel("cos(schema_i, schema_j)")
ax.set_title("exp123 — per-pair cosines across layers of Pythia 410M\n"
             "predicted-positive couplings (coloured) vs unpredicted (grey)")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp123_couplings_across_layers.png",
            dpi=120)
print("  saved exp123_couplings_across_layers.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 2 — predicted vs unpredicted distribution (M3)
# ---------------------------------------------------------------------------

pred_means = []
for a, b in PREDICTED:
    i = schemas.index(a); j = schemas.index(b)
    pred_means.append(real[:, i, j].mean())

unpred_means = []
for i, j in combinations(range(N), 2):
    pair = (schemas[i], schemas[j])
    if pair in PREDICTED or (pair[1], pair[0]) in PREDICTED:
        continue
    unpred_means.append(real[:, i, j].mean())

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-0.4, 0.5, 30)
ax.hist(unpred_means, bins=bins, alpha=0.6, label=f"unpredicted (n={len(unpred_means)})",
        color="grey", edgecolor="white")
ax.hist(pred_means, bins=bins, alpha=0.75, label=f"predicted (n={len(pred_means)})",
        color="tab:orange", edgecolor="white")
ax.axvline(np.mean(pred_means), color="tab:orange", lw=2, linestyle="--",
           label=f"predicted mean={np.mean(pred_means):+.3f}")
ax.axvline(np.mean(unpred_means), color="grey", lw=2, linestyle="--",
           label=f"unpredicted mean={np.mean(unpred_means):+.3f}")
ax.set_xlabel("mean(cos) across layers")
ax.set_ylabel("count of schema pairs")
ax.set_title("exp123 — Lakoff-predicted vs unpredicted schema couplings\n"
             "(mean cosine across all 24 layers, Pythia 410M, freq-stripped)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp123_predicted_vs_unpredicted.png",
            dpi=120)
print("  saved exp123_predicted_vs_unpredicted.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 3 — layer-similarity matrices (real vs null)
# ---------------------------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
im1 = ax1.imshow(real_sim, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
ax1.set_title(f"REAL schemas\nmean off-diag = {real_sim[~np.eye(N_LAYERS, dtype=bool)].mean():+.3f}")
ax1.set_xlabel("layer"); ax1.set_ylabel("layer")
plt.colorbar(im1, ax=ax1, fraction=0.046)

im2 = ax2.imshow(null_sim, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
ax2.set_title(f"STRONG null (random anchor partitions)\n"
              f"mean off-diag = {null_sim[~np.eye(N_LAYERS, dtype=bool)].mean():+.3f}")
ax2.set_xlabel("layer"); ax2.set_ylabel("layer")
plt.colorbar(im2, ax=ax2, fraction=0.046)

diff = real_sim - null_sim
im3 = ax3.imshow(diff, cmap="RdBu_r", vmin=-0.3, vmax=0.3, origin="lower")
ax3.set_title("real − null (excess preservation)")
ax3.set_xlabel("layer"); ax3.set_ylabel("layer")
plt.colorbar(im3, ax=ax3, fraction=0.046)

fig.suptitle("exp123 — across-layer configuration similarity\n"
             "(cos of vectorised schema-cosine matrix at layer a vs layer b)",
             fontsize=11)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp123_layer_similarity.png",
            dpi=120)
print("  saved exp123_layer_similarity.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 4 — mean schema cosine matrix across layers (the geometry summary)
# ---------------------------------------------------------------------------

mean_cos_mat = real.mean(axis=0)  # [N, N]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mean_cos_mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5, origin="upper")
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels([s[:10] for s in schemas], rotation=45, ha="right")
ax.set_yticklabels([s[:10] for s in schemas])
for i in range(N):
    for j in range(N):
        v = mean_cos_mat[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.3 else "black", fontsize=8)
ax.set_title("exp123 — mean schema cosine matrix (averaged across all 24 layers)\n"
             "Pythia 410M, freq-stripped Lakoff MML directions")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp123_mean_cosine_matrix.png", dpi=120)
print("  saved exp123_mean_cosine_matrix.png")
plt.close()
print("\ndone.")
