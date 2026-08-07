"""
SAE dictionary dimensionality reduction
========================================
Question: do the 32,768 learned feature directions live on a low-dimensional
skeleton? If yes, that skeleton is where image-schema-like structural primitives
would live -- schemas as the coordinate system the features are expressed in,
not as individual features.

The experiment is only meaningful against a null: random unit vectors in the
same dimension. Real features being MORE compressible than random is the result.

Repo: EleutherAI/sae-pythia-160m-32k  (decoder = 32768 features x 768 dims)
Loads via the EleutherAI `sae` library: pip install sae
(If the library API has drifted, hand this to Claude Code to fix the loading
 block -- the science below the loading block is the part that matters.)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# 1. LOAD THE DICTIONARY
# ----------------------------------------------------------------------
# The EleutherAI sae library exposes Sae.load_from_hub. Each layer's MLP has
# its own SAE. Pick one layer to start (mid-network tends to be most
# semantically organised); later, stack all layers and compare.

from sae import Sae  # pip install sae  (EleutherAI)

LAYER = 6  # pythia-160m has 12 layers; sweep this later
sae = Sae.load_from_hub("EleutherAI/sae-pythia-160m-32k", hookpoint=f"layers.{LAYER}.mlp")

# decoder: shape (num_latents, d_model) = (32768, 768). Rows are feature dirs.
W_dec = sae.W_dec.detach().cpu().numpy().astype(np.float64)
print(f"dictionary shape: {W_dec.shape}")

# SAE decoder rows are typically unit-normalised. Normalise anyway so the
# comparison to the unit-vector null is exact, and so PCA reads angular
# (directional) structure rather than magnitude.
W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)

n_features, d_model = W_dec.shape

# ----------------------------------------------------------------------
# 2. PCA THE REAL DICTIONARY
# ----------------------------------------------------------------------
pca_real = PCA().fit(W_dec)
var_real = pca_real.explained_variance_ratio_
cum_real = np.cumsum(var_real)

# ----------------------------------------------------------------------
# 3. THE NULL: random unit vectors, same shape
# ----------------------------------------------------------------------
# Random high-dim unit vectors are near-orthogonal -> flat, slow scree.
# This is the baseline "what does a structureless dictionary look like".
rng = np.random.default_rng(0)
R = rng.standard_normal((n_features, d_model))
R = R / np.linalg.norm(R, axis=1, keepdims=True)
pca_null = PCA().fit(R)
var_null = pca_null.explained_variance_ratio_
cum_null = np.cumsum(var_null)

# ----------------------------------------------------------------------
# 4. READ THE NUMBERS
# ----------------------------------------------------------------------
def components_for(cum, frac):
    return int(np.searchsorted(cum, frac) + 1)

for frac in (0.5, 0.9, 0.95):
    print(f"  {int(frac*100)}% variance:  real={components_for(cum_real, frac):4d}  "
          f"null={components_for(cum_null, frac):4d}  (of {d_model})")

# Effective dimensionality (participation ratio): a single number summarising
# how many directions the dictionary "really" uses. Lower than null = structure.
def participation_ratio(var):
    return (var.sum() ** 2) / (var ** 2).sum()

print(f"  participation ratio:  real={participation_ratio(var_real):.1f}  "
      f"null={participation_ratio(var_null):.1f}")

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(var_real[:100], label="real dictionary", lw=2)
ax1.plot(var_null[:100], label="random null", lw=2, ls="--")
ax1.set(xlabel="component", ylabel="explained variance ratio",
        title="Scree (first 100 components)")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(cum_real[:200], label="real dictionary", lw=2)
ax2.plot(cum_null[:200], label="random null", lw=2, ls="--")
ax2.axhline(0.9, color="grey", ls=":", lw=1)
ax2.set(xlabel="component", ylabel="cumulative variance",
        title="Cumulative variance")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sae_dictionary_pca.png", dpi=140)
print("\nsaved sae_dictionary_pca.png")

# ----------------------------------------------------------------------
# WHAT TO LOOK FOR
# ----------------------------------------------------------------------
# - Real scree drops much faster than null  -> dictionary is low-rank ->
#   there IS a structural skeleton. How many components to 90%? If ~dozens
#   rather than ~700, the "schemas as coordinate axes" story has a body.
# - Real participation ratio << null participation ratio -> same conclusion,
#   as one number.
# - If real and null sit on top of each other -> features are spread evenly
#   over the sphere, no low-dim skeleton, schemas-as-axes disconfirmed
#   (at least at this layer / this SAE's sparsity setting).
#
# NEXT STEPS (hand to Claude Code):
# - Sweep LAYER across all 12; does structure strengthen mid-network?
